import asyncio
import concurrent.futures
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
from concurrent.futures import Executor
from functools import reduce
from inspect import isfunction
from multiprocessing import Process
from pathlib import Path
from queue import Full
from time import time, sleep
from typing import List, Tuple, Dict, TypedDict

from tqdm.autonotebook import tqdm

from DQNLearner import DQNLearner
from Dadca import Dadca
from QLearning import QLearning, SparseQTable
from campaign_configuration import CampaignConfiguration, CampaignResults
from rewards import throughput_reward, delivery_reward, movement_reward, delivery_packets_reward, delivery_and_pickup, \
    unique_packets, smooth_unique_packets
from simulation import Simulation

import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration, SimulationResults
from state import MobilityState, SignedMobilityState, CommunicationMobilityState, \
    CommunicationMobilityPacketsState


def get_default_configuration() -> SimulationConfiguration:
    """
    Returns a default configuration with sensible values
    """
    return {
        'controller': DQNLearner,
        'controller_config': {
            'reward_function': smooth_unique_packets,
            'epsilon_start': 1,
            'epsilon_end': 0.1,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'memory_size': 10_000,
            'batch_size': 64,
            'hidden_layer_size': 128,
            'num_hidden_layers': 2,
            'target_network_update_rate': 'auto',
            'optimizing_rate': 10
        },
        'model_file': None,
        'state': MobilityState,
        'mission_size': 20,
        'num_agents': 2,
        'sensor_generation_frequency': 3,
        'sensor_generation_probability': 0.6,
        'sensor_packet_lifecycle': 12,
        'simulation_steps': 100_000,
        'training': True,
        'step_by_step': False,
        'plots': False,
        'verbose': True
    }


class SimulationRunner:
    def __init__(self, configuration: SimulationConfiguration):
        self.simulation = Simulation(configuration)
        self.configuration = configuration

        self.max_possible_throughput = (configuration['mission_size'] - 1) * (
                1 / configuration['sensor_generation_frequency'])
        self.expected_throughput = ((configuration['mission_size'] - 1)
                                    * (1 / configuration['sensor_generation_frequency'])
                                    * configuration['sensor_generation_probability'])

        self.throughputs: List[float] = []
        self.throughput_sum = 0
        self.agent_positions: Dict[int, List[int]] = {index: [] for index in range(configuration['num_agents'])}

    @property
    def current_step(self):
        return self.simulation.simulation_step

    def log_summarize(self):
        print(f"Maximum possible throughput {self.max_possible_throughput}")
        print(f"Expected throughput {self.expected_throughput}")
        print(f"Simulation steps: {self.simulation.simulation_step}")
        print(f"Average throughput: {self.throughput_sum / self.simulation.simulation_step}")
        print(
            f"Last throughput: {self.simulation.environment.ground_station.packets / self.simulation.simulation_step}")

    def _collect_statistics(self):
        current_throughput = self.simulation.environment.ground_station.packets / self.simulation.simulation_step

        if self.configuration['plots']:
            for index, agent in enumerate(self.simulation.environment.agents):
                self.agent_positions[index].append(agent.position)
            self.throughputs.append(current_throughput)

        self.throughput_sum += current_throughput

    def _log_step(self):
        print("---------------------")
        print(f"Ground Station: {self.simulation.environment.ground_station.packets}")
        print(
            f"Sensors: [{', '.join(str(sensor.count_update_packets(self.simulation.simulation_step, self.configuration['sensor_packet_lifecycle'])) for sensor in self.simulation.environment.sensors)}]")
        agent_string = ""
        for i in range(self.configuration['mission_size']):
            agent_string += "-("
            agent_string += ", ".join(f"{index}[{agent.packets}]"
                                      for index, agent in enumerate(self.simulation.environment.agents) if
                                      agent.position == i)
            agent_string += ")-"
        print(f"Agents: [{agent_string}]")

    def step(self) -> bool:
        if self.simulation.simulation_step >= self.configuration['simulation_steps']:
            return False
        self.simulation.simulate()
        self._collect_statistics()

        if self.configuration['step_by_step']:
            self._log_step()
        return True

    def finalize(self) -> SimulationResults:
        controller_results = self.simulation.controller.finalize()

        if self.configuration['plots']:
            self.plot_results()

        return {
            'max_possible_throughput': self.max_possible_throughput,
            'expected_throughput': self.expected_throughput,
            'avg_throughput': self.throughput_sum / self.simulation.simulation_step,
            'controller': controller_results
        }

    def plot_results(self):
        sns.lineplot(data=self.throughputs).set(title='Throughput')
        plt.show()

        plt.figure(figsize=(80, 8))
        sns.lineplot(data=self.agent_positions)
        sns.lineplot().set(title='Agent Positions', ylim=(0, self.configuration['mission_size'] - 1))
        plt.show()


class CampaignManager:
    results: List[CampaignResults]
    """ List of results from simulations triggered by running campaigns """

    result_file: Path

    process_pool: Executor
    """ Process pool responsible for executing tasks triggered during campaings """

    max_processes: int
    """ Maximum number of processes in the process pool """

    process_ids: List[int]
    """ Ids of known worker processes from the pool """

    process_manager: multiprocessing.Manager
    """ Class responsible for managing shared memory between processes """

    class CampaignSimulationStatus(TypedDict):
        routine_id: int
        step: int
        total: int
        description: str

    process_dict: Dict[int, CampaignSimulationStatus]
    """ Dictionary containing the status of each working process in the pool """

    status_update_rate: int = 1

    def __init__(self, file: Path, max_processes: int = 1):
        self.results = []
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)
        self.max_processes = max_processes
        self.process_ids = []
        self.process_manager = multiprocessing.Manager()
        self.process_dict: Dict[int, CampaignManager.CampaignSimulationStatus] = self.process_manager.dict()
        self.permutation_dict: Dict[int, bool] = self.process_manager.dict()
        self.result_file = file

    @staticmethod
    def _monitor_campaign_progress(max_processes: int,
                                   num_permutations: int,
                                   process_dict: Dict[int, CampaignSimulationStatus],
                                   permutation_dict: Dict[int, bool]):
        # Instantiating a progress bar for each process. Progress bar zero represent the main process
        progress_bars = [tqdm(position=i + 1,
                              bar_format='{desc:<50.50}{percentage:3.0f}%|{bar}| {n_fmt:>8}/{total_fmt:<8} [{elapsed:>5}<{remaining:<5} - {rate_fmt:15} {postfix:15}]')
                         for i in range(max_processes + 1)]
        progress_bars[0].set_postfix_str("Main Process")
        for i in range(max_processes):
            progress_bars[i + 1].set_postfix_str(f"Process {i + 1}")

        # Main progress bar keeps track of finished permutations
        progress_bars[0].total = num_permutations
        progress_bars[0].set_description("Total campaign progress")

        # Tracks the last routine being executed for each process
        # Used to reset the tqdm progress bar when a new routine is started
        last_routine_ids: Dict[int, int] = {}

        process_ids = []

        # Looping while not all permutations are done
        while any(not status for status in permutation_dict.values()):
            # Updating main progress bar
            progress_bars[0].n = sum(status for status in permutation_dict.values())
            progress_bars[0].refresh()

            # Checking message queue for updates from the processes
            for process, status in process_dict.items():
                if process not in process_ids:
                    process_ids.append(process)

                bar_index = process_ids.index(process) + 1

                # Resetting progress bar when a new routine is started
                if process not in last_routine_ids or status['routine_id'] != last_routine_ids[process]:
                    progress_bars[bar_index].start_t = time()
                    progress_bars[bar_index].total = status['total']
                    progress_bars[bar_index].set_description(status['description'])

                    last_routine_ids[process] = status['routine_id']

                progress_bars[bar_index].n = status['step']
                progress_bars[bar_index].refresh()

            sleep(CampaignManager.status_update_rate)
        progress_bars[0].n = num_permutations
        progress_bars[0].refresh()

        print("\n" * (max_processes + 1))

    @staticmethod
    def _run_partial_simulation(simulation: SimulationRunner,
                                steps: int,
                                message: str,
                                shared: Dict[int, CampaignSimulationStatus]) -> SimulationRunner:
        pid = os.getpid()

        # Introducing a new status
        shared[pid] = {
            "routine_id": 0 if pid not in shared else shared[pid]['routine_id'] + 1,
            "step": 0,
            "total": steps,
            "description": message
        }
        last_time = time()
        for i in range(steps):
            simulation.step()

            # Updating status at the campaigns update rate
            if (current_time := time()) - last_time >= CampaignManager.status_update_rate:
                temp = shared[pid]
                temp.update({'step': i})
                shared[pid] = temp
                last_time = current_time

        # Finalizing status
        temp = shared[pid]
        temp.update({'step': steps})
        shared[pid] = temp
        return simulation

    async def _run_permutation(self, argument: Tuple[int, dict], configuration: CampaignConfiguration):
        """
        Asynchronously runs a permutation of configuration parameters. Every simulation this permutation
        requires will be submitted to the process queue
        :param argument: Specific permutation being run in this call
        :param configuration: Campaign configuration
        """
        index, permutation = argument

        permutation_results_path = Path(f"./{index}.json")
        config: SimulationConfiguration = {**get_default_configuration(), **permutation,
                                           'model_file': permutation_results_path, 'verbose': False}

        train_config: SimulationConfiguration = config.copy()
        train_config['training'] = True
        train_config['simulation_steps'] = configuration['training_steps']

        test_config: SimulationConfiguration = config.copy()
        test_config['training'] = False
        test_config['simulation_steps'] = configuration['testing_steps']

        training_simulation = SimulationRunner(train_config)
        steps = 0
        while steps < configuration['training_steps']:
            num_steps = min(configuration['live_testing_frequency'], configuration['training_steps'])
            future = self.process_pool.submit(CampaignManager._run_partial_simulation,
                                              training_simulation,
                                              num_steps,
                                              f'Permutation {index} - Training ({steps}-{steps + num_steps}/{configuration["training_steps"]})',
                                              self.process_dict)
            training_simulation = await asyncio.wrap_future(future)
            steps += num_steps

            # Forcing output file generation
            training_simulation.simulation.controller.finalize()

            test_futures = []
            test_results = []
            for i in range(configuration['testing_repetitions']):
                testing_simulation = SimulationRunner(test_config)
                future = self.process_pool.submit(CampaignManager._run_partial_simulation,
                                                  testing_simulation,
                                                  configuration['testing_steps'],
                                                  f'Permutation {index} - Testing ({steps}/{configuration["training_steps"]})',
                                                  self.process_dict)

                if not configuration['concurrent_testing']:
                    simulation = await asyncio.wrap_future(future)
                    test_results.append(simulation.finalize())
                else:
                    test_futures.append(asyncio.wrap_future(future))

            if configuration['concurrent_testing']:
                test_results = [simulation.finalize() for simulation in (await asyncio.gather(*test_futures))]
            self.results.extend({
                                    'simulation_config': test_config,
                                    'campaign_config': configuration,
                                    'simulation_results': result,
                                    'completed_training_steps': steps
                                } for result in test_results)
            self._persist_results()

        training_results = training_simulation.finalize()
        permutation_results_path.unlink(missing_ok=True)
        self.results.append({
            'simulation_config': train_config,
            'campaign_config': configuration,
            'simulation_results': training_results,
            'completed_training_steps': configuration['training_steps']
        })
        self.permutation_dict[index] = True
        self._persist_results()

    async def run_campaign(self,
                           inputs: dict,
                           variable_keys: List[str],
                           campaign_configuration: CampaignConfiguration):
        """
        Runs a simulation campaign. A campaign is composed by the product of all value variations of the variable keys.
        Permutations are run on the process pool
        :param inputs: This is a dictionary specifying keys in the simulation configuration. Keys that are not in
        "variable_keys" will be used in all permutations. Keys that are in "variable_keys" must be lists
        :param variable_keys: Denotes that a key in the inputs dictionary is variable
        :param campaign_configuration: Campaign settings
        """
        # region Mapping campaign permutations
        # List of mutable campaign variables, making sure to shuffle values so the product is randomly ordered
        value_ranges = [(key, random.sample(inputs[key], len(inputs[key]))) for key in variable_keys]
        permutations = itertools.product(*[value_range[1] for value_range in value_ranges])

        num_permutations = reduce(lambda a, b: a * b, (len(value) for _key, value in value_ranges))
        print(f"Running {num_permutations} total permutations \n\n")

        fixed_values = {
            key: value for key, value in inputs.items() if key not in variable_keys
        }

        mapped_permutations = list(
            map(lambda p: {**fixed_values, **{value_ranges[index][0]: value for index, value in enumerate(p)}},
                permutations))

        # endregion

        # region Asynchronously running permutations
        loop = asyncio.get_running_loop()
        permutation_futures: List[asyncio.Future] = []
        for index, permutation in enumerate(mapped_permutations):
            self.permutation_dict[index] = False

        # Starting monitoring process
        monitoring_process = Process(target=CampaignManager._monitor_campaign_progress, args=(self.max_processes,
                                                                                              num_permutations,
                                                                                              self.process_dict,
                                                                                              self.permutation_dict))
        monitoring_process.start()

        for index, permutation in enumerate(mapped_permutations):
            future = loop.create_task(self._run_permutation((index, permutation), campaign_configuration))
            permutation_futures.append(future)

            if not campaign_configuration['concurrent_simulations'] \
                    and len([future for future in permutation_futures if not future.done()]) >= self.max_processes:
                await asyncio.wait(permutation_futures, return_when=asyncio.FIRST_EXCEPTION)

            for future in permutation_futures:
                if future.done() and future.exception() is not None:
                    monitoring_process.kill()
                    raise future.exception()

        await asyncio.wait(permutation_futures, return_when=asyncio.FIRST_EXCEPTION)
        # endregion

        print("-------- Campaign ended --------\n")

    def _persist_results(self):
        with self.result_file.open("w") as file:
            file.write(json.dumps(self.results, default=lambda x: str(x) if not isfunction(x) else x.__name__))

    def finalize(self):
        self._persist_results()

        self.process_pool.shutdown()
        self.process_manager.shutdown()
