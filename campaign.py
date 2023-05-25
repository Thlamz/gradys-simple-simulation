import asyncio
import concurrent.futures
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
from functools import reduce
from inspect import isfunction
from pathlib import Path
from time import time
from typing import List, Tuple, Optional, Dict, TypedDict

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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
        'controller': QLearning,
        'controller_config': {
            'epsilon_start': 1.,
            'epsilon_end': 0.001,
            'learning_rate': 0.1,
            'gamma': 0.99,
            'reward_function': unique_packets,
            'qtable_initialization_value': 0,
            'qtable_format': SparseQTable,
        },
        'model_file': None,
        'state': MobilityState,
        'mission_size': 20,
        'num_agents': 2,
        'sensor_generation_frequency': 3,
        'sensor_generation_probability': 0.6,
        'sensor_packet_lifecycle': 12,
        'maximum_simulation_steps': 1_000_000,
        'training': True,
        'testing_repetitions': 1,
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
        self.agent_positions: Dict[int, List[int]] = {index: [] for index in range(configuration['num_agents'])}

    @property
    def current_step(self):
        return self.simulation.simulation_step

    def log_summarize(self):
        print(f"Maximum possible throughput {self.max_possible_throughput}")
        print(f"Expected throughput {self.expected_throughput}")
        print(f"Simulation steps: {self.simulation.simulation_step}")
        print(f"Average throughput: {sum(self.throughputs) / self.simulation.simulation_step}")
        print(
            f"Last throughput: {self.simulation.environment.ground_station.packets / self.simulation.simulation_step}")

    def _collect_statistics(self):
        for index, agent in enumerate(self.simulation.environment.agents):
            self.agent_positions[index].append(agent.position)

        self.throughputs.append(self.simulation.environment.ground_station.packets / self.simulation.simulation_step)

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

        return {
            'max_possible_throughput': self.max_possible_throughput,
            'expected_throughput': self.expected_throughput,
            'avg_throughput': sum(self.throughputs) / self.simulation.simulation_step,
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

    class CampaignSimulationMessage(TypedDict):
        process: int
        reset: bool
        step: int
        total: int
        description: str

    def __init__(self, max_processes: int = 1):
        self.results = []
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)
        self.max_processes = max_processes
        self.process_ids = []
        self.process_manager = multiprocessing.Manager()
        self.message_queue = self.process_manager.Queue()

    @staticmethod
    def _run_partial_simulation(simulation: SimulationRunner,
                                steps: int,
                                message: str,
                                queue: multiprocessing.Queue) -> SimulationRunner:
        queue.put({
            "process": os.getpid(),
            "reset": True,
            "step": 0,
            "total": steps,
            "description": message
        })
        for i in range(steps):
            simulation.step()
            queue.put({
                "process": os.getpid(),
                "reset": False,
                "step": i,
                "total": steps,
                "description": message
            })
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
                                              self.message_queue)
            training_simulation = await asyncio.wrap_future(future)
            steps += num_steps

            # Forcing output file generation
            training_simulation.simulation.controller.finalize()

            test_futures = []
            for i in range(configuration['testing_repetitions']):
                testing_simulation = SimulationRunner(test_config)
                future = self.process_pool.submit(CampaignManager._run_partial_simulation,
                                                  testing_simulation,
                                                  configuration['testing_steps'],
                                                  f'Permutation {index} - Testing ({steps}/{configuration["training_steps"]})',
                                                  self.message_queue)
                test_futures.append(asyncio.wrap_future(future))

            testing_simulations = await asyncio.gather(*test_futures)
            self.results.extend({
                'simulation_config': test_config,
                'campaign_config': configuration,
                'simulation_results': testing_simulation.finalize(),
                'completed_training_steps': steps
            } for testing_simulation in testing_simulations)

        training_results = training_simulation.finalize()
        permutation_results_path.unlink(missing_ok=True)
        self.results.append({
            'simulation_config': train_config,
            'campaign_config': configuration,
            'simulation_results': training_results,
            'completed_training_steps': configuration['training_steps']
        })

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
            future = loop.create_task(self._run_permutation((index, permutation), campaign_configuration))
            permutation_futures.append(future)
        # endregion

        # region Monitoring processes

        # Instantiating a progress bar for each process. Progress bar zero represent the main process
        progress_bars = [tqdm(position=i + 1) for i in range(self.max_processes + 1)]
        progress_bars[0].set_postfix_str("Main Process")
        for i in range(self.max_processes):
            progress_bars[i + 1].set_postfix_str(f"Process {i + 1}")

        # Main progress bar keeps track of finished permutations
        progress_bars[0].total = num_permutations
        progress_bars[0].set_description("Total campaign progress")

        # Looping while not all permutations are done
        while any(not future.done() for future in permutation_futures):
            # Stopping campaign early if any exception is raised
            for future in permutation_futures:
                if future.done() and (exc := future.exception()) is not None:
                    raise exc

            # Updating main progress bar
            progress_bars[0].n = sum(1 for future in permutation_futures if future.done())
            progress_bars[0].refresh()

            # Checking message queue for updates from the processes
            while not self.message_queue.empty():
                message: CampaignManager.CampaignSimulationMessage = self.message_queue.get()
                if message['process'] not in self.process_ids:
                    self.process_ids.append(message['process'])
                bar_index = self.process_ids.index(message['process']) + 1

                # Resetting progress bar when step doesn't progress
                if message['reset']:
                    progress_bars[bar_index].start_t = time()
                    progress_bars[bar_index].total = message['total']

                progress_bars[bar_index].n = message['step']
                progress_bars[bar_index].set_description(message['description'])

            await asyncio.sleep(1)
        progress_bars[0].n = sum(1 for future in permutation_futures if future.done())
        progress_bars[0].refresh()
        # endregion

    def finalize(self, file: Path):
        with file.open("w") as file:
            file.write(json.dumps(self.results, default=lambda x: str(x) if not isfunction(x) else x.__name__))

        self.process_pool.shutdown()
        self.process_manager.shutdown()