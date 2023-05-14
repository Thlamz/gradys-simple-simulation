import itertools
import json
import math
import multiprocessing
import os
import random
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DQNLearner import DQNLearner
from Dadca import Dadca
from QLearning import QLearning, SparseQTable
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


def run_simulation(configuration: SimulationConfiguration) -> SimulationResults:
    """
    Runs a single simulation with a specified configuration. This function's main objective is to run the simulation
    steps and record telemetry about them.
    :param configuration: Configuration to run the simulation with
    :return: A summary of the simulation's results
    """
    simulation = Simulation(configuration)

    throughputs = []
    throughput_sum = 0
    agent_positions = {index: [] for index in range(configuration['num_agents'])}

    max_possible_throughput = (configuration['mission_size'] - 1) * (1 / configuration['sensor_generation_frequency'])
    expected_throughput = ((configuration['mission_size'] - 1)
                           * (1 / configuration['sensor_generation_frequency'])
                           * configuration['sensor_generation_probability'])
    if configuration['verbose']:
        print(f"Maximum possible throughput {max_possible_throughput}")
        print(f"Expected throughput {expected_throughput}")

    iterator = range(configuration['maximum_simulation_steps'])
    if not configuration['step_by_step'] and configuration['verbose']:
        iterator = tqdm(iterator)

    # Simulation iterations
    for _ in iterator:
        # Step by step visualization of simulation state
        if configuration['step_by_step']:
            print("---------------------")
            print(f"Ground Station: {simulation.environment.ground_station.packets}")
            print(
                f"Sensors: [{', '.join(str(sensor.count_update_packets(simulation.simulation_step, configuration['sensor_packet_lifecycle'])) for sensor in simulation.environment.sensors)}]")
            agent_string = ""
            for i in range(configuration['mission_size']):
                agent_string += "-("
                agent_string += ", ".join(f"{index}[{agent.packets}]"
                                          for index, agent in enumerate(simulation.environment.agents) if
                                          agent.position == i)
                agent_string += ")-"
            print(f"Agents: [{agent_string}]")

            input()
        simulation.simulate()

        throughput_sum += simulation.environment.ground_station.packets / simulation.simulation_step
        if configuration['plots']:
            for index, agent in enumerate(simulation.environment.agents):
                agent_positions[index].append(agent.position)

            throughputs.append(simulation.environment.ground_station.packets / simulation.simulation_step)

    controller_results = simulation.controller.finalize()

    if configuration['plots']:
        sns.lineplot(data=throughputs).set(title='Throughput')
        plt.show()

        plt.figure(figsize=(80, 8))
        sns.lineplot(data=agent_positions)
        sns.lineplot().set(title='Agent Positions', ylim=(0, configuration['mission_size'] - 1))
        plt.show()

    if configuration['verbose']:
        print(f"Simulation steps: {simulation.simulation_step}")
        print(f"Average throughput: {throughput_sum / simulation.simulation_step}")
        print(f"Last throughput: {simulation.environment.ground_station.packets / simulation.simulation_step}")
    return {
        'max_possible_throughput': max_possible_throughput,
        'expected_throughput': expected_throughput,
        'avg_throughput': throughput_sum / simulation.simulation_step,
        'config': {
            key: str(value) for key, value in configuration.items()
        },
        'controller': controller_results
    }


def _run_permutation(argument: Tuple[int, dict]) -> List[SimulationResults]:
    """
    Runs a permutation of configuration parameters. This auxiliary function is used
    to run simulation campaigns
    :param argument: Specific permutation being run in this call
    :return: List of simulation results generated by this permutation. This is a list because
    both training and testing are recorded for QLearning
    """
    index, permutation = argument

    q_table_path = Path(f"./{index}.json")
    config = {
        **get_default_configuration(),
        **permutation,
        'model_file': q_table_path
    }
    training_time = 0
    results = []
    while training_time < config.get('target_total_training_steps', 1):
        results.append(run_simulation(config))
        training_time += config['maximum_simulation_steps']

    if config['controller'] != Dadca:
        for _ in range(config['testing_repetitions']):
            test_config = config.copy()
            test_config['training'] = False
            test_config['maximum_simulation_steps'] = 10000
            test_results = run_simulation(test_config)

            # Making sure the steps reported are related to the training not testing
            test_results['config']['maximum_simulation_steps'] = str(config['maximum_simulation_steps'])
            results.append(test_results)
        os.unlink(q_table_path)

    return results


def run_campaign(inputs: dict,
                 variable_keys: List[str],
                 multi_processing: bool = False,
                 max_processes: int = None,
                 results_file: Path = Path("./analysis/results.json")):
    """
    Runs a simulation campaign. A campaign is composed by the product of all value variations of the variable keys.
    Simulation results are recorded in a results.json file in the analysis folder.
    :param inputs: This is a dictionary specifying keys in the simulation configuration. Keys that are not in
    "variable_keys" will be used in all permutations. Keys that are in "variable_keys" must be lists
    :param variable_keys: Denotes that a key in the inputs dictionary is variable
    :param multi_processing: Enable multiprocessing
    :param max_processes: Maximum number of processes to use
    :param results_file: File where the simulation results will be saved
    """
    # List of mutable campaign variables, making sure to shuffle values so the product is randomly ordered
    value_ranges = [(key, random.sample(inputs[key], len(inputs[key]))) for key in variable_keys]
    permutations = itertools.product(*[value_range[1] for value_range in value_ranges])

    fixed_values = {
        key: value for key, value in inputs.items() if key not in variable_keys
    }

    if multi_processing:
        fixed_values['verbose'] = False

    num_permutations = reduce(lambda a, b: a * b, (len(value) for _key, value in value_ranges))
    print(f"Running {num_permutations} total permutations \n\n")

    mapped_permutations = list(
        map(lambda p: {**fixed_values, **{value_ranges[index][0]: value for index, value in enumerate(p)}},
            permutations))

    if multi_processing:
        small_permutations = [permutation
                              for permutation in mapped_permutations
                              if permutation['maximum_simulation_steps'] <= 1_000_000]
        print(f"Running {len(small_permutations)} smaller simulations in parallel\n")
        small_results = list(process_map(_run_permutation,
                                         enumerate(small_permutations),
                                         max_workers=min(max_processes or 8, 8),
                                         total=len(small_permutations), ))

        medium_permutations = [permutation
                               for permutation in mapped_permutations
                               if 1_000_000 < permutation['maximum_simulation_steps'] <= 5_000_000]
        print(f"\nRunning {len(medium_permutations)} medium simulations in 3 worker parallel\n")
        medium_results = list(process_map(_run_permutation,
                                          enumerate(medium_permutations),
                                          max_workers=min(max_processes or 3, 3),
                                          total=len(medium_permutations)))

        large_permutations = [permutation
                              for permutation in mapped_permutations
                              if permutation['maximum_simulation_steps'] > 5_000_000]
        print(f"\nRunning {len(large_permutations)} larger simulations synchronously\n")
        big_results = list(tqdm(map(_run_permutation, enumerate(large_permutations)), total=len(large_permutations)))

        results = small_results + medium_results + big_results
    else:
        results = list(map(_run_permutation, enumerate(mapped_permutations)))

    campaign = {
        'campaign_variables': variable_keys + ['training'],
        'results': list(itertools.chain(*results))
    }

    with open(results_file, "w") as file:
        file.write(json.dumps(campaign, default=lambda x: None))


if __name__ == '__main__':
    # controller_config_permutation_dict = {
    #     'reward_function': [unique_packets],
    #     'epsilon_start': [1],
    #     'epsilon_end': [0.1],
    #     'learning_rate': [0.0005],
    #     'gamma': [0.99],
    #     'memory_size': [10_000],
    #     'batch_size': [64],
    #     'hidden_layer_size': [64],
    #     'num_hidden_layers': [2],
    #     'target_network_update_rate': [100, 'auto']
    # }
    # keys, values = zip(*controller_config_permutation_dict.items())
    # controller_config_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    run_campaign({
        'num_agents': [1, 2],
        'mission_size': [10, 25, 30],
        'sensor_generation_probability': 0.1,
        'sensor_packet_lifecycle': math.inf,
        'controller': DQNLearner,
        'controller_config': {
            'reward_function': unique_packets,
            'epsilon_start': 1,
            'epsilon_end': 0.1,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'memory_size': 10_000,
            'batch_size': 64,
            'hidden_layer_size': 64,
            'num_hidden_layers': 2,
            'target_network_update_rate': 'auto'
        },
        'state': CommunicationMobilityPacketsState,
        'testing_repetitions': 1,
        'maximum_simulation_steps': [int(x) for x in np.linspace(10_000, 100_000, 5)],
        'repetitions': [1, 2, 3],
    }, ['maximum_simulation_steps', 'repetitions', 'num_agents', 'mission_size'], multi_processing=True, max_processes=1)
