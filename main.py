import itertools
import json
import math
import multiprocessing
import os
import tracemalloc
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from Dadca import Dadca
from QLearning import QLearning
from rewards import throughput_reward, delivery_reward, movement_reward
from simulation import Simulation

import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration, SimulationResults
from state import MobilityState, SignedMobilityState


def get_default_configuration() -> SimulationConfiguration:
    """
    Returns a default configuration with sensible values
    """
    return {
        'controller': QLearning,
        'state': MobilityState,
        'mission_size': 20,
        'num_agents': 2,
        'sensor_generation_frequency': 3,
        'sensor_generation_probability': 0.6,
        'sensor_packet_lifecycle': 12,
        'maximum_simulation_steps': 1_000_000,
        'epsilon_start': 1.,
        'epsilon_end': 0.001,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'reward_function': throughput_reward,
        'qtable_initialization_value': 0,
        'qtable_file': None,
        'qtable_format': 'sparse',
        'training': True,
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
            print(f"Sensors: [{', '.join(str(sensor.num_packets(simulation.simulation_step, configuration['sensor_packet_lifecycle'])) for sensor in simulation.environment.sensors)}]")
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

        plt.figure(figsize=(50, 8))
        sns.lineplot(data=agent_positions)
        sns.lineplot().set(title='Agent Positions')
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
    print(f"Running permutation {index} - {permutation}")

    q_table_path = Path(f"./{index}.json")
    config = {
        **get_default_configuration(),
        **permutation,
        'qtable_file': q_table_path
    }

    results = [run_simulation(config)]
    if config['controller'] == QLearning:
        print(f"Running testing for permutation {index} - {permutation}")
        config['training'] = False
        results.append(run_simulation(config))
        os.unlink(q_table_path)

    print(f"Finished running permutation {index}\n\n")
    return results


def run_campaign(inputs: dict, variable_keys: List[str], multi_processing: bool = False, max_processes: int = None):
    """
    Runs a simulation campaign. A campaign is composed by the product of all value variations of the variable keys.
    Simulation results are recorded in a results.json file in the analysis folder.
    :param inputs: This is a dictionary specifying keys in the simulation configuration. Keys that are not in
    "variable_keys" will be used in all permutations. Keys that are in "variable_keys" must be lists
    :param variable_keys: Denotes that a key in the inputs dictionary is variable
    :param multi_processing: Enable multiprocessing
    :param max_processes: Maximum number of processes to use
    """
    value_ranges = [(key, inputs[key]) for key in variable_keys]
    permutations = itertools.product(*[value_range[1] for value_range in value_ranges])

    fixed_values = {
        key: value for key, value in inputs.items() if key not in variable_keys
    }

    if multi_processing:
        fixed_values['verbose'] = False

    print(f"Running {reduce(lambda a, b: a * b, (len(value) for _key, value in value_ranges))} total permutations \n\n")

    mapped_permutations = \
        map(lambda p: {**fixed_values, **{value_ranges[index][0]: value for index, value in enumerate(p)}}, permutations)

    if multi_processing:
        with multiprocessing.Pool(processes=max_processes) as pool:
            results = list(pool.map(_run_permutation, enumerate(mapped_permutations), chunksize=1))
    else:
        results = list(map(_run_permutation, enumerate(mapped_permutations)))

    campaign = {
        'campaign_variables': variable_keys + ['training'],
        'results': list(itertools.chain(*results))
    }

    with open("analysis/result.json", "w") as file:
        file.write(json.dumps(campaign, indent=2, default=lambda x: None))


if __name__ == '__main__':
    run_campaign({
        'num_agents': 1,
        'mission_size': [200],
        'sensor_generation_probability': 1,
        'sensor_packet_lifecycle': math.inf,
        'controller': QLearning,
        'reward_function': movement_reward,
        'state': SignedMobilityState,
        'maximum_simulation_steps': 1_000_000,
        'epsilon_end': 0.8,
        'plots': True
    }, ['mission_size'])
