import itertools
import json
import multiprocessing
import os
from functools import reduce
from pathlib import Path

from tqdm import tqdm

from Dadca import Dadca
from QLearning import QLearning
from simulation import Simulation

import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration, SimulationResults


def get_default_configuration() -> SimulationConfiguration:
    return {
        'controller': QLearning,
        'mission_size': 20,
        'num_agents': 2,
        'sensor_generation_frequency': 3,
        'sensor_generation_probability': 0.6,
        'maximum_simulation_steps': 1_000_000,
        'epsilon_start': 1.,
        'epsilon_end': 0.001,
        'learning_rate': 0.9,
        'gamma': 0.99,
        'qtable_initialization_value': 0,
        'qtable_file': None,
        'qtable_format': 'sparse',
        'training': True,
        'step_by_step': False,
        'plots': False,
        'verbose': True
    }


def run_simulation(configuration: SimulationConfiguration) -> SimulationResults:
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
            print(f"Ground Station: {simulation.ground_station['packets']}")
            print(f"Sensors: [{', '.join(str(sensor['packets']) for sensor in simulation.sensors)}]")
            agent_string = ""
            for i in range(configuration['mission_size']):
                agent_string += "-("
                agent_string += ", ".join(f"{index}[{agent['packets']}]"
                                          for index, agent in enumerate(simulation.agents) if
                                          simulation.X.mobility[index] == i)
                agent_string += ")-"
            print(f"Agents: [{agent_string}]")

            input()
        simulation.simulate()

        throughput_sum += simulation.ground_station['packets'] / simulation.simulation_step
        if configuration['plots']:
            for index, position in enumerate(simulation.X.mobility):
                agent_positions[index].append(position)

            throughputs.append(simulation.ground_station['packets'] / simulation.simulation_step)

    simulation.controller.finalize()

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
        print(f"Last throughput: {simulation.ground_station['packets'] / simulation.simulation_step}")
    return {
        'max_possible_throughput': max_possible_throughput,
        'expected_throughput': expected_throughput,
        'avg_throughput': throughput_sum / simulation.simulation_step,
        'config': {
            key: str(value) for key, value in configuration.items()
        }
    }


def run_permutation(argument):
    index, permutation = argument
    print(f"Running permutation {index} - {permutation}")

    q_table_path = Path(f"./{index}.pickle")
    config = {
        **get_default_configuration(),
        **permutation,
        'qtable_file': q_table_path
    }

    run_simulation(config)
    if config['controller'] == QLearning:
        print(f"Running testing for permutation {index} - {permutation}")
        config['training'] = False
        results = run_simulation(config)
        os.unlink(q_table_path)

    print(f"Finished running permutation {index}\n\n")
    return results


def run_campaign(inputs: dict, variable_keys: list[str], multi_processing: bool = False, max_processes: int = 4):
    value_ranges = [(key, inputs[key]) for key in variable_keys]
    permutations = itertools.product(*[value_range[1] for value_range in value_ranges])

    fixed_values = {
        key: value for key, value in inputs.items() if key not in variable_keys
    }

    if multi_processing:
        fixed_values['verbose'] = False

    print(f"Running {reduce(lambda a, b: a * b, (len(value) for _key, value in value_ranges))} total permutations \n\n")

    mapped_permutations = \
        map(lambda p: fixed_values | {value_ranges[index][0]: value for index, value in enumerate(p)}, permutations)

    if multi_processing:
        with multiprocessing.Pool(processes=max_processes) as pool:
            result = list(pool.map(run_permutation, enumerate(mapped_permutations)))
    else:
        result = list(map(run_permutation, enumerate(mapped_permutations)))

    campaign = {
        'campaign_variables': variable_keys,
        'results': result
    }

    with open("analysis/result.json", "w") as file:
        file.write(json.dumps(campaign, indent=2, default=lambda x: None))


if __name__ == '__main__':
    run_campaign({
        'num_agents': [2],
        'mission_size': 140,
        'sensor_generation_probability': [1],
        'controller': [QLearning],
        'qtable_format': ['sparse'],
        'learning_rate': 0.9,
        'maximum_simulation_steps': 10_000_000,
    }, ['num_agents', 'sensor_generation_probability', 'controller', 'qtable_format'])
