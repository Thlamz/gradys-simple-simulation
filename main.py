import itertools
import json
import multiprocessing
import tracemalloc

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from Dadca import Dadca
from QLearning import QLearning
from simulation import Simulation

import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration, SimulationResults


def get_default_configuration() -> SimulationConfiguration:
    return {
        'controller': QLearning,
        'mission_size': 50,
        'num_agents': 2,
        'sensor_generation_frequency': 3,
        'maximum_simulation_steps': 1_000_000,
        'epsilon_start': 1.,
        'epsilon_end': 0.001,
        'learning_rate': 0.1,
        'gamma': 0.99,
        'qtable_initialization_value': 0,
        'qtable_file': None,
        'qtable_format': 'sparse',
        'cache_optimal_control': True,
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

    if configuration['verbose']:
        print(f"Maximum possible throughput {(configuration['mission_size'] - 1) * (1 / configuration['sensor_generation_frequency'])}")

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

        plt.figure(figsize=(500, 8))
        sns.lineplot(data=agent_positions)
        sns.lineplot().set(title='Agent Positions')
        plt.show()

    if configuration['verbose']:
        print(f"Simulation steps: {simulation.simulation_step}")
        print(f"Average throughput: {throughput_sum / simulation.simulation_step}")
        print(f"Last throughput: {simulation.ground_station['packets'] / simulation.simulation_step}")
    return {
        'avg_throughput': throughput_sum / simulation.simulation_step
    }


def run_permutation(argument):
    index, permutation = argument
    print(f"Running permutation {index} - {permutation}")

    q_table_format, q_table_cache = permutation
    config = get_default_configuration()
    config['qtable_format'] = q_table_format
    config['cache_optimal_control'] = q_table_cache
    config['controller'] = QLearning

    results = run_simulation(config)
    print(f"Finished running permutation {index}")
    return config, results


if __name__ == '__main__':
    q_table_formats = ['sparse', 'dense']
    q_table_caches = [True, False]

    permutations = itertools.product(q_table_formats, q_table_caches)

    print(f"Running {len(q_table_formats) * len(q_table_caches)} permutations")
    result = list(map(run_permutation, enumerate(permutations)))

    print(result)
    with open("./result.txt", "w") as file:
        file.write(json.dumps(result, indent=2, default=lambda x: None))


