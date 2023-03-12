import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Dadca import Dadca
from QLearning import QLearning
from controller import Controller
from simulation import Simulation
from simulation_parameters import MAXIMUM_SIMULATION_STEPS, MISSION_SIZE, NUM_AGENTS

import seaborn as sns
import matplotlib.pyplot as plt


def run_simulation(controller: Controller, step_by_step: bool = False, plots: bool = False):
    simulation = Simulation(controller)

    throughputs = []
    agent_positions = {index: [] for index in range(NUM_AGENTS)}

    iterator = range(MAXIMUM_SIMULATION_STEPS)
    if not step_by_step:
        iterator = tqdm(iterator)

    # Simulation iterations
    for _ in iterator:
        # Step by step visualization of simulation state
        if step_by_step:
            print("---------------------")
            print(f"Ground Station: {simulation.ground_station['packets']}")
            print(f"Sensors: [{', '.join(str(sensor['packets']) for sensor in simulation.sensors)}]")
            agent_string = ""
            for i in range(MISSION_SIZE):
                agent_string += "-("
                agent_string += ", ".join(f"{index}[{agent['packets']}]"
                                          for index, agent in enumerate(simulation.agents) if
                                          simulation.X.mobility[index] == i)
                agent_string += ")-"
            print(f"Agents: [{agent_string}]")

            input()
        simulation.simulate()

        for index, position in enumerate(simulation.X.mobility):
            agent_positions[index].append(position)

        throughputs.append(simulation.ground_station['packets'] / simulation.simulation_step)

    controller.finalize()

    if plots:
        sns.lineplot(data=throughputs).set(title='Throughput')
        plt.show()

        # plt.figure(figsize=(500, 8))
        # sns.lineplot(data=agent_positions)
        # sns.lineplot().set(title='Agent Positions')
        # plt.show()

    print(f"Simulation steps: {simulation.simulation_step}")
    print(f"Average throughput: {sum(throughputs) / len(throughputs)}")
    print(f"Maximum throughput: {max(throughputs)}")
    print(f"Last throughput: {throughputs[-1]}")


if __name__ == '__main__':
    print("Simulating DADCA:")
    run_simulation(Dadca(), plots=False)

    print(f"\n\n\nTraining QLearning")
    run_simulation(QLearning(qtable_file=Path("./qtable.npy"), learning_rate=0.9), plots=False)

    print("\n\n\nTesting QLearning")
    run_simulation(QLearning(qtable_file=Path("./qtable.npy"), training=False), plots=False)
    os.remove("./qtable.npy")