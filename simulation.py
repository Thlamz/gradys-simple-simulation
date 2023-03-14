import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from Dadca import Dadca
from QLearning import QLearning
from controller import Controller
from environment import State, Control, validate_control, execute_control, MobilityCommand
from node import Node
from simulation_parameters import MISSION_SIZE, NUM_AGENTS, SENSOR_GENERATION_FREQUENCY, MAXIMUM_SIMULATION_STEPS


class SimulationException(Exception):
    pass


class Simulation:
    controller: Controller
    ground_station: Node
    agents: list[Node]
    sensors: list[Node]

    X: State
    U: Control

    simulation_step: int

    def __init__(self, controller):
        self.controller = controller
        self.simulation_step = 0

        self.ground_station = {"packets": 0}
        self.agents = []
        self.sensors = []
        for _ in range(1, MISSION_SIZE):
            self.sensors.append({"packets": 0})

        for _ in range(NUM_AGENTS):
            self.agents.append({"packets": 0})

        self.X = State(mobility=tuple(0 for _ in range(NUM_AGENTS)))
        self.U = Control(mobility=tuple(MobilityCommand.FORWARDS for _ in range(NUM_AGENTS)))

    def simulate(self):
        self.U = self.controller.get_control(self.simulation_step,
                                             self.X,
                                             self.U,
                                             self.ground_station,
                                             self.agents,
                                             self.sensors)
        if not validate_control(self.X, self.U):
            raise SimulationException("Invalid control")

        self.X = execute_control(self.X, self.U)

        # Simulating sensor packet generation
        if self.simulation_step % SENSOR_GENERATION_FREQUENCY == 0:
            for sensor in self.sensors:
                sensor['packets'] += 1

        # Simulate message exchange
        for index1, agent1 in enumerate(self.agents):
            if self.U.mobility[index1] == MobilityCommand.FORWARDS:
                continue

            for index2, agent2 in enumerate(self.agents):
                if self.U.mobility[index2] == MobilityCommand.REVERSE \
                        or self.X.mobility[index1] > self.X.mobility[index2]:
                    continue

                if self.X.mobility[index2] <= (self.X.mobility[index1] + 2):
                    agent1['packets'] += agent2['packets']
                    agent2['packets'] = 0

        # Simulating sensor packet pickup
        for index, agent in enumerate(self.agents):
            agent_mobility = self.X.mobility[index]
            if agent_mobility == 0:
                self.ground_station['packets'] += agent['packets']
                agent['packets'] = 0
            else:
                agent['packets'] += self.sensors[agent_mobility - 1]['packets']
                self.sensors[agent_mobility - 1]['packets'] = 0

        self.simulation_step += 1
