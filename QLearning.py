import itertools
import math
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Literal, Union

import numpy as np
import numpy.random

from controller import Controller
from environment import State, Control, MobilityCommand, Environment
import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration
from node import Node


class QTable(ABC):
    configuration: SimulationConfiguration
    environment: Environment

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        self.configuration = configuration
        self.environment = environment

    @abstractmethod
    def initialize_q_table(self):
        pass

    @abstractmethod
    def get_q_value(self, state: State, control: Control) -> float:
        pass

    @abstractmethod
    def set_q_value(self, state: State, control: Control, q_value: float):
        pass

    @abstractmethod
    def get_optimal_control(self, state: State) -> Control:
        pass

    @abstractmethod
    def export_qtable(self):
        pass


class SparseQTable(QTable):
    optimize_cache: bool
    q_table: dict[State, dict[Control, float]]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.rng = numpy.random.default_rng()
        self.initialize_q_table()

    def load_q_table(self):
        if self.optimize_cache:
            self.q_table_optimal_cache: dict[State, Control] = {}

        with self.configuration['qtable_file'].open("rb") as file:
            self.q_table = pickle.load(file)

    def initialize_q_table(self):
        if self.configuration['qtable_file'] is not None:
            self.load_q_table()
        else:
            self.q_table = defaultdict(lambda: defaultdict(lambda: self.configuration['qtable_initialization_value']))

    def get_q_value(self, state: State, control: Control) -> float:
        return self.q_table[state][control]

    def get_optimal_control(self, state: State) -> Control:
        optimal_control = None
        optimal_q_value = -math.inf
        for control, q_value in self.q_table[state].items():
            if q_value > optimal_q_value:
                optimal_control = control
                optimal_q_value = q_value
        if optimal_control is not None:
            return optimal_control
        else:
            return self.environment.generate_random_control(state)

    def set_q_value(self, state: State, control: Control, q_value: float):
        self.q_table[state][control] = q_value

    def export_qtable(self):
        if self.configuration['qtable_file'] is not None:
            with self.configuration['qtable_file'].open("wb") as file:
                pickle.dump(self.q_table, file)


class DenseQTable(QTable):
    optimize_cache: bool
    q_table: numpy.ndarray[float]
    state_id: dict
    control_id: dict

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.rng = numpy.random.default_rng()
        self.initialize_q_table()
        self.calculate_state_ids()
        self.calculate_control_ids()

    def calculate_state_ids(self):
        self.state_id = {}
        for index, permutation in enumerate(itertools.product([i for i in range(self.configuration['mission_size'])],
                                                              repeat=self.configuration['num_agents'])):
            self.state_id[index] = State(mobility=permutation)
            self.state_id[State(mobility=permutation)] = index

    def calculate_control_ids(self):
        self.control_id = {}
        for index, permutation in enumerate(itertools.product([MobilityCommand.FORWARDS, MobilityCommand.REVERSE],
                                                              repeat=self.configuration['num_agents'])):
            self.control_id[index] = Control(mobility=permutation)
            self.control_id[Control(mobility=permutation)] = index

    def load_q_table(self):
        self.q_table = np.load(str(self.configuration['qtable_file']))

    def initialize_q_table(self):
        if self.configuration['qtable_file'] is not None:
            self.load_q_table()
        else:
            possible_states = self.configuration['mission_size'] ** self.configuration['num_agents']
            possible_actions = len(MobilityCommand) ** self.configuration['num_agents']
            self.q_table = np.full((possible_states, possible_actions),
                                   self.configuration['qtable_initialization_value'])

    def get_q_value(self, state: State, control: Control) -> float:
        return self.q_table[self.state_id[state], self.control_id[control]]

    def get_optimal_control(self, state: State) -> Control:
        if np.any(self.q_table[self.state_id[state], :] > self.configuration['qtable_initialization_value']):
            return self.control_id[
                np.argmax(self.q_table[self.state_id[state], :])
            ]
        else:
            return self.environment.generate_random_control(state)

    def set_q_value(self, state: State, control: Control, q_value: float):
        self.q_table[self.state_id[state], self.control_id[control]] = q_value

    def export_qtable(self):
        if self.configuration['qtable_file'] is not None:
            with self.configuration['qtable_file'].open("wb") as file:
                pickle.dump(self.q_table, file)


class QLearning(Controller):
    training: bool
    """ Training flag. If set to false doesn't try to update the Q Table """

    qtable_file: Optional[Path] = None
    """ Path where a Q Table pickle is located. If set will try to load the Q Table at initialization """

    # Variables updated during execution
    last_state: Optional[State]
    epsilon: float
    q_table: QTable

    # Statistic collection
    total_cost: float
    cum_avg_costs: list[float]
    epsilons: list[float]

    # Configuration variables
    epsilon_start: float
    epsilon_end: float
    epsilon_horizon: int
    learning_rate: float
    gamma: float
    qtable_initialization_value: float
    qtable_format: Literal['sparse', 'dense']
    cache_optimal_control: bool

    configuration: SimulationConfiguration

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.training = self.configuration['training']
        if self.training:
            self.epsilon = configuration['epsilon_start']
        if not self.training:
            self.epsilon = 0

        self.qtable_file = self.configuration['qtable_file']
        self.qtable_format = self.configuration['qtable_format']
        self.q_table = SparseQTable(configuration, environment) if self.qtable_format == 'sparse' else DenseQTable(configuration, environment)

        if configuration['verbose']:
            size = (len(MobilityCommand) ** self.configuration['num_agents'] *
                    self.configuration['mission_size'] ** self.configuration['num_agents'])
            print(f"QTable size: {size}")

        self.last_state = None
        self.total_cost = 0
        self.cum_avg_costs = []
        self.epsilons = []

        self.epsilon_start = self.configuration['epsilon_start']
        self.epsilon_end = self.configuration['epsilon_end']
        self.epsilon_horizon = self.configuration['maximum_simulation_steps']
        self.learning_rate = self.configuration['learning_rate']
        self.gamma = self.configuration['gamma']
        self.qtable_initialization_value = self.configuration['qtable_initialization_value']
        self.rng = numpy.random.default_rng()

    def decay_epsilon(self) -> None:
        if self.epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon *= (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_horizon)

        if self.configuration['plots']:
            self.epsilons.append(self.epsilon)

    def compute_cost(self,
                     simulation_step: int,
                     state: State,
                     ground_station: Node,
                     agents: list[Node],
                     sensors: list[Node]):
        if simulation_step == 0:
            return 0
        highest_throughput = (self.configuration['mission_size'] - 1) * (
                    simulation_step / self.configuration['sensor_generation_frequency'])
        return -ground_station['packets'] / highest_throughput

    def get_control(self,
                    simulation_step: int,
                    current_state: State,
                    current_control: Control,
                    ground_station: Node,
                    agents: list[Node],
                    sensors: list[Node]) -> Control:
        cost = self.compute_cost(simulation_step, current_state, ground_station, agents, sensors)
        self.total_cost += cost
        if simulation_step > 0 and self.configuration['plots']:
            self.cum_avg_costs.append(self.total_cost / simulation_step)

        if self.last_state is not None and self.training:
            reward = -cost

            previous_qvalue = self.q_table.get_q_value(self.last_state, current_control)
            next_state_qvalue = self.q_table.get_q_value(current_state, self.q_table.get_optimal_control(current_state))

            q_value = ((1 - self.learning_rate) * previous_qvalue
                       + self.learning_rate * (reward + self.gamma * next_state_qvalue))

            self.q_table.set_q_value(self.last_state, current_control, q_value)

            self.decay_epsilon()

        if self.rng.uniform(0, 1) < self.epsilon:
            control = self.environment.generate_random_control(current_state)
        else:
            control = self.q_table.get_optimal_control(current_state)

        self.last_state = current_state

        return control

    def finalize(self):
        if self.configuration['plots']:
            sns.lineplot(data=self.cum_avg_costs).set(title="Cum Avg Train Cost")
            plt.show()
        self.q_table.export_qtable()
