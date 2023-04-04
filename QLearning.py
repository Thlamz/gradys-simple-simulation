import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Dict, List

import numpy as np
import numpy.random

from controller import Controller
from environment import Control, MobilityCommand, Environment
import seaborn as sns
import matplotlib.pyplot as plt

from simulation_configuration import SimulationConfiguration
from state import State


class QTable(ABC):
    """
    Abstract class implementing methods necessary for a QTable implementation
    """
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
    """
    Q Table implementation with a sparse implementation using dictionaries
    """
    q_table: Dict[int, Dict[int, float]]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.initialize_q_table()

    def load_q_table(self):
        with self.configuration['qtable_file'].open("r") as file:
            self.q_table: dict = json.load(file, object_hook=lambda d: {int(k): v for k, v in d.items()})

    def initialize_q_table(self):
        if self.configuration['qtable_file'] is not None and self.configuration['qtable_file'].is_file():
            self.load_q_table()
        else:
            self.q_table = {}

    def get_q_value(self, state: State, control: Control) -> float:
        state_id = hash(state)
        control_id = hash(control)
        if state_id not in self.q_table or control_id not in self.q_table[state_id]:
            return self.configuration['qtable_initialization_value']
        return self.q_table[state_id][control_id]

    def get_optimal_control(self, state: State) -> Control:
        state_id = hash(state)
        if state_id in self.q_table:
            optimal_control: Optional[int] = None
            optimal_q_value = -math.inf

            for control, q_value in self.q_table[state_id].items():
                if q_value > optimal_q_value:
                    optimal_control = control
                    optimal_q_value = q_value
            if optimal_control is not None:
                return Control.unhash(optimal_control, self.configuration)
        return self.environment.generate_random_control()

    def set_q_value(self, state: State, control: Control, q_value: float):
        state_id = hash(state)
        if state_id not in self.q_table:
            self.q_table[state_id] = {}
        self.q_table[state_id][hash(control)] = q_value

    def export_qtable(self):
        if self.configuration['qtable_file'] is not None:
            with self.configuration['qtable_file'].open("w") as file:
                json.dump(self.q_table, file)


class DenseQTable(QTable):
    """
    Dense Q Table implementation that used numpy arrays
    """
    q_table: numpy.ndarray

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.initialize_q_table()

    def load_q_table(self):
        self.q_table = np.load(str(self.configuration['qtable_file']))

    def initialize_q_table(self):
        if self.configuration['qtable_file'] is not None and self.configuration['qtable_file'].is_file():
            self.load_q_table()
        else:
            possible_states = self.configuration['mission_size'] ** self.configuration['num_agents']
            possible_actions = len(MobilityCommand) ** self.configuration['num_agents']
            self.q_table = np.full((possible_states, possible_actions),
                                   self.configuration['qtable_initialization_value'],
                                   dtype=float)

    def get_q_value(self, state: State, control: Control) -> float:
        return self.q_table[hash(state), hash(control)]

    def get_optimal_control(self, state: State) -> Control:
        if np.any(self.q_table[hash(state), :] > self.configuration['qtable_initialization_value']):
            hash_id = np.argmax(self.q_table[hash(state), :])
            return Control.unhash(hash_id, self.configuration)
        else:
            return self.environment.generate_random_control()

    def set_q_value(self, state: State, control: Control, q_value: float):
        state_id = hash(state)
        control_id = hash(control)
        self.q_table[state_id, control_id] = q_value

    def export_qtable(self):
        if self.configuration['qtable_file'] is not None:
            with self.configuration['qtable_file'].open("wb") as file:
                np.save(file, self.q_table)


class QLearning(Controller):
    """
    Controller that implements a Centralized Q Learning algorithm
    """

    training: bool
    """ Training flag. If set to false doesn't try to update the Q Table """

    qtable_file: Optional[Path] = None
    """ 
    Path where a Q Table should be serialized and persisted. 
    If set will try to load the Q Table at initialization.
    """

    # Variables updated during execution
    last_state: Optional[State]
    epsilon: float
    q_table: QTable
    """QTable class implementing QTable operations"""

    # Statistic collection
    total_reward: float
    cum_avg_rewards: List[float]
    epsilons: List[float]

    # Configuration variables
    epsilon_start: float
    epsilon_end: float
    epsilon_horizon: int
    learning_rate: float
    gamma: float
    qtable_initialization_value: float
    qtable_format: Literal['sparse', 'dense']

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
        self.q_table = SparseQTable(configuration, environment) \
            if self.qtable_format == 'sparse' \
            else DenseQTable(configuration, environment)

        if configuration['verbose']:
            size = (len(MobilityCommand) ** self.configuration['num_agents'] *
                    self.configuration['mission_size'] ** self.configuration['num_agents'])
            print(f"QTable size: {size}")

        self.last_state = None
        self.total_reward = 0
        self.cum_avg_rewards = []
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

    def compute_reward(self, simulation_step):
        if simulation_step == 0:
            return 0
        highest_throughput = (self.configuration['mission_size'] - 1) * (
                    simulation_step / self.configuration['sensor_generation_frequency'])
        return self.environment.ground_station.packets / highest_throughput

    def get_control(self, simulation_step: int, current_state: State, current_control: Control) -> Control:
        reward = self.compute_reward(simulation_step)
        self.total_reward += reward
        if simulation_step > 0 and self.configuration['plots']:
            self.cum_avg_rewards.append(self.total_reward / simulation_step)

        if self.training and self.last_state is not None:
            previous_qvalue = self.q_table.get_q_value(self.last_state, current_control)
            next_state_qvalue = self.q_table.get_q_value(current_state, self.q_table.get_optimal_control(current_state))

            q_value = ((1 - self.learning_rate) * previous_qvalue
                       + self.learning_rate * (reward + self.gamma * next_state_qvalue))

            self.q_table.set_q_value(self.last_state, current_control, q_value)

            self.decay_epsilon()

        if self.training and self.rng.random() < self.epsilon:
            control = self.environment.generate_random_control()
        else:
            control = self.q_table.get_optimal_control(current_state)

        self.last_state = current_state

        return control

    def finalize(self):
        if self.configuration['plots']:
            sns.lineplot(data=self.cum_avg_rewards).set(title="Cum Avg Train Rewards")
            plt.show()

            sns.lineplot(data=self.epsilons).set(title="Epsilons")
            plt.show()
        self.q_table.export_qtable()
