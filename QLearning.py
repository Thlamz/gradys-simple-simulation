import json
import math
import os
import pickle
import random
import shelve
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Dict, List

import numpy as np
import numpy.random

from control import Control, MobilityCommand, generate_random_control
from controller import Controller, QLearningParameters
from environment import Environment
import seaborn as sns
import matplotlib.pyplot as plt

from rng import rng
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
    q_table: Dict[State, Dict[Control, float]]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.initialize_q_table()

    def load_q_table(self):
        with self.configuration['qtable_file'].open("rb") as file:
            self.q_table = pickle.load(file)

    def initialize_q_table(self):
        if self.configuration['qtable_file'] is not None and self.configuration['qtable_file'].is_file():
            self.load_q_table()
        else:
            self.q_table = {}

    def get_q_value(self, state: State, control: Control) -> float:
        try:
            return self.q_table[state][control]
        except KeyError:
            return self.configuration['controller_config']['qtable_initialization_value']

    def get_optimal_control(self, state: State) -> Control:
        try:
            optimal_control: Optional[Control] = None
            optimal_q_value = -math.inf

            for control, q_value in self.q_table[state].items():
                if q_value > optimal_q_value:
                    optimal_control = control
                    optimal_q_value = q_value
            if optimal_control is not None:
                return optimal_control
        except KeyError:
            return generate_random_control(self.configuration, self.environment)

    def set_q_value(self, state: State, control: Control, q_value: float):
        # Optimization: Raising exception is faster than checking is the key exists at every call
        try:
            self.q_table[state][control] = q_value
        except KeyError:
            self.q_table[state] = {}
            self.q_table[state][control] = q_value

    def export_qtable(self):
        if self.configuration['qtable_file'] is not None:
            with self.configuration['qtable_file'].open("wb") as file:
                pickle.dump(self.q_table, file)


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
    last_control: Optional[Control]
    last_q_value: float
    epsilon: float
    q_table: QTable


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

    configuration: SimulationConfiguration
    controller_configuration: QLearningParameters

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.controller_configuration = configuration['controller_config']

        self.training = self.configuration['training']
        if self.training:
            self.epsilon = self.controller_configuration['epsilon_start']
        if not self.training:
            self.epsilon = 0

        self.q_table = self.controller_configuration['qtable_format'](configuration, environment)

        if configuration['verbose']:
            state_size = configuration['state'].possible_states(configuration, environment)
            control_size = len(MobilityCommand) ** configuration['num_agents']
            print(f"QTable size: {state_size * control_size}")

        self.last_state = None
        self.last_control = None
        self.last_q_value = self.controller_configuration['qtable_initialization_value']
        self.total_reward = 0
        self.cum_avg_rewards = []
        self.epsilons = []

        self.epsilon_start = self.controller_configuration['epsilon_start']
        self.epsilon_end = self.controller_configuration['epsilon_end']
        self.epsilon_horizon = self.configuration['maximum_simulation_steps']
        self.learning_rate = self.controller_configuration['learning_rate']
        self.gamma = self.controller_configuration['gamma']
        self.qtable_initialization_value = self.controller_configuration['qtable_initialization_value']

    def decay_epsilon(self) -> None:
        if self.epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon *= (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_horizon)

        if self.configuration['plots']:
            self.epsilons.append(self.epsilon)

    def compute_reward(self, simulation_step):
        return self.controller_configuration['reward_function'](self, simulation_step)

    def get_control(self, simulation_step: int, current_state: State, current_control: Control) -> Control:
        reward = self.compute_reward(simulation_step)
        self.total_reward += reward
        if simulation_step > 0 and self.configuration['plots']:
            self.cum_avg_rewards.append(self.total_reward / simulation_step)

        if self.training and self.last_state is not None:
            next_state_optimal_control = self.q_table.get_optimal_control(current_state)
            next_state_optimal_control_q_value = self.q_table.get_q_value(current_state, next_state_optimal_control)
            q_value = ((1 - self.learning_rate) * self.q_table.get_q_value(self.last_state, self.last_control) +
                       self.learning_rate * (reward + self.gamma * next_state_optimal_control_q_value))
            self.q_table.set_q_value(self.last_state, self.last_control, q_value)

            self.decay_epsilon()

        if self.training and rng.random() < self.epsilon:
            control = generate_random_control(self.configuration, self.environment)
        else:
            # Optimization: Tries to use previously calculated optimal control before picking new one
            try:
                control = next_state_optimal_control
            except NameError:
                control = self.q_table.get_optimal_control(current_state)

        self.last_state = current_state
        self.last_control = control

        return control

    def finalize(self):
        if self.configuration['plots']:
            sns.lineplot(data=self.cum_avg_rewards).set(title="Cum Avg Train Rewards")
            plt.show()

            sns.lineplot(data=self.epsilons).set(title="Epsilons")
            plt.show()
        if self.training:
            self.q_table.export_qtable()
        return {
            'avg_reward': self.total_reward / self.configuration['maximum_simulation_steps']
        }
