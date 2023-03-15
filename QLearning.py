import math
import pickle
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

_control_choices = [MobilityCommand.FORWARDS, MobilityCommand.REVERSE]


class QLearning(Controller):
    training: bool
    """ Training flag. If set to false doesn't try to update the Q Table """

    qtable_file: Optional[Path] = None
    """ Path where a Q Table pickle is located. If set will try to load the Q Table at initialization """

    # Variables updated during execution
    last_state: Optional[State]
    epsilon: float
    q_table: Union[np.ndarray[float], dict[State, dict[Control, float]]]
    q_table_optimal_cache: State[State, Control]
    """ Cache of optimal control for each state, only used if cache_optimal_control is true """

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
        self.cache_optimal_control = self.configuration['cache_optimal_control']
        if self.configuration['qtable_file'] is not None and self.configuration['qtable_file'].is_file():
            self.load_q_table()
        else:
            self.initialize_q_table()

        if configuration['verbose']:
            size = (len(_control_choices) ** self.configuration['num_agents'] *
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

    def load_q_table(self):
        if self.cache_optimal_control:
            self.q_table_optimal_cache: dict[State, Control] = {}

        if self.qtable_format == 'sparse':
            with self.qtable_file.open("rb") as file:
                self.q_table = pickle.load(file)

            # Populating cache based on loaded q_table
            if self.cache_optimal_control:
                for state in self.q_table.keys():
                    optimal_control = None
                    optimal_q_value = -math.inf
                    for control, q_value in self.q_table[state].items():
                        if q_value > optimal_q_value:
                            optimal_control = control
                            optimal_q_value = q_value
                    if optimal_control is not None:
                        self.q_table_optimal_cache[state] = optimal_control
        else:
            self.q_table = np.load(str(self.qtable_file))
            for state, id in self.environment.state_id.items():
                if isinstance(state, int):
                    continue
                optimal_control_id = np.argmax(self.q_table[id, :])
                self.q_table_optimal_cache[state] = self.environment.control_id[optimal_control_id]

    def initialize_q_table(self):
        if self.cache_optimal_control:
            self.q_table_optimal_cache: dict[State, Control] = {}

        if self.qtable_format == 'sparse':
            self.q_table = defaultdict(lambda: defaultdict(lambda: self.qtable_initialization_value))
        else:
            possible_states = self.configuration['mission_size'] ** self.configuration['num_agents']
            possible_actions = len(_control_choices) ** self.configuration['num_agents']
            self.q_table = np.full((possible_states, possible_actions),
                                   self.configuration['qtable_initialization_value'])

    def get_q_value(self, state: State, control: Control) -> float:
        if self.qtable_format == 'sparse':
            return self.q_table[state][control]
        else:
            return self.q_table[self.environment.state_id[state], self.environment.control_id[control]]

    def set_q_value(self, state: State, control: Control, q_value: float):
        if self.cache_optimal_control:
            if state not in self.q_table_optimal_cache:
                self.q_table_optimal_cache[state] = control
            else:
                optimal_q_value = self.get_q_value(state, self.get_optimal_control(state))
                if q_value > optimal_q_value:
                    self.q_table_optimal_cache[state] = control

        if self.qtable_format == 'sparse':
            self.q_table[state][control] = q_value
        else:
            self.q_table[self.environment.state_id[state], self.environment.control_id[control]] = q_value

    def get_optimal_control(self, state: State) -> Control:
        if self.cache_optimal_control:
            if state in self.q_table_optimal_cache:
                return self.q_table_optimal_cache[state]
            else:
                return self.generate_random_control(state)
        else:
            if self.qtable_format == 'sparse':
                optimal_control = None
                optimal_q_value = -math.inf
                for control, q_value in self.q_table[state].items():
                    if q_value > optimal_q_value:
                        optimal_control = control
                        optimal_q_value = q_value
                if optimal_control is not None:
                    return optimal_control
                else:
                    return self.generate_random_control(state)
            else:
                if np.any(self.q_table[self.environment.state_id[state], :] > self.qtable_initialization_value):
                    return self.environment.control_id[
                        np.argmax(self.q_table[self.environment.state_id[state], :])
                    ]
                else:
                    return self.generate_random_control(state)

    def generate_random_control(self, state: State) -> Control:
        while True:
            control = Control(mobility=tuple(
                _control_choices[self.rng.integers(0, len(_control_choices))] for _ in
                range(self.configuration['num_agents'])))
            if self.environment.validate_control(state, control):
                break
        return control

    def decay_epsilon(self) -> None:
        if self.epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon *= (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_horizon)
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
        if simulation_step > 0:
            self.cum_avg_costs.append(self.total_cost / simulation_step)

        if self.last_state is not None and self.training:
            reward = -cost

            previous_qvalue = self.get_q_value(self.last_state, current_control)
            next_state_qvalue = self.get_q_value(current_state, self.get_optimal_control(current_state))

            q_value = ((1 - self.learning_rate) * previous_qvalue
                       + self.learning_rate * (reward + self.gamma * next_state_qvalue))

            self.set_q_value(self.last_state, current_control, q_value)

            self.decay_epsilon()

        if self.rng.uniform(0, 1) < self.epsilon:
            control = self.generate_random_control(current_state)
        else:
            control = self.get_optimal_control(current_state)

        self.last_state = current_state

        return control

    def finalize(self):
        if self.configuration['plots']:
            sns.lineplot(data=self.cum_avg_costs).set(title="Cum Avg Train Cost")
            plt.show()

        if self.qtable_file is not None:
            if self.qtable_format == 'spase':
                with self.qtable_file.open("wb") as file:
                    pickle.dump(self.q_table, file)
            else:
                np.save(str(self.qtable_file), self.q_table)
