import pickle
import random
from pathlib import Path

import numpy as np
import numpy.random

from controller import Controller
from environment import State, Control, MobilityCommand, validate_control, STATE_ID, CONTROL_ID
import seaborn as sns
import matplotlib.pyplot as plt

from node import Node
from simulation_parameters import NUM_AGENTS, MISSION_SIZE, MAXIMUM_SIMULATION_STEPS, SENSOR_GENERATION_FREQUENCY

_control_choices = [MobilityCommand.FORWARDS, MobilityCommand.REVERSE]


class QLearning(Controller):
    training: bool
    """ Training flag. If set to false doesn't try to update the Q Table """

    qtable_file: Path | None = None
    """ Path where a Q Table pickle is located. If set will try to load the Q Table at initialization """

    # Variables updated during execution
    last_state: State | None
    epsilon: float
    q_table: np.ndarray[float]

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
    qtable_initialization_value: int

    def __init__(self,
                 epsilon_start=1.,
                 epsilon_end=0.001,
                 epsilon_horizon=MAXIMUM_SIMULATION_STEPS,
                 learning_rate=0.1,
                 gamma=0.99,
                 qtable_initialization_value=0,
                 qtable_file: Path | None = None,
                 training: bool = True):
        super().__init__()
        self.training = training
        if self.training:
            self.epsilon = epsilon_start
        if not self.training:
            self.epsilon = 0

        self.qtable_file = qtable_file
        if qtable_file is not None and qtable_file.is_file():
            self.q_table = np.load(str(self.qtable_file))
        else:
            possible_states = MISSION_SIZE ** NUM_AGENTS
            possible_actions = len(_control_choices) ** NUM_AGENTS
            self.q_table = np.full((possible_states, possible_actions), qtable_initialization_value)
        print(f"QTable size: {self.q_table.size}")

        self.last_state = None
        self.total_cost = 0
        self.cum_avg_costs = []
        self.epsilons = []

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_horizon = epsilon_horizon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.qtable_initialization_value = qtable_initialization_value

        self.rng = numpy.random.default_rng()

    def generate_random_control(self, state: State) -> Control:
        while True:
            control = Control(mobility=tuple(
                _control_choices[self.rng.integers(0, len(_control_choices))] for _ in range(NUM_AGENTS)))
            if validate_control(state, control):
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
        highest_throughput = (MISSION_SIZE - 1) * (simulation_step / SENSOR_GENERATION_FREQUENCY)
        return -(ground_station['packets'] / simulation_step) / highest_throughput

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

            previous_qvalue = self.q_table[STATE_ID[self.last_state], CONTROL_ID[current_control]]
            next_state_qvalue = np.max(self.q_table[STATE_ID[current_state], :])

            q_value = ((1 - self.learning_rate) * previous_qvalue
                       + self.learning_rate * (reward + self.gamma * next_state_qvalue))

            self.q_table[STATE_ID[self.last_state], CONTROL_ID[current_control]] = q_value

            self.decay_epsilon()

        if self.rng.uniform(0, 1) < self.epsilon:
            control = self.generate_random_control(current_state)
        else:
            if np.any(self.q_table[STATE_ID[current_state], :] > self.qtable_initialization_value):
                control = CONTROL_ID[np.argmax(self.q_table[STATE_ID[current_state], :])]
            else:
                control = self.generate_random_control(current_state)

        self.last_state = current_state

        return control

    def finalize(self):
        sns.lineplot(data=self.cum_avg_costs).set(title="Cum Avg Train Cost")
        plt.show()

        if self.qtable_file is not None:
            np.save(str(self.qtable_file), self.q_table)
