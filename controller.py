from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypedDict, Callable, Union, List

import numpy as np

from control import Control
from environment import Environment
from simulation_configuration import SimulationConfiguration
from state import State


class Controller(ABC):
    """
    Abstract class representing a controller. Controllers are responsible for reading the simulation state and
    choosing the next global control that should be executed.
    """
    configuration: SimulationConfiguration
    environment: Environment

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        self.configuration = configuration
        self.environment = environment

        self.statistics_bin_size = self.configuration['simulation_steps'] // 1000
        self.bins = np.linspace(0, self.configuration['simulation_steps'], 1000)[:-1].tolist()

    def compute_statistics(self, value, buffer: List, statistic: List):
        buffer.append(value)
        if len(buffer) >= self.statistics_bin_size:
            statistic.append(sum(buffer) / len(buffer))
            buffer.clear()

    def finalize_statistics(self, buffer: List, statistic: List):
        # Adding any remaining data in buffer to the statistic
        if len(buffer) < self.statistics_bin_size - 1:
            statistic.append(sum(buffer) / (len(buffer) or 1))
        return statistic

    @abstractmethod
    def get_control(self, simulation_step: int, current_state: State, current_control: Optional[Control]) -> Control:
        pass

    def finalize(self) -> dict:
        return {}


class QLearningParameters(TypedDict):
    # Q Learning parameters
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float
    reward_function: Callable
    qtable_initialization_value: float
    qtable_file: Union[Path, None]
    qtable_format: Callable
