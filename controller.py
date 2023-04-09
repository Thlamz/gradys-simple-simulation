from abc import ABC, abstractmethod
from typing import Optional

from environment import Control, Environment
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

    @abstractmethod
    def get_control(self, simulation_step: int, current_state: State, current_control: Optional[Control]) -> Control:
        pass

    def finalize(self) -> dict:
        return {}
