from abc import ABC, abstractmethod

from environment import State, Control, Environment
from simulation_configuration import SimulationConfiguration
from node import Node


class Controller(ABC):
    configuration: SimulationConfiguration
    environment: Environment

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        self.configuration = configuration
        self.environment = environment

    @abstractmethod
    def get_control(self, simulation_step: int,
                    current_state: State,
                    current_control: Control,
                    ground_station: Node,
                    agents: list[Node],
                    sensors: list[Node]) -> Control:
        pass

    def finalize(self):
        pass