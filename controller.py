from abc import ABC, abstractmethod

from environment import State, Control
from node import Node


class Controller(ABC):
    def __init__(self):
        pass

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