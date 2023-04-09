from abc import abstractmethod, ABC

from environment import Environment
from simulation_configuration import SimulationConfiguration


class IntSerializable(ABC):
    @abstractmethod
    def serialize(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, environment: Environment):
        pass
