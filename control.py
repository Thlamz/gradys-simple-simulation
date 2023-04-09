from enum import Enum
from typing import NamedTuple, List

from base_serializer import base_id_to_tuple, tuple_to_base_id
from serializable import IntSerializable
from simulation_configuration import SimulationConfiguration


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = 0


class Control(IntSerializable):
    """
    The control in this scenario has only one key, mobility. The enum value mobility[i] represents the direction
    agent i will move
    """
    mobility: List[MobilityCommand]

    def __init__(self, mobility: tuple, configuration: SimulationConfiguration):
        self.mobility = list(mobility)

    def serialize(self) -> int:
        return tuple_to_base_id(tuple(cmd.value for cmd in self.mobility), len(MobilityCommand))

    def __eq__(self, other):
        return self.mobility == other.mobility

    @classmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, _environment):
        deserialized_value = base_id_to_tuple(serialized, len(MobilityCommand), configuration['num_agents'])
        return Control(tuple(MobilityCommand(value) for value in deserialized_value),
                       configuration)
