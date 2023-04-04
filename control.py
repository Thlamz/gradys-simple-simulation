from enum import Enum
from typing import NamedTuple, List

from base_serializer import base_id_to_tuple, tuple_to_base_id
from simulation_configuration import SimulationConfiguration


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = 0


class Control:
    """
    The control in this scenario has only one key, mobility. The enum value mobility[i] represents the direction
    agent i will move
    """
    mobility: List[MobilityCommand]

    def __init__(self, mobility: tuple, configuration: SimulationConfiguration):
        self.mobility = list(mobility)

    def __hash__(self):
        return tuple_to_base_id(tuple(cmd.value for cmd in self.mobility), len(MobilityCommand))

    @classmethod
    def unhash(cls, hash_id: int, configuration: SimulationConfiguration):
        return Control(base_id_to_tuple(hash_id, len(MobilityCommand), configuration['num_agents']),
                       configuration)
