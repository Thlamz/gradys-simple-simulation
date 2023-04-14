from enum import Enum
from typing import NamedTuple, List

import numpy

from base_serializer import base_id_to_tuple, tuple_to_base_id
from environment import Environment
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

    def __hash__(self):
        return hash(tuple(self.mobility))

    @classmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, _environment):
        deserialized_value = base_id_to_tuple(serialized, len(MobilityCommand), configuration['num_agents'])
        return Control(tuple(MobilityCommand(value) for value in deserialized_value),
                       configuration)


def validate_control(control: Control, configuration: SimulationConfiguration, environment: Environment) -> bool:
    """
    Validates that a given control applied to the current state generates a valid next state
    """
    for agent, command in zip(environment.agents, control.mobility):
        position = agent.position
        if position == 0 and command == MobilityCommand.REVERSE:
            return False

        if position == configuration['mission_size'] - 1 and command == MobilityCommand.FORWARDS:
            return False
    return True


rng = numpy.random.default_rng()

rng_batch = []

# Optimization: Batching RNG generation for better performance
def _command_rng(count) -> List[int]:
    global rng_batch
    if len(rng_batch) < count:
        rng_batch = rng.integers(0, len(MobilityCommand), 1_000_000)
    batch = rng_batch[-count:]
    rng_batch = rng_batch[:-count]
    return batch


def generate_random_control(configuration: SimulationConfiguration, environment: Environment) -> Control:
    """
    Generates a random, valid, control
    """
    mobility_choices = _command_rng(configuration['num_agents'])
    mobility_control = [MobilityCommand(i) for i in mobility_choices]

    for index, control in enumerate(mobility_control):
        if environment.agents[index].position == 0:
            mobility_control[index] = MobilityCommand.FORWARDS
        elif environment.agents[index].position == configuration['mission_size'] - 1:
            mobility_control[index] = MobilityCommand.REVERSE

    control = Control(tuple(mobility_control), configuration)
    assert validate_control(control, configuration, environment)
    return control


def execute_control(control: Control, _configuration: SimulationConfiguration, environment: Environment):
    """
    Executes a control on the current state, generating a next state
    """

    for agent, command in zip(environment.agents, control.mobility):
        if command == MobilityCommand.FORWARDS:
            agent.position += 1
            agent.reversed = False
        else:
            agent.position -= 1
            agent.reversed = True
