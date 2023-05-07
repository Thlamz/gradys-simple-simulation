import json
from enum import Enum
from typing import NamedTuple, List

import numpy
import torch

from base_serializer import base_id_to_tuple, tuple_to_base_id
from device import device
from environment import Environment
from rng import rng
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

    def __init__(self, mobility: tuple):
        self.mobility = list(mobility)

    def serialize(self) -> str:
        return json.dumps([command.value for command in self.mobility])

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([command.value for command in self.mobility], dtype=torch.int64, device=device).unsqueeze(0)

    def __eq__(self, other):
        return self.mobility == other.mobility

    def __hash__(self):
        return hash(tuple(self.mobility))

    @classmethod
    def deserialize(cls, serialized: str):
        deserialized_value = json.loads(serialized)
        return Control(tuple(MobilityCommand(value) for value in deserialized_value))

    @classmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment):
        return configuration['num_agents']


def validate_control(control: Control, configuration: SimulationConfiguration, environment: Environment) -> bool:
    """
    Validates that a given control applied to the current state generates a valid next state
    """
    return len(control.mobility) == configuration['num_agents']


rng_batch = []
rng_batch_cursor = 0


# Optimization: Batching RNG generation for better performance
def _command_rng(count) -> List[int]:
    global rng_batch, rng_batch_cursor
    if len(rng_batch) - rng_batch_cursor < count:
        rng_batch = rng.integers(0, len(MobilityCommand), 1_000_000)
        rng_batch_cursor = 0
    batch = rng_batch[rng_batch_cursor:rng_batch_cursor + count]
    rng_batch_cursor += count
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

    control = Control(tuple(mobility_control))
    assert validate_control(control, configuration, environment)
    return control


def execute_control(control: Control, configuration: SimulationConfiguration, environment: Environment):
    """
    Executes a control on the current state, generating a next state
    """

    for agent, command in zip(environment.agents, control.mobility):
        if command == MobilityCommand.FORWARDS:
            agent.position = min(agent.position + 1, configuration['mission_size'] - 1)
            agent.reversed = False
        else:
            agent.position = max(agent.position - 1, 0)
            agent.reversed = True
