import itertools
import math
from enum import Enum
from typing import NamedTuple, Union

import numpy

from simulation_configuration import SimulationConfiguration


class State(NamedTuple):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """
    mobility: tuple


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = 0


class Control(NamedTuple):
    """
    The control in this scenario has only one key, mobility. The enum value mobility[i] represents the direction
    agent i will move
    """
    mobility: tuple


_control_choices = [v for v in MobilityCommand]


def tuple_to_base_id(t: tuple, base: int) -> int:
    """
    Auxiliary function that is used to serialize a tuple with a known max value to an integer
    :param t: Tuple of integers
    :param base: Known max value of the tuple
    :return: Integer representing the tuple
    """
    result = 0
    for index, data in enumerate(t):
        result += base ** index * data
    return result


def base_id_to_tuple(n: int, base: int, size: int) -> tuple:
    """
    Auxiliary functions that deserializes an integer to a tuple with a known max value and size
    :param n: Number to deserialzie
    :param base: Known max value of the tuple
    :param size: Size of the tuple
    :return: Deserialized tuple
    """
    result = [0 for _ in range(size)]
    digit = 0
    while n >= base:
        remainder = n % base
        result[digit] = int(remainder)
        n = (n - remainder) / base
        digit += 1
    result[digit] = int(n)
    return tuple(result)


class StateSerializer:
    """
    Auxiliary class that used tuple serialization/deserialization to serialize the state
    """
    base: int

    def __init__(self, max_value: int, num_agents: int):
        self.base = max_value
        self.size = num_agents

    def __getitem__(self, item: State | int):
        if isinstance(item, State):
            return tuple_to_base_id(item.mobility, self.base)
        else:
            return State(mobility=base_id_to_tuple(item, self.base, self.size))


class ControlSerializer:
    """
    Auxiliary class that used tuple serialization/deserialization to serialize the control
    """
    base: int

    def __init__(self, max_value: int, num_agents: int):
        self.base = max_value
        self.size = num_agents

    def __getitem__(self, item: Control | int):
        if isinstance(item, Control):
            return tuple_to_base_id(tuple(cmd.value for cmd in item.mobility), self.base)
        else:
            return Control(mobility=tuple(MobilityCommand(value) for value in base_id_to_tuple(item, self.base, self.size)))


class Environment:
    """
    Represents the simulation's environment. Determines the rules of the simulation by validating the controls
    that are possible for a state. Also offers some utilitary functions
    """
    def __init__(self, configuration: SimulationConfiguration):
        self.configuration = configuration
        self.mission_size = self.configuration['mission_size']
        self.rng = numpy.random.default_rng(seed=0)

        self.state_id = StateSerializer(self.configuration['mission_size'], self.configuration['num_agents'])
        self.control_id = ControlSerializer(len(MobilityCommand), self.configuration['num_agents'])

    def validate_control(self, current_state: State, control: Control) -> bool:
        """
        Validates that a given control applied to the current state generates a valid next state
        """
        for position, command in zip(current_state.mobility, control.mobility):
            if position == 0 and command == MobilityCommand.REVERSE:
                return False

            if position == self.configuration['mission_size'] - 1 and command == MobilityCommand.FORWARDS:
                return False
        return True

    def generate_random_control(self, state: State) -> Control:
        """
        Generates a random, valid, control
        """
        mobility_choices = self.rng.integers(0, len(_control_choices), self.configuration['num_agents'])
        mobility_control = [_control_choices[i] for i in mobility_choices]

        for index, control in enumerate(mobility_control):
            if state.mobility[index] == 0:
                mobility_control[index] = MobilityCommand.FORWARDS
            elif state.mobility[index] == self.configuration['mission_size'] - 1:
                mobility_control[index] = MobilityCommand.REVERSE

        control = Control(mobility=tuple(mobility_control))
        assert self.validate_control(state, control)
        return control

    def execute_control(self, current_state: State, control: Control) -> State:
        """
        Executes a control on the current state, generating a next state
        """
        new_mobility_state: list[int] = []

        for position, command in zip(current_state.mobility, control.mobility):
            if command == MobilityCommand.FORWARDS:
                new_mobility_state.append(position + 1)
            else:
                new_mobility_state.append(position - 1)

        return State(mobility=tuple(new_mobility_state))
