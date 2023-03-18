import itertools
from enum import Enum
from typing import NamedTuple, Union

import numpy

from simulation_configuration import SimulationConfiguration


class State(NamedTuple):
    mobility: tuple[int, ...]


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = -1


class Control(NamedTuple):
    mobility: tuple[MobilityCommand, ...]


_control_choices = [v for v in MobilityCommand]


class Environment:
    def __init__(self, configuration: SimulationConfiguration):
        self.configuration = configuration
        self.rng = numpy.random.default_rng()

    def validate_control(self, current_state: State, control: Control) -> bool:
        for position, command in zip(current_state.mobility, control.mobility):
            if position == 0 and command == MobilityCommand.REVERSE:
                return False

            if position == self.configuration['mission_size'] - 1 and command == MobilityCommand.FORWARDS:
                return False
        return True

    def generate_random_control(self, state: State) -> Control:
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
        new_mobility_state: list[int] = []

        for position, command in zip(current_state.mobility, control.mobility):
            if command == MobilityCommand.FORWARDS:
                new_mobility_state.append(position + 1)
            else:
                new_mobility_state.append(position - 1)

        return State(mobility=tuple(new_mobility_state))
