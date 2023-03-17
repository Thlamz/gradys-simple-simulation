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
        while True:
            control = Control(mobility=tuple(
                _control_choices[self.rng.integers(0, len(_control_choices))] for _ in
                range(self.configuration['num_agents'])))
            if self.validate_control(state, control):
                break
        return control

    def execute_control(self, current_state: State, control: Control) -> State:
        new_mobility_state: list[int] = []

        for position, command in zip(current_state.mobility, control.mobility):
            if command == MobilityCommand.FORWARDS:
                new_mobility_state.append(position + 1)
            else:
                new_mobility_state.append(position - 1)

        return State(mobility=tuple(new_mobility_state))
