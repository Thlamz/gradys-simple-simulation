import itertools
from enum import Enum
from typing import NamedTuple, Union

from simulation_configuration import SimulationConfiguration


class State(NamedTuple):
    mobility: tuple[int, ...]


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = -1


class Control(NamedTuple):
    mobility: tuple[MobilityCommand, ...]


class Environment:
    state_id: dict[Union[State, int], Union[State, int]]
    control_id: dict[Union[Control, int], Union[Control, int]]

    def __init__(self, configuration: SimulationConfiguration):
        self.configuration = configuration

        self.calculate_state_ids()
        self.calculate_control_ids()

    def calculate_state_ids(self):
        self.state_id = {}
        for index, permutation in enumerate(itertools.product([i for i in range(self.configuration['mission_size'])],
                                                              repeat=self.configuration['num_agents'])):
            self.state_id[index] = State(mobility=permutation)
            self.state_id[State(mobility=permutation)] = index

    def calculate_control_ids(self):
        self.control_id = {}
        for index, permutation in enumerate(itertools.product([MobilityCommand.FORWARDS, MobilityCommand.REVERSE],
                                                              repeat=self.configuration['num_agents'])):
            self.control_id[index] = Control(mobility=permutation)
            self.control_id[Control(mobility=permutation)] = index

    def validate_control(self, current_state: State, control: Control) -> bool:
        for position, command in zip(current_state.mobility, control.mobility):
            if position == 0 and command == MobilityCommand.REVERSE:
                return False

            if position == self.configuration['mission_size'] - 1 and command == MobilityCommand.FORWARDS:
                return False
        return True

    def execute_control(self, current_state: State, control: Control) -> State:
        new_mobility_state: list[int] = []

        for position, command in zip(current_state.mobility, control.mobility):
            if command == MobilityCommand.FORWARDS:
                new_mobility_state.append(position + 1)
            else:
                new_mobility_state.append(position - 1)

        return State(mobility=tuple(new_mobility_state))
