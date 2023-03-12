import itertools
from enum import Enum
from typing import NamedTuple, Annotated

from simulation_parameters import MISSION_SIZE, NUM_AGENTS


class State(NamedTuple):
    mobility: tuple[int, ...]


STATE_ID: dict[State | int, State | int] = {}
for index, permutation in enumerate(itertools.product([i for i in range(MISSION_SIZE)], repeat=NUM_AGENTS)):
    STATE_ID[index] = State(mobility=permutation)
    STATE_ID[State(mobility=permutation)] = index


class MobilityCommand(Enum):
    FORWARDS = 1
    REVERSE = -1


class Control(NamedTuple):
    mobility: tuple[MobilityCommand, ...]


CONTROL_ID: dict[Control | int, Control | int] = {}
for index, permutation in enumerate(itertools.product([MobilityCommand.FORWARDS, MobilityCommand.REVERSE],
                                                      repeat=NUM_AGENTS)):
    CONTROL_ID[index] = Control(mobility=permutation)
    CONTROL_ID[Control(mobility=permutation)] = index


def validate_control(current_state: State, control: Control) -> bool:
    for position, command in zip(current_state.mobility, control.mobility):
        if position == 0 and command == MobilityCommand.REVERSE:
            return False

        if position == MISSION_SIZE - 1 and command == MobilityCommand.FORWARDS:
            return False
    return True


def execute_control(current_state: State, control: Control) -> State:
    new_mobility_state: list[int] = []

    for position, command in zip(current_state.mobility, control.mobility):
        if command == MobilityCommand.FORWARDS:
            new_mobility_state.append(position + 1)
        else:
            new_mobility_state.append(position - 1)

    return State(mobility=tuple(new_mobility_state))
