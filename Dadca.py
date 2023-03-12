import itertools
import math
from typing import TypedDict

from controller import Controller
from environment import State, Control, MobilityCommand
from node import Node
from simulation_parameters import MISSION_SIZE, NUM_AGENTS


class DadcaCommand(TypedDict):
    destination: int
    proceed: MobilityCommand


class Dadca(Controller):
    agent_neighbours: list[list[int, int]]
    agent_commands: dict[int, DadcaCommand]

    def __init__(self):
        super().__init__()
        self.agent_neighbours = [[0, 0] for _ in range(NUM_AGENTS)]
        self.agent_commands = {}

    def get_control(self,
                    simulation_step: int,
                    current_state: State,
                    current_control: Control,
                    ground_station: Node,
                    agents: list[Node],
                    sensors: list[Node]) -> Control:
        new_mobility_commands = list(current_control.mobility)

        for combination in itertools.combinations(enumerate(agents), 2):
            index1, agent1 = combination[0]
            index2, agent2 = combination[1]

            # Ignoring agents that already have orders
            if index1 in self.agent_commands or index2 in self.agent_commands:
                continue

            # Ignoring agents that are not in the same position
            if current_state.mobility[index1] != current_state.mobility[index2]:
                continue

            if current_control.mobility[index1] == MobilityCommand.REVERSE:
                left_index, left_agent = index2, agent2
                right_index, right_agent = index1, agent1
            elif current_control.mobility[index2] == MobilityCommand.REVERSE:
                left_index, left_agent = index1, agent1
                right_index, right_agent = index2, agent2
            else:
                left_index, left_agent = index1, agent1
                right_index, right_agent = index2, agent2

            self.agent_neighbours[left_index][1] = self.agent_neighbours[right_index][1] + 1
            self.agent_neighbours[right_index][0] = self.agent_neighbours[left_index][0] + 1

            interaction_boundary = \
                (self.agent_neighbours[left_index][0] + 1) / (sum(self.agent_neighbours[left_index]) + 1)
            destination = math.floor(interaction_boundary * MISSION_SIZE)
            self.agent_commands[left_index] = {
                "destination": destination,
                "proceed": MobilityCommand.REVERSE
            }

            self.agent_commands[right_index] = {
                "destination": destination,
                "proceed": MobilityCommand.FORWARDS
            }

        for agentIndex, command in self.agent_commands.copy().items():
            if current_state.mobility[agentIndex] < command['destination']:
                new_mobility_commands[agentIndex] = MobilityCommand.FORWARDS
            elif current_state.mobility[agentIndex] > command['destination']:
                new_mobility_commands[agentIndex] = MobilityCommand.REVERSE
            else:
                new_mobility_commands[agentIndex] = command['proceed']
                self.agent_commands.pop(agentIndex)

        # Respecting mission boundaries
        for index, agent in enumerate(agents):
            if current_state.mobility[index] == 0:
                new_mobility_commands[index] = MobilityCommand.FORWARDS
            elif current_state.mobility[index] == MISSION_SIZE - 1:
                new_mobility_commands[index] = MobilityCommand.REVERSE

        return Control(mobility=tuple(new_mobility_commands))
