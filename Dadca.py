import itertools
import math
from typing import TypedDict, Optional, List, Dict, Tuple

from controller import Controller
from environment import Control, MobilityCommand, Environment
from node import Agent
from simulation_configuration import SimulationConfiguration
from state import State


class DadcaCommand(TypedDict):
    destination: int
    proceed: MobilityCommand


class Dadca(Controller):
    """
    Controller that implements DADCA
    """

    agent_neighbours: List[list]
    agent_commands: Dict[int, DadcaCommand]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.agent_neighbours = [[0, 0] for _ in range(self.configuration['num_agents'])]
        self.agent_commands = {}

    def get_control(self, simulation_step: int, current_state: State, current_control: Optional[Control]) -> Control:
        if current_control is not None:
            new_mobility_commands = list(current_control.mobility)
        else:
            mobility = [MobilityCommand.FORWARDS for _ in range(self.configuration['num_agents'])]
            current_control = Control(tuple(mobility), self.configuration)
            new_mobility_commands = mobility.copy()

        combination: Tuple[Tuple[int, Agent], Tuple[int, Agent]]
        for combination in itertools.combinations(enumerate(self.environment.agents), 2):
            index1, agent1 = combination[0]
            index2, agent2 = combination[1]

            # Ignoring agents that already have orders
            if index1 in self.agent_commands or index2 in self.agent_commands:
                continue

            # Ignoring agents that are not in the same position
            if agent1.position != agent2.position:
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
            destination = math.floor(interaction_boundary * self.configuration['mission_size'])
            self.agent_commands[left_index] = {
                "destination": destination,
                "proceed": MobilityCommand.REVERSE
            }

            self.agent_commands[right_index] = {
                "destination": destination,
                "proceed": MobilityCommand.FORWARDS
            }

        for agentIndex, command in self.agent_commands.copy().items():
            if self.environment.agents[agentIndex].position < command['destination']:
                new_mobility_commands[agentIndex] = MobilityCommand.FORWARDS
            elif self.environment.agents[agentIndex].position > command['destination']:
                new_mobility_commands[agentIndex] = MobilityCommand.REVERSE
            else:
                new_mobility_commands[agentIndex] = command['proceed']
                self.agent_commands.pop(agentIndex)

        # Respecting mission boundaries
        for index, agent in enumerate(self.environment.agents):
            if agent.position == 0:
                new_mobility_commands[index] = MobilityCommand.FORWARDS
            elif agent.position == self.configuration['mission_size'] - 1:
                new_mobility_commands[index] = MobilityCommand.REVERSE

        return Control(tuple(new_mobility_commands), self.configuration)
