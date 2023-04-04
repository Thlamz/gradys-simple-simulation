from typing import List

import numpy

from control import MobilityCommand, Control
from node import GroundStation, Agent, Sensor
from simulation_configuration import SimulationConfiguration

_control_choices = [v for v in MobilityCommand]


class Environment:
    """
    Represents the simulation's environment. Determines the rules of the simulation by validating the controls
    that are possible for a state. Also offers some utilitary functions
    """
    ground_station: GroundStation
    agents: List[Agent]
    sensors: List[Sensor]

    def __init__(self, configuration: SimulationConfiguration):
        self.configuration = configuration
        self.mission_size = self.configuration['mission_size']
        self.rng = numpy.random.default_rng(seed=0)

        self.ground_station = GroundStation()
        self.agents = []
        self.sensors = []

        for _ in range(1, configuration['mission_size']):
            self.sensors.append(Sensor())

        for _ in range(configuration['num_agents']):
            self.agents.append(Agent())

    def validate_control(self, control: Control) -> bool:
        """
        Validates that a given control applied to the current state generates a valid next state
        """
        for agent, command in zip(self.agents, control.mobility):
            position = agent.position
            if position == 0 and command == MobilityCommand.REVERSE:
                return False

            if position == self.configuration['mission_size'] - 1 and command == MobilityCommand.FORWARDS:
                return False
        return True

    def generate_random_control(self) -> Control:
        """
        Generates a random, valid, control
        """
        mobility_choices = self.rng.integers(0, len(_control_choices), self.configuration['num_agents'])
        mobility_control = [_control_choices[i] for i in mobility_choices]

        for index, control in enumerate(mobility_control):
            if self.agents[index].position == 0:
                mobility_control[index] = MobilityCommand.FORWARDS
            elif self.agents[index].position == self.configuration['mission_size'] - 1:
                mobility_control[index] = MobilityCommand.REVERSE

        control = Control(tuple(mobility_control), self.configuration)
        assert self.validate_control(control)
        return control

    def execute_control(self, control: Control):
        """
        Executes a control on the current state, generating a next state
        """

        for agent, command in zip(self.agents, control.mobility):
            if command == MobilityCommand.FORWARDS:
                agent.position += 1
            else:
                agent.position -= 1
