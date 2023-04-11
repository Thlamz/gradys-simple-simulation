from typing import List

from node import GroundStation, Agent, Sensor
from simulation_configuration import SimulationConfiguration


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

        self.ground_station = GroundStation()
        self.agents = []
        self.sensors = []

        for _ in range(1, configuration['mission_size']):
            self.sensors.append(Sensor(self.configuration))

        for _ in range(configuration['num_agents']):
            self.agents.append(Agent())
