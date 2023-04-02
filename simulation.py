from typing import Optional

import numpy

from control import Control
from controller import Controller
from environment import MobilityCommand, Environment
from simulation_configuration import SimulationConfiguration
from state import State


class SimulationException(Exception):
    pass


class Simulation:
    """
    Represents an instance of a simulation. This class acts as a coordinator between all the nodes in the simulation,
    and the environment
    """

    controller: Controller

    configuration: SimulationConfiguration

    X: State
    U: Optional[Control]

    simulation_step: int

    def __init__(self, configuration: SimulationConfiguration):
        self.simulation_step = 0
        self.configuration = configuration
        self.environment = Environment(configuration)
        self.controller = configuration['controller'](configuration, self.environment)

        self.rng = numpy.random.default_rng()

        self.X = self.configuration['state'](self.configuration, self.environment)
        self.U = None

    def simulate(self):
        """ Runs a single simulation step """
        self.U = self.controller.get_control(self.simulation_step, self.X, self.U)
        if not self.environment.validate_control(self.U):
            raise SimulationException("Invalid control")

        self.environment.execute_control(self.U)

        # Simulating sensor packet generation
        if self.simulation_step % self.configuration['sensor_generation_frequency'] == 0:
            probabilities = self.rng.random(size=self.configuration['mission_size'] - 1)
            for sensor, probability in zip(self.environment.sensors, probabilities):
                if probability < self.configuration['sensor_generation_probability']:
                    sensor.lifecycle_packets.append({'created_at': self.simulation_step})

        # Simulate message exchange
        for index1, agent1 in enumerate(self.environment.agents):
            if self.U.mobility[index1] == MobilityCommand.FORWARDS:
                continue

            for index2, agent2 in enumerate(self.environment.agents):
                if self.U.mobility[index2] == MobilityCommand.REVERSE \
                        or agent1.position > agent2.position:
                    continue

                if agent2.position <= (agent1.position + 2):
                    agent1.packets += agent2.packets
                    agent2.packets = 0

        # Simulating sensor packet pickup
        for index, agent in enumerate(self.environment.agents):
            agent_mobility = agent.position
            if agent_mobility == 0:
                self.environment.ground_station.packets += agent.packets
                agent.packets = 0
            else:
                num_packets = 0
                for packet in self.environment.sensors[agent_mobility - 1].lifecycle_packets:
                    if self.simulation_step - packet['created_at'] <= self.configuration['sensor_packet_lifecycle']:
                        num_packets += 1
                agent.packets += num_packets
                self.environment.sensors[agent_mobility - 1].lifecycle_packets = []

        self.X = self.configuration['state'](self.configuration, self.environment)
        self.simulation_step += 1
