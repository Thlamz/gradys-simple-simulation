from typing import Optional

import numpy

from control import Control, MobilityCommand, validate_control, execute_control
from controller import Controller
from environment import Environment
from rng import rng
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

        self.X = self.configuration['state'].build(self.configuration, self.environment)
        self.U = None

    def simulate(self):
        """ Runs a single simulation step """
        self.U = self.controller.get_control(self.simulation_step, self.X, self.U)
        if not validate_control(self.U, self.configuration, self.environment):
            raise SimulationException("Invalid control")

        execute_control(self.U, self.configuration, self.environment)

        # Simulating sensor packet generation
        if self.simulation_step % self.configuration['sensor_generation_frequency'] == 0:
            probabilities = rng.random(size=self.configuration['mission_size'] - 1)
            for sensor, probability in zip(self.environment.sensors, probabilities):
                if probability < self.configuration['sensor_generation_probability']:
                    sensor.add_packet(self.simulation_step)

        # Simulating sensor packet pickup
        for index, agent in enumerate(self.environment.agents):
            agent_mobility = agent.position
            if agent.position > 0:
                agent.packets += self.environment.sensors[agent_mobility - 1].count_update_packets(self.simulation_step)
                agent.sources.add(agent.position)
                self.environment.sensors[agent_mobility - 1].clear_packets()

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
                    agent1.sources = agent1.sources.union(agent2.sources)
                    agent2.packets = 0
                    agent2.sources.clear()

        # Simulating packet delivery
        for index, agent in enumerate(self.environment.agents):
            agent_mobility = agent.position
            if agent_mobility == 0:
                self.environment.ground_station.packets += agent.packets
                agent.packets = 0
                agent.sources.clear()

        self.X = self.configuration['state'].build(self.configuration, self.environment)
        self.simulation_step += 1
