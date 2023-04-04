from abc import ABC, abstractmethod
from typing import List

from base_serializer import tuple_to_base_id, base_id_to_tuple
from environment import Environment
from simulation_configuration import SimulationConfiguration


class State(ABC):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """
    configuration: SimulationConfiguration
    environment: Environment

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        self.configuration = configuration
        self.environment = environment

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @classmethod
    @abstractmethod
    def unhash(cls, hash_id: int, configuration: SimulationConfiguration, environment: Environment):
        pass


class MobilityState(State):
    mobility: List[int]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.mobility = []
        for agent in environment.agents:
            self.mobility.append(agent.position)

    def __hash__(self):
        return tuple_to_base_id(tuple(self.mobility), self.configuration['mission_size'])

    def __eq__(self, other):
        return self.mobility == other.mobility

    @classmethod
    def unhash(cls, hash_id: int, configuration: SimulationConfiguration, environment: Environment):
        mobility = base_id_to_tuple(hash_id, configuration['num_agents'], configuration['num_agents'])
        state = MobilityState(configuration, environment)
        state.mobility = list(mobility)
        return state