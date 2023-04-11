from abc import ABC, abstractmethod
from typing import List

from base_serializer import tuple_to_base_id, base_id_to_tuple
from environment import Environment
from serializable import IntSerializable
from simulation_configuration import SimulationConfiguration


class State(IntSerializable):
    configuration: SimulationConfiguration
    environment: Environment

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        self.configuration = configuration
        self.environment = environment

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def serialize(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, environment: Environment):
        pass

    @classmethod
    @abstractmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        pass


class MobilityState(State):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """

    mobility: List[int]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.mobility = []
        for agent in environment.agents:
            self.mobility.append(agent.position)

    def serialize(self) -> int:
        return tuple_to_base_id(tuple(self.mobility), self.configuration['mission_size'])

    def __eq__(self, other):
        return self.mobility == other.mobility

    @classmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, environment: Environment):
        mobility = base_id_to_tuple(serialized, configuration['mission_size'], configuration['num_agents'])
        state = MobilityState(configuration, environment)
        state.mobility = list(mobility)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['mission_size'] ** configuration['num_agents']


class SignedMobilityState(State):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """

    mobility: List[int]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)
        self.mobility = []
        for agent in environment.agents:
            self.mobility.append(agent.position
                                 if not agent.reversed
                                 else self.configuration['mission_size'] + agent.position)

    def serialize(self) -> int:
        return tuple_to_base_id(tuple(self.mobility), self.configuration['mission_size'] * 2)

    def __eq__(self, other):
        return self.mobility == other.mobility

    @classmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, environment: Environment):
        mobility = base_id_to_tuple(serialized, configuration['mission_size'] * 2, configuration['num_agents'])
        state = MobilityState(configuration, environment)
        state.mobility = list(mobility)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['mission_size'] ** configuration['num_agents'] * 2


class CommunicationMobilityState(State):
    mobility: List[int]
    communication: List[int]

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.mobility = []
        self.communication = []
        for agent in environment.agents:
            self.mobility.append(agent.position)

        for sensor in environment.sensors:
            self.communication.append(1 if sensor.count_packets() > 0 else 0)

    def __eq__(self, other):
        return self.mobility == other.mobility and self.communication == other.communication

    def serialize(self) -> int:
        mobility_serialized = tuple_to_base_id(tuple(self.mobility), self.configuration['mission_size'])
        communication_serialized = tuple_to_base_id(tuple(self.communication), 2)

        communication_count: int = 2 ** (self.configuration['mission_size'] - 1)

        mixed_num = communication_serialized
        mixed_num += mobility_serialized << communication_count.bit_length()
        return mixed_num

    @classmethod
    def deserialize(cls, serialized: int, configuration: SimulationConfiguration, environment: Environment):
        communication_count: int = 2 ** (configuration['mission_size'] - 1)

        mobility_serialized = serialized >> communication_count.bit_length()
        communication_serialized = serialized - (mobility_serialized << communication_count.bit_length())

        mobility = base_id_to_tuple(mobility_serialized, configuration['mission_size'], configuration['num_agents'])
        communication = base_id_to_tuple(communication_serialized, 2, configuration['mission_size'] - 1)

        state = State(configuration, environment)
        state.mobility = mobility
        state.communication = communication
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        mobility_count = configuration['mission_size'] ** configuration['num_agents']
        communication_count = 2 ** (configuration['mission_size'] - 1)
        return mobility_count * communication_count

