import json
from abc import ABC, abstractmethod
from typing import List

from base_serializer import tuple_to_base_id, base_id_to_tuple
from environment import Environment
from simulation_configuration import SimulationConfiguration


class State():

    @classmethod
    @abstractmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def serialize(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized: str):
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

    def __init__(self, mobility: List[int]):
        self.mobility = mobility

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        return cls(mobility)

    def serialize(self) -> str:
        return json.dumps(self.mobility)

    def __eq__(self, other):
        return self.mobility == other.mobility

    def __hash__(self):
        return hash(tuple(self.mobility))

    @classmethod
    def deserialize(cls, serialized: str):
        mobility = json.loads(serialized)
        return cls(mobility)

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['mission_size'] ** configuration['num_agents']


class SignedMobilityState(State):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """

    mobility: List[int]

    def __init__(self, mobility: List[int]):
        self.mobility = mobility

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position
                    if agent.reversed
                    else configuration['mission_size'] + agent.position for agent in environment.agents]
        return cls(mobility)

    def serialize(self) -> str:
        return json.dumps(self.mobility)

    def __hash__(self):
        return hash(tuple(self.mobility))

    def __eq__(self, other):
        return self.mobility == other.mobility

    @classmethod
    def deserialize(cls, serialized: str):
        mobility = json.loads(serialized)
        state = cls(mobility)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return (configuration['mission_size'] * 2) ** configuration['num_agents']


class CommunicationMobilityState(State):
    mobility: List[int]
    communication: List[int]

    def __init__(self, mobility: List[int], communication: List[int]):
        self.mobility = mobility
        self.communication = communication

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        communication = [1 if sensor.count_packets() > 0 else 0 for sensor in environment.sensors]
        return cls(mobility, communication)

    def __hash__(self):
        return hash(tuple(self.mobility)) + hash(tuple(self.communication))

    def __eq__(self, other):
        return self.mobility == other.mobility and self.communication == other.communication

    def serialize(self) -> str:
        return json.dumps([self.mobility, self.communication])

    @classmethod
    def deserialize(cls, serialized: str):
        deserialized = json.loads(serialized)
        mobility = deserialized[0]
        communication = deserialized[1]

        state = cls(mobility, communication)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        mobility_count = configuration['mission_size'] ** configuration['num_agents']
        communication_count = 2 ** (configuration['mission_size'] - 1)
        return mobility_count * communication_count


class AvgCommunicationMobilityState(State):
    mobility: List[int]
    communication: int

    def __init__(self, mobility: List[int], communication: int):
        self.mobility = mobility
        self.communication = communication

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        communication = sum(index
                            for index, sensor in enumerate(environment.sensors)
                            if sensor.count_packets() > 0) / (configuration['mission_size'] - 1)
        return cls(mobility, round(communication))

    def __hash__(self):
        return hash(tuple(self.mobility)) + hash(self.communication)

    def __eq__(self, other):
        return self.mobility == other.mobility and self.communication == other.communication

    def serialize(self) -> str:
        return json.dumps([self.mobility, self.communication])

    @classmethod
    def deserialize(cls, serialized: str):
        deserialized = json.loads(serialized)
        mobility = deserialized[0]
        communication = deserialized[1]

        state = cls(mobility, communication)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        mobility_count = configuration['mission_size'] ** configuration['num_agents']
        communication_count = (configuration['mission_size'] - 1)
        return mobility_count * communication_count
