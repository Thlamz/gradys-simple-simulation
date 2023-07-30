import json
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from device import device
from environment import Environment
from simulation_configuration import SimulationConfiguration


class State:

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

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized: str):
        pass

    @classmethod
    @abstractmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        pass

    @classmethod
    @abstractmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        pass


class MobilityState(State):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """

    mobility: Tuple[int]

    def __init__(self, mobility: List[int]):
        self.mobility = tuple(mobility)

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        return cls(mobility)

    def serialize(self) -> str:
        return json.dumps(self.mobility)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.mobility, dtype=torch.float32, device=device).unsqueeze(0)

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

    @classmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['num_agents']


class SignedMobilityState(State):
    """
    The state in this scenario has only one key, mobility. It is a tuple of integers where each integer
    mobility[i] represents the waypoint position o agent i
    """

    mobility: Tuple[int]

    def __init__(self, mobility: List[int]):
        self.mobility = tuple(mobility)

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position
                    if agent.reversed
                    else configuration['mission_size'] + agent.position for agent in environment.agents]
        return cls(mobility)

    def serialize(self) -> str:
        return json.dumps(self.mobility)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.mobility, dtype=torch.float32, device=device).unsqueeze(0)

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

    @classmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['num_agents']


class CommunicationMobilityState(State):
    mobility: Tuple[int]
    communication: Tuple[int]

    def __init__(self, mobility: List[int], communication: List[int]):
        self.mobility = tuple(mobility)
        self.communication = tuple(communication)

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        communication = [1 if sensor.count_packets() > 0 else 0 for sensor in environment.sensors]
        return cls(mobility, communication)

    def __hash__(self):
        return hash(self.mobility) + hash(self.communication)

    def __eq__(self, other):
        return self.mobility == other.mobility and self.communication == other.communication

    def serialize(self) -> str:
        return json.dumps([self.mobility, self.communication])

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.mobility + self.communication, dtype=torch.float32, device=device).unsqueeze(0)

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

    @classmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['num_agents'] + (configuration['mission_size'] - 1)


class CommunicationMobilityPacketsState(State):
    mobility: Tuple[int]
    packets: Tuple[int]
    communication: Tuple[int]

    def __init__(self, mobility: List[int], packets: List[int], communication: List[int]):
        self.mobility = tuple(mobility)
        self.packets = tuple(packets)
        self.communication = tuple(communication)

    @classmethod
    def build(cls, configuration: SimulationConfiguration, environment: Environment):
        mobility = [agent.position for agent in environment.agents]
        packets = [len(agent.sources) for agent in environment.agents]
        communication = [1 if sensor.count_packets() > 0 else 0 for sensor in environment.sensors]
        return cls(mobility, packets, communication)

    def __hash__(self):
        return hash(self.mobility) + hash(self.packets) + hash(self.communication)

    def __eq__(self, other):
        return self.mobility == other.mobility and self.packets == other.packets and self.communication == other.communication

    def serialize(self) -> str:
        return json.dumps([self.mobility, self.packets, self.communication])

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.mobility + self.packets + self.communication, dtype=torch.float32, device=device).unsqueeze(0)

    @classmethod
    def deserialize(cls, serialized: str):
        deserialized = json.loads(serialized)
        mobility = deserialized[0]
        packets = deserialized[1]
        communication = deserialized[2]

        state = cls(mobility, packets, communication)
        return state

    @classmethod
    def possible_states(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        mobility_count = configuration['mission_size'] ** configuration['num_agents']
        packets_count = (configuration['mission_size'] - 1) ** configuration['num_agents']
        communication_count = 2 ** (configuration['mission_size'] - 1)
        return mobility_count * packets_count * communication_count

    @classmethod
    def size(cls, configuration: SimulationConfiguration, environment: Environment) -> int:
        return configuration['num_agents'] * 2 + (configuration['mission_size'] - 1)
