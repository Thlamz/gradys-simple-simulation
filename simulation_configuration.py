from pathlib import Path
from typing import TypedDict, Callable, Optional, Union, Type

import torch.optim


class QLearningParameters(TypedDict):
    # Regular QLearning parameters
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float
    reward_function: Callable

    # QTable parameters
    qtable_initialization_value: float
    qtable_format: Callable


class DQLearnerParameters(TypedDict):
    # Regular QLearning parameters
    reward_function: Callable
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float

    # Network training parameters
    memory_size: int
    batch_size: int
    target_network_update_rate: int

    # Network architecture
    num_hidden_layers: int
    hidden_layer_size: int


class SimulationConfiguration(TypedDict):
    controller: object
    controller_config: Union[QLearningParameters, DQLearnerParameters]
    model_file: Union[Path, None]

    # State parameters
    state: Callable

    # Environment parameters
    mission_size: int
    num_agents: int
    sensor_generation_frequency: int
    sensor_generation_probability: float
    sensor_packet_lifecycle: Optional[int]

    # Simulation parameters
    maximum_simulation_steps: int
    target_total_training_steps: Optional[int]
    training: bool
    step_by_step: bool
    testing_repetitions: int
    plots: bool
    verbose: bool


class SimulationResults(TypedDict):
    max_possible_throughput: float
    expected_throughput: float
    avg_throughput: float
    config: SimulationConfiguration
    controller: dict
