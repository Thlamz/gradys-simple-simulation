from pathlib import Path
from typing import TypedDict, Callable, Optional, Union


class QLearningParameters(TypedDict):
    # Q Learning parameters
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float
    reward_function: Callable
    qtable_initialization_value: float
    qtable_file: Union[Path, None]
    qtable_format: Callable


class SimulationConfiguration(TypedDict):
    controller: object
    controller_config: Union[QLearningParameters, ]

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
    target_total_training_time: Optional[int]
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
