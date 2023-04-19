from pathlib import Path
from typing import TypedDict, Callable, Union, Literal, Optional


class SimulationConfiguration(TypedDict):
    controller: Callable
    """Controller used to generate controls at every simulation step"""

    # State parameters
    state: Callable

    # Environment parameters
    mission_size: int
    num_agents: int
    sensor_generation_frequency: int
    sensor_generation_probability: float
    sensor_packet_lifecycle: Optional[int]
    maximum_simulation_steps: int

    # Q Learning parameters
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float
    reward_function: Callable
    qtable_initialization_value: float
    qtable_file: Union[Path, None]
    qtable_format: Callable

    # Simulation parameters
    training: bool
    step_by_step: bool
    plots: bool
    verbose: bool


class SimulationResults(TypedDict):
    max_possible_throughput: float
    expected_throughput: float
    avg_throughput: float
    config: SimulationConfiguration
    controller: dict
