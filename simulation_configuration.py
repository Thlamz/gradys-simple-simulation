from pathlib import Path
from typing import TypedDict, Callable


class SimulationConfiguration(TypedDict):
    controller: Callable
    """Controller used to generate controls at every simulation step"""

    # Environment parameters
    mission_size: int
    num_agents: int
    sensor_generation_frequency: int
    maximum_simulation_steps: int

    # Q LEarning parameters
    epsilon_start: float
    epsilon_end: float
    learning_rate: float
    gamma: float
    qtable_initialization_value: float
    qtable_file: Path | None

    # Simulation parameters
    training: bool
    step_by_step: bool
    plots: bool