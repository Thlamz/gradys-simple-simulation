from typing import Optional, TypedDict, List

from simulation_configuration import SimulationResults, SimulationConfiguration


class CampaignConfiguration(TypedDict):
    # Simulation parameters
    training_steps: int
    testing_steps: int
    live_testing_frequency: Optional[int]
    testing_repetitions: int
    concurrent_simulations: bool
    concurrent_testing: bool


class CampaignResults(TypedDict):
    simulation_config: SimulationConfiguration
    campaign_config: CampaignConfiguration
    simulation_results: SimulationResults
    completed_training_steps: int
