import argparse
import asyncio
import math
import multiprocessing
from pathlib import Path

from DQNLearner import DQNLearner
from QLearning import SparseQTable, QLearning
from campaign import CampaignManager
from rewards import smooth_unique_packets, unique_packets
from state import CommunicationMobilityPacketsState

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='Gradys Simple Simulation')
    parser.add_argument('--max_processes', required=False, default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    campaign_manager = CampaignManager(Path("analysis/results.json"), max_processes=args.max_processes)

    # controller_config_permutation_dict = {
    #     'reward_function': [unique_packets],
    #     'epsilon_start': [1],
    #     'epsilon_end': [0.1],
    #     'learning_rate': [0.0005],
    #     'gamma': [0.99],
    #     'memory_size': [10_000],
    #     'batch_size': [128],
    #     'hidden_layer_size': [128],
    #     'num_hidden_layers': [2],
    #     'target_network_update_rate': [10_000],
    #     'optimizing_rate': [10]
    # }
    # keys, values = zip(*controller_config_permutation_dict.items())
    # controller_config_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    # asyncio.run(campaign_manager.run_campaign(
    #     {
    #         'num_agents': [1],
    #         'mission_size': [10],
    #         'sensor_generation_probability': 0.1,
    #         'sensor_packet_lifecycle': math.inf,
    #         'controller': DQNLearner,
    #         'controller_config': controller_config_permutations,
    #         'state': CommunicationMobilityPacketsState,
    #         'repetitions': [1],
    #         'plots': True
    #     },
    #     ['repetitions', 'mission_size', 'num_agents', 'controller_config'],
    #     {
    #         'training_steps': 1_000_000,
    #         'testing_steps': 10_000,
    #         'live_testing_frequency': 1_000_000,
    #         'testing_repetitions': 1
    #     }
    # ))

    for num_agents in [1, 2, 4]:
        for mission_size in [10, 20, 40]:
            for repetition in [1, 2, 3]:
                asyncio.run(campaign_manager.run_campaign(
                    {
                        'num_agents': [num_agents],
                        'mission_size': [mission_size],
                        'sensor_generation_probability': 0.1,
                        'sensor_packet_lifecycle': math.inf,
                        'controller': QLearning,
                        'controller_config': {
                            'epsilon_start': 1.,
                            'epsilon_end': 0.001,
                            'learning_rate': 0.9,
                            'gamma': 0.99,
                            'reward_function': unique_packets,
                            'qtable_initialization_value': 0,
                            'qtable_format': SparseQTable
                        },
                        'state': CommunicationMobilityPacketsState,
                        'repetitions': [repetition],
                    },
                    ['repetitions', 'mission_size', 'num_agents'],
                    {
                        'training_steps': 10_000_000,
                        'testing_steps': 10_000,
                        'live_testing_frequency': 1_000_000,
                        'testing_repetitions': 5
                    }
                ))

    campaign_manager.finalize()
