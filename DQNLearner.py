import random
from collections import deque
from typing import Optional, TypedDict, Deque, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from base_serializer import base_id_to_tuple, tuple_to_base_id
from control import Control, generate_random_control, MobilityCommand
from controller import Controller
from device import device
from environment import Environment
from rng import rng
from simulation_configuration import SimulationConfiguration, DQLearnerParameters
from state import State

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Defining the Deep neural network which will be optimized to approximate the Q Value
    """

    def __init__(self, state_size: int, control_size: int, hidden_layers: int, hidden_layer_size: int):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(state_size, hidden_layer_size)
        self.hidden_layers = \
            nn.ModuleList(nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(hidden_layers))
        self.output_layer = nn.Linear(hidden_layer_size, control_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class Memory(TypedDict):
    state: torch.Tensor
    control: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor


class MemoryBuffer:
    """
    This class stores recollections of the previous 'memory_size' iterations of the simulation
    """

    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        memory_size: int = configuration['controller_config']['memory_size']

        state_size = configuration['state'].size(configuration, environment)

        self.state_memory = torch.zeros((memory_size, state_size), dtype=torch.float32, device=device)
        self.control_memory = torch.zeros((memory_size, 1), dtype=torch.int64, device=device)
        self.next_state_memory = torch.zeros((memory_size, state_size), dtype=torch.float32, device=device)
        self.reward_memory = torch.zeros((memory_size, 1), dtype=torch.float32, device=device)

        self.current_index = 0
        self.current_size = 0

        self.size = memory_size

    def append(self, memory: Memory):
        self.state_memory[self.current_index, :] = memory['state']
        self.control_memory[self.current_index, :] = memory['control']
        self.next_state_memory[self.current_index, :] = memory['next_state']
        self.reward_memory[self.current_index, :] = memory['reward']

        self.current_index += 1
        self.current_size = min(self.current_size + 1, self.size)

        if self.current_index >= self.size:
            self.current_index = 0

    def sample(self, number) -> Memory:
        indexes = torch.tensor(rng.choice(self.current_size, number, replace=False, shuffle=False), device=device)

        return {
            "state": self.state_memory[indexes],
            "control": self.control_memory[indexes],
            "next_state": self.next_state_memory[indexes],
            "reward": self.reward_memory[indexes]
        }

    def __len__(self):
        return self.current_size


class DQNLearner(Controller):
    def __init__(self, configuration: SimulationConfiguration, environment: Environment):
        super().__init__(configuration, environment)

        self.configuration: SimulationConfiguration = configuration
        self.controller_configuration: DQLearnerParameters = configuration['controller_config']

        if configuration['verbose']:
            print(f"Executing on device {device}")

        if self.configuration['verbose']:
            state_size = configuration['state'].possible_states(configuration, environment)
            control_size = len(MobilityCommand) ** configuration['num_agents']
            print(f"QTable size: {state_size * control_size}")

        self.training = self.configuration['training']

        state_size = configuration['state'].size(configuration, environment)
        control_size = 2 ** configuration['num_agents']

        # region Setting up twin neural networks

        # The first neural network is the policy network. This network is the one that will be optimized
        # every simulation step and is the one responsible for evaluating the Q Values that will dictate
        # the algorithms controls.
        self.policy_model = DQN(state_size,
                                control_size,
                                self.controller_configuration['num_hidden_layers'],
                                self.controller_configuration['hidden_layer_size']).to(device)

        # Loading existing network if it exists
        if configuration['model_file'] is not None and configuration['model_file'].is_file():
            self.policy_model.load_state_dict(torch.load(configuration['model_file']))
            self.policy_model.training = self.training

        if self.training:
            # The second neural network is the target network. It isn't directly optimized, every
            # 'target_network_update_rate' steps the weights from the policy network are copied into
            # this one. It is only necessary during training.
            self.target_model = DQN(state_size,
                                    control_size,
                                    self.controller_configuration['num_hidden_layers'],
                                    self.controller_configuration['hidden_layer_size']).to(device)
            # This network is initialized with the same weights as the policy network
            self.target_model.load_state_dict(self.policy_model.state_dict())

            # Update rate of the target network
            if self.controller_configuration['target_network_update_rate'] == 'auto':
                self.target_network_update_rate = self.configuration['simulation_steps'] // 100
            else:
                self.target_network_update_rate = self.controller_configuration['target_network_update_rate']
        # endregion

        if self.training:
            self.optimizer = optim.AdamW(self.policy_model.parameters(),
                                         lr=self.controller_configuration['learning_rate'],
                                         amsgrad=True)
            self.criterion = nn.SmoothL1Loss()

        self.memory_buffer = MemoryBuffer(configuration, environment)

        # Statistical variables, only useful to generate metrics
        self.total_reward = 0
        self.cum_avg_rewards = []
        self.cum_avg_rewards_buffer = []
        self.losses = []
        self.losses_buffer = []

        # Variables that control the epsilon greedy approach
        self.epsilon = self.controller_configuration['epsilon_start']
        self.epsilon_start = self.controller_configuration['epsilon_start']
        self.epsilon_end = self.controller_configuration['epsilon_end']
        self.epsilon_horizon = self.configuration['simulation_steps']
        self.epsilons = []
        self.epsilons_buffer = []

        self.statistics_bin_size = self.configuration['simulation_steps'] // 1000

        # Control variables. These store the last state and control visited in the simulation
        self.last_state: Optional[State] = None
        self.last_control: Optional[Control] = None

    def compute_statistics(self, value, buffer: List, statistic: List):
        buffer.append(value)
        if len(buffer) >= self.statistics_bin_size:
            statistic.append(sum(buffer) / len(buffer))
            buffer.clear()

    def decay_epsilon(self) -> None:
        if self.epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon *= (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_horizon)

        self.compute_statistics(self.epsilon, self.epsilons_buffer, self.epsilons)

    def compute_reward(self, simulation_step):
        return self.controller_configuration['reward_function'](self, simulation_step)

    def optimize(self, simulation_step):
        batch_size = self.controller_configuration['batch_size']

        # Sampling 'batch_size' examples from the memory buffer. This is used to implement a strategy called
        # memory replay. Instead of training the model with the most current state and action pairs, previous
        # simulation iterations are recalled and fed through the network. This improves traininig performance.
        memory_batch = self.memory_buffer.sample(batch_size)

        state_batch = memory_batch['state']
        next_state_batch = memory_batch['next_state']
        control_batch = memory_batch['control']
        reward_batch = memory_batch['reward'].view((batch_size,))

        # Batch is fed through the policy model, the result is the estimated QValue for every action/control pair
        state_q_values = self.policy_model(state_batch)
        # Gathering only the QValues from the actions actually performed by the agents
        state_action_q_values = state_q_values.gather(1, control_batch).view((batch_size,))

        with torch.no_grad():
            # The next state is the state observed after applying a control to a state. The target model is used to
            # sample the max QValue of the next state's actions
            next_state_q_values = self.target_model(next_state_batch).max(1)[0]

        # Applying bellman's equation to calculate the expected q values
        expected_q_values = (next_state_q_values * self.controller_configuration['gamma']) + reward_batch

        # A loss function is calculated based on how far state_action_q_values were from expected_q_values
        loss = self.criterion(state_action_q_values, expected_q_values)

        # Optimize the model to minimize this loss
        self.optimizer.zero_grad()
        loss.backward()

        self.compute_statistics(loss.item(), self.losses_buffer, self.losses)

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

    def get_control(self, simulation_step: int, current_state: State, current_control: Control) -> Control:
        reward = self.compute_reward(simulation_step)
        self.total_reward += reward
        if simulation_step > 0:
            self.compute_statistics(self.total_reward / simulation_step, self.cum_avg_rewards_buffer, self.cum_avg_rewards)

        if self.training and self.last_state is not None:
            self.memory_buffer.append({
                'state': self.last_state.to_tensor(),
                'control': torch.tensor([tuple_to_base_id(tuple(cmd.value for cmd in self.last_control.mobility), 2)],
                                        device=device),
                'next_state': current_state.to_tensor(),
                'reward': torch.tensor([reward], device=device)
            })

            if len(self.memory_buffer) >= self.controller_configuration['batch_size'] \
                    and simulation_step % self.controller_configuration['optimizing_rate'] == 0:
                self.optimize(simulation_step)
            else:
                # Propagating last loss if loss wasn't updated
                self.compute_statistics(self.losses[-1] if len(self.losses) > 0 else 0, self.losses_buffer, self.losses)

            if simulation_step % self.target_network_update_rate == 0 and simulation_step:
                policy_model_state_dict = self.policy_model.state_dict()
                self.target_model.load_state_dict(policy_model_state_dict)

            self.decay_epsilon()

        if self.training and rng.random() < self.epsilon:
            control = generate_random_control(self.configuration, self.environment)
        else:
            with torch.no_grad():
                state_tensor = current_state.to_tensor()
                q_values = self.policy_model(state_tensor)
                policy = q_values.max(1)[1].item()
                mobility = base_id_to_tuple(policy, 2, self.configuration['num_agents'])
                control = Control(tuple(MobilityCommand(value) for value in mobility))

        self.last_state = current_state
        self.last_control = control

        return control

    def finalize(self) -> dict:
        if self.configuration['model_file'] is not None and self.training:
            torch.save(self.policy_model.state_dict(), self.configuration['model_file'])

        if self.configuration['plots']:
            sns.lineplot(data=self.cum_avg_rewards).set(title="Cum Avg Train Rewards")
            plt.show()

            if self.training:
                sns.lineplot(data=self.losses).set(title="Loss")
                plt.show()

        bins = np.linspace(0, self.configuration['simulation_steps'], 1000)
        if len(self.cum_avg_rewards_buffer) < self.statistics_bin_size - 1:
            self.cum_avg_rewards.append(sum(self.cum_avg_rewards_buffer) / (len(self.cum_avg_rewards_buffer) or 1))

        if len(self.losses_buffer) < self.statistics_bin_size - 1:
            self.losses.append(sum(self.losses_buffer) / (len(self.losses_buffer) or 1))

        if len(self.epsilons_buffer) < self.statistics_bin_size - 1:
            self.epsilons.append(sum(self.losses_buffer) / (len(self.losses_buffer) or 1))
        return {
            'avg_reward': self.total_reward / self.configuration['simulation_steps'],
            'cum_avg_rewards': self.cum_avg_rewards,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'step_bins': bins[:-1].tolist()
        }
