from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DDPGFC350EPidSettings:
    hidden_units: int = 350
    num_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 0.0001
    critic_learning_rate: float = 0.001
    batch_size: int = 256
    buffer_size: int = 1000000
    exploration_noise: float = 0.2
    exploration_noise_min: float = 0.02
    exploration_noise_decay_steps: int = 1500000
    action_l2_weight: float = 0.0001


class FullyConnectedELU(nn.Module):
    def __init__(self, input_size: int, output_size: int, settings: DDPGFC350EPidSettings):
        super().__init__()
        layers = []
        last_size = input_size
        for _ in range(settings.num_layers):
            layers.append(nn.Linear(last_size, settings.hidden_units))
            layers.append(nn.ELU())
            last_size = settings.hidden_units
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Actor(nn.Module):
    def __init__(self, obs_size: int, action_size: int, settings: DDPGFC350EPidSettings):
        super().__init__()
        self.network = FullyConnectedELU(obs_size, action_size, settings)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs))


class Critic(nn.Module):
    def __init__(self, obs_size: int, action_size: int, settings: DDPGFC350EPidSettings):
        super().__init__()
        self.network = FullyConnectedELU(obs_size + action_size, 1, settings)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([obs, action], dim=-1))
