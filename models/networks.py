from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ActorNetwork(nn.Module):
    """Mạng Actor: nhận obs của từng agent → phân phối xác suất hành động."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            probs: (batch, n_actions) — xác suất softmax
        """
        return F.softmax(self.net(obs), dim=-1)


class CentralCriticNetwork(nn.Module):
    """Mạng Critic trung tâm: nhận global state + joint actions → Q-value mỗi agent."""

    def __init__(self, state_dim: int, n_actions: int, n_agents: int = 11) -> None:
        super().__init__()
        # Input = State toàn cục + Hành động one-hot của tất cả agents
        self.input_dim: int = state_dim + (n_actions * n_agents)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents),  # Q-value cho mỗi agent
        )

    def forward(self, state: Tensor, joint_action_one_hot: Tensor) -> Tensor:
        """
        Args:
            state:                (batch, state_dim)
            joint_action_one_hot: (batch, n_agents, n_actions)
        Returns:
            q_values: (batch, n_agents)
        """
        # Flatten (batch, n_agents, n_actions) → (batch, n_agents * n_actions)
        action_flat = joint_action_one_hot.view(joint_action_one_hot.size(0), -1)
        x = torch.cat([state, action_flat], dim=-1)
        return self.net(x)