from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from typing import Tuple

class ActorNetwork(nn.Module):
    """Mạng Actor: nhận obs của từng agent → phân phối xác suất hành động."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.rnn = nn.GRUCell(128, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, obs: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            obs:          (batch, obs_dim)
            hidden_state: (batch, 128)
        Returns:
            probs:        (batch, n_actions) — xác suất softmax
            hidden_state: (batch, 128) — trạng thái ẩn mới
        """
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, 128)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return F.softmax(q, dim=-1), h


class CentralCriticNetwork(nn.Module):
    """Mạng Critic trung tâm: nhận global state + joint actions → Q-value mỗi agent."""

    def __init__(self, state_dim: int, n_actions: int, n_agents: int = 11) -> None:
        super().__init__()
        # Input = State toàn cục + Hành động one-hot của tất cả agents
        self.input_dim: int = state_dim + (n_actions * n_agents)
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.rnn = nn.GRUCell(256, 128)
        self.fc2 = nn.Linear(128, n_agents)

    def forward(self, state: Tensor, joint_action_one_hot: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            state:                (batch, state_dim)
            joint_action_one_hot: (batch, n_agents, n_actions)
            hidden_state:         (batch, 128)
        Returns:
            q_values:     (batch, n_agents)
            hidden_state: (batch, 128)
        """
        # Flatten (batch, n_agents, n_actions) → (batch, n_agents * n_actions)
        action_flat = joint_action_one_hot.view(joint_action_one_hot.size(0), -1)
        x = torch.cat([state, action_flat], dim=-1)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, 128)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h