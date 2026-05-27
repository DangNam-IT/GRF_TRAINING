from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from models.networks import ActorNetwork, CentralCriticNetwork
from utils.buffer import RolloutBuffer


class HES_COMA_Agent:
    """
    Hierarchical Energy-field Shared COMA Agent.

    Dùng chung cho cả GAgent (Phase 1) và LAgent (Phase 2).
    Kiến trúc: Actor (per-agent) + Centralised Critic (COMA counterfactual baseline).
    """

    def __init__(
        self,
        state_dim:  int,
        obs_dim:    int,
        n_actions:  int,
        n_agents:   int   = 11,
        lr:         float = 5e-4,
        gamma:      float = 0.99,
    ) -> None:
        self.device:    torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions: int          = n_actions
        self.n_agents:  int          = n_agents
        self.gamma:     float        = gamma

        self.actor:  ActorNetwork          = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic: CentralCriticNetwork  = CentralCriticNetwork(state_dim, n_actions, n_agents).to(self.device)

        self.actor_optimizer:  torch.optim.Optimizer = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optimizer: torch.optim.Optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def get_actions(
        self,
        obs_batch: NDArray[np.float32],  # (n_agents, obs_dim)
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """
        Lấy hành động cho 11 agents cùng lúc (inference, no grad).

        Returns:
            actions: (n_agents,) — chỉ số hành động.
            probs:   (n_agents, n_actions) — xác suất mỗi hành động.
        """
        obs_tensor: Tensor = torch.FloatTensor(obs_batch).to(self.device)
        with torch.no_grad():
            probs: Tensor = self.actor(obs_tensor)
            dist  = torch.distributions.Categorical(probs)
            actions: Tensor = dist.sample()
        return actions.cpu().numpy(), probs.cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> None:
        """
        Cập nhật tham số Actor và Critic theo thuật toán COMA.

        Args:
            buffer: RolloutBuffer chứa dữ liệu episode.
        """
        states_np, obses_np, actions_np, rewards_np, next_states_np, dones_np = buffer.get_data()
        if len(states_np) == 0:
            return

        # ── Chuyển sang Tensor ──────────────────────────────────────────────
        states:      Tensor = torch.FloatTensor(states_np).to(self.device)       # (T, state_dim)
        obses:       Tensor = torch.FloatTensor(obses_np).to(self.device)        # (T, n_agents, obs_dim)
        actions:     Tensor = torch.LongTensor(actions_np).to(self.device)       # (T, n_agents)
        rewards:     Tensor = torch.FloatTensor(rewards_np).to(self.device)      # (T, n_agents)
        next_states: Tensor = torch.FloatTensor(next_states_np).to(self.device)  # (T, state_dim)
        dones:       Tensor = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)  # (T, 1)

        batch_size: int = states.size(0)

        joint_actions_onehot: Tensor = F.one_hot(actions, self.n_actions).float()  # (T, n_agents, n_actions)

        # ── Q-values hiện tại ───────────────────────────────────────────────
        q_values: Tensor = self.critic(states, joint_actions_onehot)   # (T, n_agents)

        # ── Counterfactual Baseline ─────────────────────────────────────────
        baseline: Tensor = torch.zeros_like(q_values).to(self.device)  # (T, n_agents)
        probs: Tensor = self.actor(
            obses.view(-1, obses.shape[-1])
        ).view(batch_size, self.n_agents, self.n_actions)               # (T, n_agents, n_actions)

        for i in range(self.n_agents):
            for a in range(self.n_actions):
                temp_joint_actions: Tensor = actions.clone()
                temp_joint_actions[:, i]   = a
                temp_joint_onehot:  Tensor = F.one_hot(temp_joint_actions, self.n_actions).float()
                q_temp: Tensor = self.critic(states, temp_joint_onehot)[:, i].detach()
                baseline[:, i] += probs[:, i, a] * q_temp

        advantage: Tensor = q_values.detach() - baseline  # (T, n_agents)

        # ── TD-target cho Critic ────────────────────────────────────────────
        with torch.no_grad():
            next_probs: Tensor   = self.actor(
                obses.view(-1, obses.shape[-1])
            ).view(batch_size, self.n_agents, self.n_actions)
            next_actions: Tensor = next_probs.argmax(dim=-1)
            next_joint_onehot: Tensor = F.one_hot(next_actions, self.n_actions).float()
            next_q_values: Tensor = self.critic(next_states, next_joint_onehot)  # (T, n_agents)

        td_target: Tensor = rewards + self.gamma * next_q_values * (1 - dones)

        # ── Cập nhật Critic ─────────────────────────────────────────────────
        critic_loss: Tensor = F.mse_loss(q_values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Cập nhật Actor ──────────────────────────────────────────────────
        dist_actor = torch.distributions.Categorical(probs)
        log_probs: Tensor  = dist_actor.log_prob(actions)           # (T, n_agents)
        actor_loss: Tensor = -(log_probs * advantage).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Giải phóng buffer ───────────────────────────────────────────────
        buffer.clear()

    def save_model(self, filepath: str, episode: int) -> None:
        """Lưu trọng số và cấu hình không gian môi trường của Agent."""
        file_name = f"{filepath}_ep_{episode}.pth"
        import os
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        checkpoint = {
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "config": {
                "state_dim":  self.critic.input_dim - (self.n_actions * self.n_agents),
                "obs_dim":    self.actor.net[0].in_features,
                "n_actions":  self.n_actions,
                "n_agents":   self.n_agents,
            },
        }
        torch.save(checkpoint, file_name)
        print(f"Đã lưu mô hình tại: {file_name}")

    def load_model(self, filepath: str, episode: int) -> None:
        """Tải lại trọng số cho Agent từ file."""
        file_name = f"{filepath}_ep_{episode}.pth"
        checkpoint: dict = torch.load(file_name, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        print(f"Đã tải thành công mô hình từ: {file_name}")