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
        n_agents:   int ,
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
        obs_batch: NDArray[np.float32],  # (num_envs, n_agents, obs_dim)
        hidden_states: Tensor,           # (num_envs * n_agents, 128)
        epsilon: float = 0.0,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], Tensor]:
        """
        Lấy hành động với GRU và Epsilon-Greedy.
        """
        num_envs = obs_batch.shape[0]
        obs_flat = obs_batch.reshape(-1, obs_batch.shape[-1])
        obs_tensor: Tensor = torch.FloatTensor(obs_flat).to(self.device)
        
        with torch.no_grad():
            probs, new_hidden_states = self.actor(obs_tensor, hidden_states)
            
            # Khám phá Epsilon-Greedy
            if np.random.rand() < epsilon:
                # Lấy hành động ngẫu nhiên
                actions = torch.randint(0, self.n_actions, (num_envs * self.n_agents,), device=self.device)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                
        actions_np = actions.cpu().numpy().reshape(-1, self.n_agents)
        probs_np = probs.cpu().numpy().reshape(-1, self.n_agents, self.n_actions)
        
        return actions_np, probs_np, new_hidden_states

    def update(self, buffer: RolloutBuffer, num_envs: int = 1) -> None:
        """
        Cập nhật tham số Actor và Critic theo thuật toán COMA (với BPTT qua GRU).
        """
        states_np, obses_np, actions_np, rewards_np, next_states_np, next_obses_np, dones_np, active_masks_np = buffer.get_data()
        if len(states_np) == 0:
            buffer.clear()
            return

        # ── Chuyển sang Tensor ──────────────────────────────────────────────
        N = len(states_np)
        T = N // num_envs
        
        # Reshape thành (T, num_envs, ...)
        states:       Tensor = torch.FloatTensor(states_np).view(T, num_envs, -1).to(self.device)
        obses:        Tensor = torch.FloatTensor(obses_np).view(T, num_envs, self.n_agents, -1).to(self.device)
        actions:      Tensor = torch.LongTensor(actions_np).view(T, num_envs, self.n_agents).to(self.device)
        rewards:      Tensor = torch.FloatTensor(rewards_np).view(T, num_envs, self.n_agents).to(self.device)
        next_states:  Tensor = torch.FloatTensor(next_states_np).view(T, num_envs, -1).to(self.device)
        next_obses:   Tensor = torch.FloatTensor(next_obses_np).view(T, num_envs, self.n_agents, -1).to(self.device)
        dones:        Tensor = torch.FloatTensor(dones_np).view(T, num_envs, 1).to(self.device)
        active_masks: Tensor = torch.FloatTensor(active_masks_np).view(T, num_envs, self.n_agents).to(self.device)

        joint_actions_onehot: Tensor = F.one_hot(actions, self.n_actions).float()

        # ── Forward qua thời gian (BPTT) cho Critic và Actor ─────────────────
        # Khởi tạo hidden states
        h_actor = torch.zeros(num_envs * self.n_agents, 128, device=self.device)
        h_critic = torch.zeros(num_envs, 128, device=self.device)
        
        h_actor_next = torch.zeros(num_envs * self.n_agents, 128, device=self.device)
        h_critic_next = torch.zeros(num_envs, 128, device=self.device)

        q_values_list = []
        baseline_list = []
        next_q_values_list = []
        probs_list = []

        for t in range(T):
            # Reset hidden state nếu episode trước đó đã kết thúc
            if t > 0:
                mask = (1 - dones[t-1]).view(num_envs, 1)
                h_actor = h_actor * mask.repeat_interleave(self.n_agents, dim=0)
                h_critic = h_critic * mask
                h_actor_next = h_actor_next * mask.repeat_interleave(self.n_agents, dim=0)
                h_critic_next = h_critic_next * mask

            obs_t = obses[t].view(-1, obses.shape[-1])
            state_t = states[t]
            action_onehot_t = joint_actions_onehot[t]

            # 1. Critic tính Q-values
            q_t, h_critic_new = self.critic(state_t, action_onehot_t, h_critic)
            
            # 2. Actor tính Probs
            probs_t, h_actor_new = self.actor(obs_t, h_actor)
            probs_t = probs_t.view(num_envs, self.n_agents, self.n_actions)

            # 3. Counterfactual Baseline
            baseline_t = torch.zeros_like(q_t)
            with torch.no_grad():
                for i in range(self.n_agents):
                    for a in range(self.n_actions):
                        temp_joint_onehot = action_onehot_t.clone()
                        temp_joint_onehot[:, i, :] = 0
                        temp_joint_onehot[:, i, a] = 1.0
                        q_temp, _ = self.critic(state_t, temp_joint_onehot, h_critic)
                        baseline_t[:, i] += probs_t[:, i, a].detach() * q_temp[:, i].detach()

            # 4. TD-Target (Next Q-values)
            next_obs_t = next_obses[t].view(-1, next_obses.shape[-1])
            next_state_t = next_states[t]
            with torch.no_grad():
                next_probs_t, h_actor_next_new = self.actor(next_obs_t, h_actor_next)
                next_probs_t = next_probs_t.view(num_envs, self.n_agents, self.n_actions)
                next_actions_t = next_probs_t.argmax(dim=-1)
                next_joint_onehot = F.one_hot(next_actions_t, self.n_actions).float()
                next_q_t, h_critic_next_new = self.critic(next_state_t, next_joint_onehot, h_critic_next)

            q_values_list.append(q_t)
            baseline_list.append(baseline_t)
            next_q_values_list.append(next_q_t)
            probs_list.append(probs_t)

            h_actor = h_actor_new
            h_critic = h_critic_new
            h_actor_next = h_actor_next_new
            h_critic_next = h_critic_next_new

        q_values = torch.stack(q_values_list, dim=0)          # (T, num_envs, n_agents)
        baseline = torch.stack(baseline_list, dim=0)
        next_q_values = torch.stack(next_q_values_list, dim=0)
        probs = torch.stack(probs_list, dim=0)                # (T, num_envs, n_agents, n_actions)

        # Advantage = Q - Baseline
        advantage = q_values.detach() - baseline.detach()

        # TD Target
        td_target = rewards + self.gamma * next_q_values * (1 - dones.expand_as(next_q_values))

        # ── Cập nhật Critic ─────────────────────────────────────────
        critic_loss_per_agent = F.mse_loss(q_values, td_target.detach(), reduction='none')
        critic_loss = (critic_loss_per_agent * active_masks).sum() / (active_masks.sum() + 1e-8)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # ── Cập nhật Actor ─────────────────────────────────────────
        dist_actor = torch.distributions.Categorical(probs)
        log_probs = dist_actor.log_prob(actions)
        entropy_bonus = dist_actor.entropy()
        beta_entropy = 0.01

        actor_loss = - ((log_probs * advantage + beta_entropy * entropy_bonus) * active_masks).sum() / (active_masks.sum() + 1e-8)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

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
                "obs_dim":    self.actor.fc1.in_features,
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