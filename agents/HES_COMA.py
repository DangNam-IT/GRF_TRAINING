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

        Thứ tự chuẩn:
          1. Tính Q-values và Counterfactual Baseline dựa trên tham số Critic cũ.
          2. Tính TD-target (no_grad).
          3. Cập nhật Critic (critic_optimizer.step).
          4. Tính advantage từ Q-values đã detach — bảo toàn gradient riêng.
          5. Cập nhật Actor với per-agent advantage (reduction='none' → mean over batch).
          6. Giải phóng buffer.

        [MODULE 1 - FIX 2]: Bảo toàn hàm phần thưởng bất đối xứng
          Critic loss dùng `reduction='none'` + mean theo batch để giữ nguyên
          gradient Q cá thể hóa (T, n_agents) theo Trường Năng lượng.
          Không dùng global `.mean()` vì sẽ san bằng chênh lệch phần thưởng.

        Args:
            buffer: RolloutBuffer chứa dữ liệu episode.
        """
        states_np, obses_np, actions_np, rewards_np, next_states_np, next_obses_np, dones_np = buffer.get_data()
        if len(states_np) == 0:
            # [MODULE 1 - FIX 3]: Giải phóng bộ đệm ngay cả khi buffer rỗng
            buffer.clear()
            return

        # ── Chuyển sang Tensor ──────────────────────────────────────────────
        states:       Tensor = torch.FloatTensor(states_np).to(self.device)        # (T, state_dim)
        obses:        Tensor = torch.FloatTensor(obses_np).to(self.device)         # (T, n_agents, obs_dim)
        actions:      Tensor = torch.LongTensor(actions_np).to(self.device)        # (T, n_agents)
        rewards:      Tensor = torch.FloatTensor(rewards_np).to(self.device)       # (T, n_agents)
        next_states:  Tensor = torch.FloatTensor(next_states_np).to(self.device)   # (T, state_dim)
        next_obses:   Tensor = torch.FloatTensor(next_obses_np).to(self.device)    # (T, n_agents, obs_dim)
        dones:        Tensor = torch.FloatTensor(dones_np).unsqueeze(1).to(self.device)  # (T, 1)

        batch_size: int = states.size(0)

        joint_actions_onehot: Tensor = F.one_hot(actions, self.n_actions).float()  # (T, n_agents, n_actions)

        # ── BƯỚC 1: Q-values và Counterfactual Baseline (dùng tham số Critic CŨ) ──
        # [MODULE 1 - FIX 1]: Toàn bộ tính toán advantage phải hoàn tất
        # trước critic_optimizer.step() để baseline không bị nhiễm bởi tham số mới.
        q_values: Tensor = self.critic(states, joint_actions_onehot)   # (T, n_agents)

        # Counterfactual Baseline: E_a'[Q(s, a'_i, a_{-i})] theo chính sách hiện tại
        baseline: Tensor = torch.zeros_like(q_values)  # (T, n_agents)
        with torch.no_grad():
            probs_for_baseline: Tensor = self.actor(
                obses.view(-1, obses.shape[-1])
            ).view(batch_size, self.n_agents, self.n_actions)  # (T, n_agents, n_actions)

        for i in range(self.n_agents):
            for a in range(self.n_actions):
                temp_joint_actions: Tensor = actions.clone()
                temp_joint_actions[:, i]   = a
                temp_joint_onehot:  Tensor = F.one_hot(temp_joint_actions, self.n_actions).float()
                # Detach để tách khỏi đồ thị tính gradient của Critic
                q_temp: Tensor = self.critic(states, temp_joint_onehot)[:, i].detach()
                baseline[:, i] += probs_for_baseline[:, i, a] * q_temp

        # Advantage = Q(s,a) - Baseline — detach Q khỏi đồ thị Critic
        # để gradient Actor không chạy ngược vào Critic weights.
        advantage: Tensor = q_values.detach() - baseline.detach()  # (T, n_agents)

        # ── BƯỚC 2: TD-target (no_grad) dùng o_{t+1} — Bước 14 algo_logic ──
        # QUAN TRỌNG: Phải dùng next_obses (o_{t+1}), KHÔNG phải obses (o_t).
        # Theo Bước 14: δ = r + γ·Q(s', π(o_{t+1})) - Q(s, a)
        with torch.no_grad():
            next_probs: Tensor    = self.actor(
                next_obses.view(-1, next_obses.shape[-1])   # dùng o_{t+1}
            ).view(batch_size, self.n_agents, self.n_actions)
            next_actions: Tensor  = next_probs.argmax(dim=-1)
            next_joint_onehot: Tensor = F.one_hot(next_actions, self.n_actions).float()
            next_q_values: Tensor = self.critic(next_states, next_joint_onehot)  # (T, n_agents)

        td_target: Tensor = rewards + self.gamma * next_q_values * (1 - dones)  # (T, n_agents)

        # ── BƯỚC 3: Cập nhật Critic ─────────────────────────────────────────
        # [MODULE 1 - FIX 2]: reduction='none' → loss shape (T, n_agents)
        # giữ nguyên hàm giá trị Q cá thể hóa theo Trường Năng lượng.
        # Chỉ mean theo chiều batch (dim=0), KHÔNG san bằng toàn bộ.
        critic_loss_per_agent: Tensor = F.mse_loss(
            q_values, td_target.detach(), reduction='none'
        )                                                     # (T, n_agents)
        critic_loss: Tensor = critic_loss_per_agent.mean()   # scalar — mean theo batch
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── BƯỚC 4: Cập nhật Actor (dùng advantage đã tính từ Critic CŨ) ───
        # Tính lại probs để có gradient flow cho Actor
        probs: Tensor = self.actor(
            obses.view(-1, obses.shape[-1])
        ).view(batch_size, self.n_agents, self.n_actions)     # (T, n_agents, n_actions)

        dist_actor  = torch.distributions.Categorical(probs)
        log_probs: Tensor  = dist_actor.log_prob(actions)     # (T, n_agents)

        # [FIX] Entropy regularization: ngăn Policy Collapse (chỉ chọn 1 hướng)
        # H = -Σ p*log(p) — entropy cao = khám phá nhiều hướng hơn.
        # β_entropy điều chỉnh mức độ khám phá: 0.005 — 0.01 là khoảng tốt cho MARL.
        entropy_bonus: Tensor = dist_actor.entropy()          # (T, n_agents)
        beta_entropy:  float  = 0.01

        # mean() theo batch+agent để tổng hợp gradient — advantage per-agent
        # đã được bảo toàn từ bước tính riêng rẽ ở trên.
        actor_loss: Tensor = -(log_probs * advantage + beta_entropy * entropy_bonus).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)  # [FIX] Gradient clipping
        self.actor_optimizer.step()

        # ── BƯỚC 5: Giải phóng buffer (Memory Leak Guard) ───────────────────
        # [MODULE 1 - FIX 3]: buffer.clear() luôn được gọi cuối update()
        # để giải phóng RAM sau mỗi episode, tránh tích lũy transitions cũ.
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