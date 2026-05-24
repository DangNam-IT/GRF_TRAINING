import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.networks import ActorNetwork, CentralCriticNetwork

class HES_COMA_Agent:
    def __init__(self, state_dim, obs_dim, n_actions, n_agents=11, lr=5e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.gamma = gamma

        self.actor = ActorNetwork(obs_dim, n_actions).to(self.device)
        self.critic = CentralCriticNetwork(state_dim, n_actions, n_agents).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def get_actions(self, obs_batch):
        """Lấy hành động cho 11 agents cùng lúc"""
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
        return actions.cpu().numpy(), probs.cpu().numpy()

    def update(self, buffer):
        states, obses, actions, rewards, next_states, dones = buffer.get_data()
        if len(states) == 0: return

        # 1. Chuyển đổi dữ liệu sang Tensor hệ thống
        states = torch.FloatTensor(states).to(self.device)                # (Batch, State_Dim)
        obses = torch.FloatTensor(obses).to(self.device)                  # (Batch, 11, Obs_Dim)
        actions = torch.LongTensor(actions).to(self.device)               # (Batch, 11)
        next_states = torch.FloatTensor(next_states).to(self.device)      # (Batch, State_Dim)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)     # (Batch, 1)
        
        # ĐÚNG CHUẨN: Giữ nguyên Tensor phần thưởng độc lập dạng (Batch, 11) không lấy .mean()
        rewards = torch.FloatTensor(rewards).to(self.device)              

        batch_size = states.size(0)
        joint_actions_onehot = F.one_hot(actions, self.n_actions).float() # (Batch, 11, n_actions)
        
        # --- BƯỚC CỐT LÕI: TÍNH TOÁN TOÀN BỘ GIÁ TRỊ Q QUA CRITIC HIỆN TẠI (TRƯỚC KHI CẬP NHẬT) ---
        # [Bước 20]: Tính toán Q-values thô cho hành động thực tế đã chọn
        q_values = self.critic(states, joint_actions_onehot)              # (Batch, 11)
        
        # [Bước 21]: Tính Counterfactual Baseline toán học dựa trên Critic hiện tại
        baseline = torch.zeros_like(q_values).to(self.device)             # (Batch, 11)
        probs = self.actor(obses.view(-1, obses.shape[-1])).view(batch_size, self.n_agents, self.n_actions)
        
        for i in range(self.n_agents):
            for a in range(self.n_actions):
                temp_joint_actions = actions.clone()
                temp_joint_actions[:, i] = a
                temp_joint_onehot = F.one_hot(temp_joint_actions, self.n_actions).float()
                
                # Tính Q trung tâm của cấu hình hành động phản thực tế
                q_temp = self.critic(states, temp_joint_onehot)[:, i].detach()
                baseline[:, i] += probs[:, i, a] * q_temp
                
        # [Bước 24]: Tính lợi thế phản thực tế (Counterfactual Advantage) dựa trên mạng cũ
        advantage = q_values.detach() - baseline
        
        # [Bước 22]: Tính toán Q-value của trạng thái tiếp theo để làm bia mục tiêu TD-error
        with torch.no_grad():
            next_probs = self.actor(obses.view(-1, obses.shape[-1])).view(batch_size, self.n_agents, self.n_actions)
            next_actions = next_probs.argmax(dim=-1)
            next_joint_actions_onehot = F.one_hot(next_actions, self.n_actions).float()
            next_q_values = self.critic(next_states, next_joint_actions_onehot) # (Batch, 11)
            
        td_target = rewards + self.gamma * next_q_values * (1 - dones)
        
        # --- BƯỚC CẬP NHẬT THAM SỐ CHÍNH THỨC ---
        # [Bước 23]: Tối thiểu hóa sai số bình phương TD-error để cập nhật Critic
        critic_loss = F.mse_loss(q_values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # [Bước 25]: Thực thi Policy Gradient để tối ưu hóa Actor
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)                                # (Batch, 11)
        actor_loss = -(log_probs * advantage).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # [Bước 27]: GIẢI PHÓNG BỘ NHỚ ĐỆM ĐỂ TRÁNH TRÀN DATA
        buffer.clear()

    def save_model(self, filepath):
        """Lưu trọng số và cấu hình không gian môi trường của Agent"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': {
                'state_dim': self.critic.input_dim - (self.n_actions * self.n_agents),
                'obs_dim': self.actor.net[0].in_features,
                'n_actions': self.n_actions,
                'n_agents': self.n_agents
            }
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        """Tải lại trọng số cho Agent từ file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Đã tải thành công mô hình từ: {filepath}")