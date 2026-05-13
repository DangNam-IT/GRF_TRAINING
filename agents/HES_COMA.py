import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.networks import Actor, CentralizedCritic

class HES_COMA_Agent:
    def __init__(self, state_dim, obs_dim, n_actions, lr=1e-3, gamma=0.99):
        # Tự động chọn device phù hợp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(obs_dim, n_actions).to(self.device)
        self.critic = CentralizedCritic(state_dim, n_actions).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.n_actions = n_actions
        self.gamma = gamma

    def get_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), probs.cpu().numpy()

    def update(self, buffer):
        data = buffer.get_data()
        if not data: return
        
        states = torch.FloatTensor(np.array([t[0] for t in data])).to(self.device)
        obses = torch.FloatTensor(np.array([t[1] for t in data])).to(self.device)
        actions = torch.LongTensor(np.array([t[2] for t in data])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[3] for t in data])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[4] for t in data])).to(self.device)
        dones = torch.FloatTensor(np.array([t[5] for t in data])).to(self.device)

        actions_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        
        # 1. Q-value hiện tại
        q_values = self.critic(states, actions_onehot).squeeze()
        
        # 2. Q-value target
        with torch.no_grad():
            next_probs = self.actor(obses) 
            next_actions_onehot = F.one_hot(next_probs.argmax(dim=-1), num_classes=self.n_actions).float()
            next_q_values = self.critic(next_states, next_actions_onehot).squeeze()
            
        td_target = rewards + self.gamma * next_q_values * (1 - dones)
        td_error = td_target - q_values
        
        # Cập nhật Critic
        critic_loss = (td_error ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Tính Baseline và Advantage
        probs = self.actor(obses)
        baseline = 0
        for a in range(self.n_actions):
            a_onehot = F.one_hot(torch.tensor([a]*len(states)).to(self.device), num_classes=self.n_actions).float()
            q_a = self.critic(states, a_onehot).squeeze().detach()
            baseline += probs[:, a] * q_a

        advantage = q_values.detach() - baseline
        
        # Cập nhật Actor
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantage).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()