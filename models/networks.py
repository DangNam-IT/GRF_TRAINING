import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.action_head(x), dim=-1)
        return probs

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(CentralizedCritic, self).__init__()
        # Critic nhận cả trạng thái toàn cục và hành động (dạng one-hot)
        self.fc1 = nn.Linear(state_dim + n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, state, action_one_hot):
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)