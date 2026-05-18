import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, obs):
        return F.softmax(self.net(obs), dim=-1)

class CentralCriticNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_agents=11):
        super(CentralCriticNetwork, self).__init__()
        # Input bao gồm State toàn cục + Hành động one-hot của tất cả 11 agents
        self.input_dim = state_dim + (n_actions * n_agents)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents) # Xuất Q-value cho mỗi agent
        )

    def forward(self, state, joint_action_one_hot):
        # Flatten joint_action_one_hot từ (Batch, 11, n_actions) -> (Batch, 11 * n_actions)
        action_flat = joint_action_one_hot.view(joint_action_one_hot.size(0), -1)
        x = torch.cat([state, action_flat], dim=-1)
        return self.net(x)