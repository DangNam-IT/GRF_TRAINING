import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.obses = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def store(self, state, obses, actions, rewards, next_state, done):
        """Lưu trữ transition của toàn bộ 11 agents trong 1 step"""
        self.states.append(state)
        self.obses.append(obses)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.obses.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        
    def get_data(self):
        return (
            np.array(self.states),
            np.array(self.obses),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones)
        )