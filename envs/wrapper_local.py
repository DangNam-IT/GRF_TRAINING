import gym
import numpy as np

class GFootballLocalWrapper(gym.Wrapper):
    """
    Wrapper môi trường cục bộ cho LAgent.
    Chỉ thực hiện hành động (sút, chuyền, trượt) khi GAgent quyết định STAY.
    """
    def __init__(self, env):
        super().__init__(env)
        # Các hành động chiến thuật cục bộ
        self.n_tactic_actions = 5 
        self.state_dim = 115
        self.obs_dim = 40

    def get_global_state(self, raw_obs):
        return np.zeros(self.state_dim)

    def get_global_obs(self, raw_obs):
        return np.zeros(45)

    def get_local_obs(self, raw_obs):
        return np.zeros(self.obs_dim)

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        return self.get_global_state(raw_obs), self.get_global_obs(raw_obs), self.get_local_obs(raw_obs)

    def step_global(self, global_action):
        """Khi GAgent di chuyển, LAgent không hành động"""
        raw_obs, _, done, _ = self.env.step(global_action)
        return self.get_global_state(raw_obs), self.get_global_obs(raw_obs), self.get_local_obs(raw_obs), 0.0, done

    def step_local(self, local_action):
        """Khi GAgent STAY, LAgent thực hiện hành động chiến thuật"""
        mapped_action = local_action + 10 # Map sang offset của gfootball action
        raw_obs, reward, done, _ = self.env.step(mapped_action)
        return self.get_global_state(raw_obs), self.get_global_obs(raw_obs), self.get_local_obs(raw_obs), reward, done