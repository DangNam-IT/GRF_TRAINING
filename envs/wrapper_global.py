import gym
import numpy as np
from utils.energy_field import get_total_energy_field

class GFootballGlobalWrapper(gym.Wrapper):
    """
    Wrapper môi trường toàn cục cho GAgent.
    Chỉ cho phép 8 hướng di chuyển + 2 hành động dừng.
    """
    def __init__(self, env):
        super().__init__(env)
        # 10 hành động: 0 (Idle), 1-8 (Di chuyển 8 hướng), 9 (Dừng chiến thuật)
        self.action_space = gym.spaces.Discrete(10)
        
        # Kích thước state và obs giả định (cần tùy chỉnh theo gfootball observation thực tế)
        self.state_dim = 115 
        self.obs_dim = 45    

    def get_global_state(self, raw_obs):
        # Trích xuất state toàn cục từ gfootball
        return np.zeros(self.state_dim) # Placeholder

    def get_partial_observation(self, raw_obs):
        """
        raw_obs lúc này là một list gồm 11 dictionaries (do control 11 người).
        Hàm này cần trả về một mảng chứa 11 partial_obs.
        """
        obs_list = []
        # Lặp qua dữ liệu raw của từng cầu thủ
        for i in range(len(raw_obs)):
            ray_info = np.zeros(40) # Placeholder: Cần thay bằng logic tính ray của bạn
            
            # Giả lập tính Energy Field cho cầu thủ thứ i
            # Ở thực tế bạn trích xuất tọa độ: agent_pos = raw_obs[i]['left_team'][i]
            agent_pos = [0, 0] 
            goals = [[1, 0]]
            obstacles = [[0.5, 0.5]]
            energy_val = get_total_energy_field(agent_pos, goals, obstacles)
            
            energy_field_info = np.array([energy_val]*5)
            obs_i = np.concatenate([ray_info, energy_field_info])
            obs_list.append(obs_i)
            
        return np.array(obs_list) # Kích thước sẽ là (11, obs_dim)

    def step(self, actions):
        """
        Nhận vào mảng 11 actions.
        """
        raw_obs, rewards, done, info = self.env.step(actions)
        
        next_state = self.get_global_state(raw_obs)
        next_obs = self.get_partial_observation(raw_obs)
        
        # Cập nhật energy reward cho từng cầu thủ
        energy_rewards = []
        for i in range(len(rewards)):
            # Cộng reward mặc định của gfootball với Energy Reward của tác tử i
            e_reward = rewards[i] + next_obs[i][-1] 
            energy_rewards.append(e_reward)
            
        return next_state, next_obs, energy_rewards, done

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        state = self.get_global_state(raw_obs)
        obs = self.get_partial_observation(raw_obs)
        return state, obs 