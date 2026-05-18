import gym
import numpy as np
from utils.energy_field import EnergyFieldDefiner

class GFootballGlobalWrapper(gym.Wrapper):
    def __init__(self, env, num_agents=11):
        super().__init__(env)
        self.num_agents = num_agents
        self.action_space = gym.spaces.MultiDiscrete([10] * num_agents)
        self.energy_definer = EnergyFieldDefiner()
        self.state_dim = 46 
        self.obs_dim = 45   
        self.last_raw_obs = None

    def _extract_positions(self, raw_obs):
        base_obs = raw_obs[0] 
        left_team = np.array(base_obs['left_team'])
        right_team = np.array(base_obs['right_team'])
        ball = np.array(base_obs['ball'][:2])
        
        # --- BẮT ĐẦU LOGIC XỬ LÝ PHẠT GÓC ---
        # Kiểm tra xem có phải phạt góc không (Bóng nằm ở 4 góc sân)
        # Tọa độ góc sân xấp xỉ X = +/- 1.0, Y = +/- 0.42
        is_corner_kick = (abs(abs(ball[0]) - 1.0) < 0.05) and (abs(abs(ball[1]) - 0.42) < 0.05)
        
        if is_corner_kick:
            # 1. Đổi mục tiêu (Lực hút) vào trong vòng cấm địa thay vì quả bóng
            # Giả sử ta đang tấn công khung thành bên phải (X = 1.0)
            target_sign = 1.0 if ball[0] > 0 else -1.0 
            
            near_post = [target_sign * 0.9, ball[1] * 0.1] # Gần cột góc hơn
            far_post = [target_sign * 0.9, -ball[1] * 0.1] # Góc đối diện
            penalty_spot = [target_sign * 0.8, 0.0]        # Chấm 11m
            
            goals = [near_post, far_post, penalty_spot]
            
            # 2. Thêm quả bóng (ở cột góc) vào danh sách CHƯỚNG NGẠI VẬT (Lực đẩy)
            # Điều này ép các cầu thủ phải tránh xa khu vực đá phạt
            obstacles = list(right_team) + [ball] 
            
        else:
            # Nếu là bóng sống bình thường, bóng vẫn là mục tiêu (Lực hút)
            goals = [ball]
            obstacles = right_team
            
        # --- KẾT THÚC LOGIC ---
            
        return left_team, goals, obstacles

    def _process_single_obs(self, single_obs, energy_val):
        ray_info = np.zeros(40) # Giả lập Ray-cast
        energy_field_info = np.array([energy_val]*5)
        return np.concatenate([ray_info, energy_field_info])

    def _build_global_state(self, base_obs):
        return np.concatenate([base_obs['left_team'].flatten(), base_obs['right_team'].flatten(), base_obs['ball'][:2]])

    def _get_corner_kicker_id(self, raw_obs):
        """Xác định ai là người đá phạt và có đang phạt góc hay không."""
        # Kiểm tra an toàn: Nếu chưa có obs nào (chưa reset) thì bỏ qua
        if raw_obs is None:
            return None
            
        base_obs = raw_obs[0]
        ball = np.array(base_obs['ball'][:2])
        
        # Kiểm tra xem bóng có nằm ở 4 góc sân không (Xấp xỉ X=1.0, Y=0.42)
        is_corner_kick = (abs(abs(ball[0]) - 1.0) < 0.05) and (abs(abs(ball[1]) - 0.42) < 0.05)
        
        if is_corner_kick:
            left_team = np.array(base_obs['left_team'])
            distances = np.sum((left_team - ball)**2, axis=1)
            return np.argmin(distances) # Trả về ID người gần bóng nhất ở góc
            
        return None # Nếu là bóng sống bình thường thì trả về None

    def _map_global_actions(self, agent_actions, kicker_id = None):
        """
        Bộ chuyển đổi cô lập hành động cho GAgent.
        GAgent xuất ra giá trị [0 -> 9]:
        - 0 đến 7: Map sang 1 đến 8 (8 hướng di chuyển của gfootball)
        - 8 và 9: Map sang 0 (Hành động Idle/Dừng lại của gfootball)
        """
        mapped_actions = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            if i == kicker_id:
                mapped_actions[i] = 9
            else:
                act = agent_actions[i]
                if act < 8 & act > 0:
                    # Các hành động di chuyển (1: Left, 2: TopLeft, ..., 8: BottomLeft)
                    mapped_actions[i] = act + 1 
                else:
                    # act == 8 hoặc act == 9 đều là STAY (Idle)
                    mapped_actions[i] = 0 
                
        return mapped_actions
    
    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields = self.energy_definer.calculate_field_for_agents(left_team, goals, obstacles)
        
        state = self._build_global_state(raw_obs[0])
        obses = np.array([self._process_single_obs(raw_obs[i], energy_fields[i]) for i in range(self.num_agents)])
        return state, obses

    def step(self, actions):
        kicker_id = self._get_corner_kicker_id(self.last_raw_obs)
        
        safe_actions = self._map_global_actions(actions, kicker_id)
        
        raw_obs, rewards, done, info = self.env.step(safe_actions)
        
        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields = self.energy_definer.calculate_field_for_agents(left_team, goals, obstacles)
        
        # Kết hợp phần thưởng Energy
        shaped_rewards = np.array(rewards) + energy_fields
        
        state = self._build_global_state(raw_obs[0])
        obses = np.array([self._process_single_obs(raw_obs[i], energy_fields[i]) for i in range(self.num_agents)])
        
        return state, obses, shaped_rewards, done