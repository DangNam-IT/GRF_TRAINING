from random import random
import random

import gym
import numpy as np
from utils.energy_field import EnergyFieldDefiner
from utils.field_visualizer import FieldVisualizer  # Real-time debug window

class GFootballGlobalWrapper(gym.Wrapper):
    def __init__(self, env, num_agents=11):
        """
        Args:
            visualize: Bật cửa sổ hiển thị tia raycast + trường năng lượng.
            viz_update_every: Cập nhật visualizer mỗi N step (tăng lên nếu training chậm).
        """
        super().__init__(env)
        self.num_agents = num_agents
        self.action_space = gym.spaces.MultiDiscrete([10] * num_agents)
        self.energy_definer = EnergyFieldDefiner()
        self.state_dim = 46
        self.obs_dim = 33   # 8 rays × 4 object types + 1 energy value
        self.last_raw_obs = None
        self.last_energy_fields = None

        self._kicker_id: int | None = None
        self._has_kicked: bool = False

    def _extract_positions(self, raw_obs):
        """
        Tạo goals và obstacles cho Energy Field.
        Hoàn toàn tách biệt 2 trường hợp để tránh lẫn lộn mục tiêu.
        """
        base_obs = raw_obs[0]
        left_team = np.array(base_obs['left_team'])
        right_team = np.array(base_obs['right_team'])
        ball = np.array(base_obs['ball'][:2])

        # Kiểm tra phạt góc: bóng nằm xấp xỉ 4 góc sân (X≈±1.0, Y≈±0.42)
        is_corner_kick = (abs(abs(ball[0]) - 1.0) < 0.05) and (abs(abs(ball[1]) - 0.42) < 0.05)

        if is_corner_kick:
            # ── PHẠT GÓC: kéo cầu thủ vào vùng cấm, đẩy ra khỏi bóng ────────
            target_sign = 1.0 if ball[0] > 0 else -1.0
            near_post    = np.array([target_sign * 0.9,  ball[1] * 0.1])
            far_post     = np.array([target_sign * 0.9, -ball[1] * 0.1])
            penalty_spot = np.array([target_sign * 0.8,  0.0])

            # sigma lớn hơn → gradient rộng hơn → dẫn đường từ xa hiệu quả hơn
            goals = [
                {'position': near_post,    'sigma': 0.35, 'scale': -1.5},   # Cột gần
                {'position': far_post,     'sigma': 0.35, 'scale': -1.0},   # Cột xa
                {'position': penalty_spot, 'sigma': 0.40, 'scale': -1.2},   # Chấm 11m
            ]

            # sigma đủ rộng để tạo vùng đẩy có tác dụng (~0.18 đơn vị)
            obstacles = [{'position': pos, 'sigma': 0.18, 'scale': 1.2} for pos in right_team]
            obstacles.append({'position': ball, 'sigma': 0.20, 'scale': 1.8})  # Đẩy ra khỏi bóng ở góc

        else:
            # ── BÓNG SỐNG: chỉ bóng là mục tiêu, đối thủ là chướng ngại ────
            goals = [
                {'position': ball, 'sigma': 0.40, 'scale': -1.0},   # Kéo về phía bóng
            ]
            # Đối thủ là chướng ngại vật, đồng đội KHÔNG phải obstacle khi bóng sống
            obstacles = [{'position': pos, 'sigma': 0.18, 'scale': 1.0} for pos in right_team]

        return left_team, goals, obstacles

    def _raycast_from_agent(self, agent_pos, left_team, right_team, ball, goals, max_distance=2.0):
        """
        Ray-cast từ agent theo 8 hướng, detect closest objects.
        Returns: array shape (32,) = 8 rays × 4 object types, normalized [0, 1]
        """
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        detection_radius = 0.15
        ray_distances = []

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])

            distances = {
                'ball': max_distance,
                'opponent': max_distance,
                'teammate': max_distance,
                'goal': max_distance
            }

            # --- BALL ---
            ball_vec = ball - agent_pos
            dist = np.linalg.norm(ball_vec)
            if dist > 0:
                projection = np.dot(ball_vec, direction)
                if projection > 0:
                    perp_dist = np.sqrt(max(0, dist**2 - projection**2))
                    if perp_dist < detection_radius:
                        distances['ball'] = dist

            # --- OPPONENTS ---
            min_opponent_dist = max_distance
            for opp in right_team:
                opp_vec = opp - agent_pos
                dist = np.linalg.norm(opp_vec)
                if dist > 0:
                    projection = np.dot(opp_vec, direction)
                    if projection > 0:
                        perp_dist = np.sqrt(max(0, dist**2 - projection**2))
                        if perp_dist < detection_radius:
                            min_opponent_dist = min(min_opponent_dist, dist)
            distances['opponent'] = min_opponent_dist

            # --- TEAMMATES ---
            min_teammate_dist = max_distance
            for teammate in left_team:
                if not np.allclose(teammate, agent_pos):
                    teammate_vec = teammate - agent_pos
                    dist = np.linalg.norm(teammate_vec)
                    if dist > 0:
                        projection = np.dot(teammate_vec, direction)
                        if projection > 0:
                            perp_dist = np.sqrt(max(0, dist**2 - projection**2))
                            if perp_dist < detection_radius:
                                min_teammate_dist = min(min_teammate_dist, dist)
            distances['teammate'] = min_teammate_dist

            # --- GOALS ---
            min_goal_dist = max_distance
            for goal in goals:
                goal_pos = np.array(goal['position']) if isinstance(goal, dict) else np.array(goal)
                goal_vec = goal_pos - agent_pos
                dist = np.linalg.norm(goal_vec)
                if dist > 0:
                    projection = np.dot(goal_vec, direction)
                    if projection > 0:
                        perp_dist = np.sqrt(max(0, dist**2 - projection**2))
                        if perp_dist < detection_radius:
                            min_goal_dist = min(min_goal_dist, dist)
            distances['goal'] = min_goal_dist

            ray_distances.extend([distances['ball'], distances['opponent'],
                                distances['teammate'], distances['goal']])

        ray_distances = np.array(ray_distances, dtype=np.float32)
        ray_distances = np.minimum(ray_distances / max_distance, 1.0)
        return ray_distances

    def _process_single_obs(self, agent_pos, left_team, right_team, ball, goals, energy_val):
        ray_info = self._raycast_from_agent(agent_pos, left_team, right_team, ball, goals, max_distance=2.0)
        energy_field_info = np.array([energy_val], dtype=np.float32)
        return np.concatenate([ray_info, energy_field_info]), ray_info  # trả thêm ray_info thô

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

    def _map_global_actions(self, agent_actions, raw_obs = None):
        """
        Bộ chuyển đổi cô lập hành động cho GAgent.
        GAgent xuất ra giá trị [0 -> 9]:
        - 0 đến 7: Map sang 1 đến 8 (8 hướng di chuyển của gfootball)
        - 8 và 9: Map sang 0 (Hành động Idle/Dừng lại của gfootball)
        """

        kicker_id = self._kicker_id
        mapped_actions = np.zeros(self.num_agents, dtype=int)
        # for i in range(self.num_agents):
        #     if i == kicker_id and not self._has_kicked:
        #         mapped_actions[i] = 9
        #         # random.choice([9, 10])
        #     else:
        #         act = agent_actions[i]
        #         if act < 8:
        #             # Các hành động di chuyển (1: Left, 2: TopLeft, ..., 8: BottomLeft)
        #             mapped_actions[i] = act + 1
        #         else:
        #             mapped_actions[i] = random.choice([0, 13, 14, 15, 17, 18]) 
        if kicker_id is not None and self._has_kicked and raw_obs is not None:
            base_obs = raw_obs[0]
            ball = np.array(base_obs['ball'][:2])
            ball_speed = np.linalg.norm(base_obs['ball'][3:5]) if len(base_obs['ball']) >= 5 else 0.0
            kicker_pos = np.array(base_obs['left_team'][kicker_id])
            ball_dist_from_kicker = np.linalg.norm(ball - kicker_pos)
            # Bóng đã chuyển động hoặc đã rời xa kicker → reset state machine
            if ball_speed > 0.1 or ball_dist_from_kicker > 0.05:
                self._kicker_id = None
                self._has_kicked = False
                kicker_id = None

        for i in range(self.num_agents):
            if i == kicker_id:
                if not self._has_kicked:
                    # PHASE 2: Trigger duy nhất 1 frame High Pass
                    mapped_actions[i] = random.choice([9, 10])   # High Pass
                    self._has_kicked = True
                else:
                    # PHASE 3: Idle cho đến khi bóng rời góc
                    mapped_actions[i] = 0
            else:
                act = agent_actions[i]
                if 0 <= act < 8:
                    # Các hành động di chuyển (1: Top, 2: TopLeft, ..., 8: TopRight)
                    mapped_actions[i] = act + 1
                else:
                    mapped_actions[i] = random.choice([0, 14])
                
        return mapped_actions
    
    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields = self.energy_definer.calculate_field_for_agents(left_team, goals, obstacles)

        self.last_energy_fields = energy_fields

        base_obs = raw_obs[0]
        right_team = np.array(base_obs['right_team'])
        ball = np.array(base_obs['ball'][:2])

        state = self._build_global_state(base_obs)

        # Gọi _process_single_obs — bây giờ trả về (obs_vec, ray_info)
        results = [self._process_single_obs(left_team[i], left_team, right_team, ball, goals, energy_fields[i])
                   for i in range(self.num_agents)]
        obses    = np.array([r[0] for r in results])
        ray_data = np.array([r[1] for r in results])  # shape (11, 32)

        return state, obses

    def step(self, actions):
        kicker_id = self._get_corner_kicker_id(self.last_raw_obs)

        if kicker_id is not None and not self._has_kicked:
            self._kicker_id = kicker_id

        safe_actions = self._map_global_actions(actions, raw_obs=self.last_raw_obs)

        raw_obs, rewards, done, info = self.env.step(safe_actions)

        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields = self.energy_definer.calculate_field_for_agents(left_team, goals, obstacles)

        # ── Cân bằng quy mô phần thưởng ─────────────────────────────────────
        # GRF 'scoring': khi thủng/ghi bàn, MỖI agent nhận ±1.0
        # → np.sum() qua 11 agent = ±11, áp đảo energy (~±0.75) và ball_owned
        #
        # Giải pháp: chia rewards_env cho num_agents
        #   → mỗi event ghi bàn = ±1.0 TỔNG CỘNG (±1/11 per agent)
        #   → quy mô tương đương với energy signal và ball_owned
        #
        # Lưu ý: agent vẫn phân biệt được ghi bàn vs thủng lưới,
        #         chỉ là magnitude được normalize lại.
        ENV_SCALE       = 1.0 / self.num_agents   # ±1/11 ≈ ±0.091 per agent per goal
        ENERGY_SCALE    = 0.05
        BALL_OWNED_REWARD = 0.1   # chia đều cho 11 agent ≈ 0.009/agent

        rewards_env        = np.array(rewards) * ENV_SCALE                                # shape (11,), normalized
        rewards_energy     = (self.last_energy_fields - energy_fields) * ENERGY_SCALE    # shape (11,), đã scale
        rewards_ball_owned = np.zeros(self.num_agents)                                    # shape (11,)

        shaped_rewards = rewards_env + rewards_energy

        # --- LOGIC PHẦN THƯỞNG QUYỀN SỞ HỮU BÓNG ---
        current_ball_owned_team = raw_obs[0]['ball_owned_team']
        last_ball_owned_team = self.last_raw_obs[0]['ball_owned_team'] if self.last_raw_obs is not None else -1

        # Phạt khi ĐỐI THỦ VỪA GIÀNH bóng (ball_owned_team: 0/−1 → 1)
        if current_ball_owned_team == 1 and last_ball_owned_team != 1:
            ball_possess = np.full(self.num_agents, -BALL_OWNED_REWARD / self.num_agents)
            rewards_ball_owned += ball_possess
            shaped_rewards     += ball_possess
        # Thưởng khi ĐỘI TA VỪA GIÀNH LẠI bóng (ball_owned_team: 1/−1 → 0)
        elif current_ball_owned_team == 0 and last_ball_owned_team != 0:
            ball_possess = np.full(self.num_agents,  BALL_OWNED_REWARD / self.num_agents)
            rewards_ball_owned += ball_possess
            shaped_rewards     += ball_possess

        rewards_view = {
            'rewards_env':        rewards_env,
            'rewards_energy':     rewards_energy,
            'rewards_ball_owned': rewards_ball_owned
        }

        # Lưu đệm trạng thái hiện tại làm mốc cho bước kế tiếp k+1
        self.last_raw_obs = raw_obs
        self.last_energy_fields = energy_fields

        base_obs = raw_obs[0]
        right_team = np.array(base_obs['right_team'])
        ball = np.array(base_obs['ball'][:2])

        state = self._build_global_state(base_obs)

        # Gọi _process_single_obs — bây giờ trả về (obs_vec, ray_info)
        results = [self._process_single_obs(left_team[i], left_team, right_team, ball, goals, energy_fields[i])
                   for i in range(self.num_agents)]
        obses    = np.array([r[0] for r in results])
        ray_data = np.array([r[1] for r in results])  # shape (11, 32)


        return state, obses, shaped_rewards, rewards_view, done