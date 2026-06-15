# Tệp: envs/wrapper_e_coma.py (Dành cho E-COMA)
import gym
import numpy as np

from utils.energy_field import EnergyFieldDefiner, FieldItem
from numpy.typing import NDArray
from typing import Tuple, List, Dict, Any, Optional
RawObs = List[Dict[str, Any]]
class ECOMA_Wrapper(gym.Wrapper):
    """
    Wrapper cho E-COMA (Không Phân Cấp).
    Sử dụng 1 mạng duy nhất với n_actions = 13 (8 di chuyển + 4 chuyền/sút + 1 idle).
    Vẫn sử dụng Energy Field để tính reward chạy chỗ.
    """
    def __init__(self, env, num_agents=11):
        super().__init__(env)
        self.num_agents = num_agents
        # Không gian hành động lớn hơn (bao gồm cả sút và chuyền)
        self.action_space = gym.spaces.MultiDiscrete([14] * num_agents) 
        self.energy_definer = EnergyFieldDefiner()
        self.N_RAYS = 16
        self.state_dim = 46
        self.obs_dim = 70 
        
        self.last_energy_fields = None

    def _extract_positions(
        self,
        raw_obs: RawObs,
    ) -> Tuple[
        NDArray[np.float32],   # left_team   (11, 2)
        NDArray[np.float32],   # right_team  (11, 2)
        NDArray[np.float32],   # ball        (2,)
        list[FieldItem],       # goals
        list[FieldItem],       # obstacles
    ]:
        """
        Tạo goals/obstacles cho Energy Field — mirror wrapper_global.
        """
        base_obs   = raw_obs[0]
        left_team:  NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])

        obstacles = []

        target_sign:  float = 1.0 if ball[0] > 0 else -1.0
        near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  ball[1] * 0.1])
        far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -ball[1] * 0.1])
        penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])

        # THAY ĐỔI: Thu hẹp Sigma, Đào sâu Scale để tạo khoảng trống rõ rệt
        goals = [
            {"position": near_post,    "sigma": 0.3, "scale": -2.0},
            {"position": far_post,     "sigma": 0.3, "scale": -1.5},
            {"position": penalty_spot, "sigma": 0.6, "scale": -2.5},
        ]
        # THAY ĐỔI: Thu hẹp Sigma của hậu vệ để Agent có kẽ hở luồn lách
        obstacles = []
        for pos in right_team:
            obstacles.append({"position": pos, "sigma": 0.05, "scale": 0.1})
        obstacles.append({"position": ball, "sigma": 0.25, "scale": 0.1})
        return left_team, right_team, ball, goals, obstacles

    def _raycast_from_agent(
        self,
        agent_pos:    NDArray[np.float32],
        left_team:    NDArray[np.float32],
        right_team:   NDArray[np.float32],
        targets:      list[NDArray[np.float32]],   # [near_post, far_post, penalty_spot]
        max_distance: float = 0.3,
    ) -> NDArray[np.float32]:
        """
        16 hướng × 3 kênh → (48,) normalized [0, 1].
        Mirror wrapper_global._raycast_from_agent.

        Kênh:
          [0] opponent   — khoảng cách tia chạm đối thủ gần nhất
          [1] teammate   — khoảng cách tia chạm đồng đội gần nhất
          [2] target     — khoảng cách tia chạm điểm mục tiêu gần nhất
        """
        angles:           NDArray[np.float32] = np.linspace(0, 2 * np.pi, self.N_RAYS, endpoint=False)
        detection_radius: float               = 0.02
        ray_distances:    list[float]         = []

        for angle in angles:
            direction: NDArray[np.float32] = np.array([np.cos(angle), np.sin(angle)])

            # ── Kênh 0: OPPONENTS ────────────────────────────────────────────
            min_opp: float = max_distance
            for opp in right_team:
                ov:   NDArray[np.float32] = opp - agent_pos
                dist: float = float(np.linalg.norm(ov))
                if dist > 0:
                    proj: float = float(np.dot(ov, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                        min_opp = min(min_opp, dist)

            # ── Kênh 1: TEAMMATES ────────────────────────────────────────────
            min_tm: float = max_distance
            for tm in left_team:
                if not np.allclose(tm, agent_pos):
                    tv:   NDArray[np.float32] = tm - agent_pos
                    dist = float(np.linalg.norm(tv))
                    if dist > 0:
                        proj = float(np.dot(tv, direction))
                        if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                            min_tm = min(min_tm, dist)

            # ── Kênh 2: TARGET ZONES (near_post / far_post / penalty_spot) ──
            min_target: float = max_distance
            for tgt in targets:
                tv2:  NDArray[np.float32] = tgt - agent_pos
                dist = float(np.linalg.norm(tv2))
                if dist > 0:
                    proj = float(np.dot(tv2, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                        min_target = min(min_target, dist)

            ray_distances.extend([min_opp, min_tm, min_target])

        arr: NDArray[np.float32] = np.array(ray_distances, dtype=np.float32)
        return np.minimum(arr / max_distance, 1.0)  # (48,) = 16 × 3

    # =========================================================================
    # PHẦN 3: XÂY DỰNG QUAN SÁT
    # =========================================================================
    def _process_single_obs(
        self,
        agent_idx:  int,
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      List[FieldItem],
        energy_val: float,
        raw_obs_i:  Dict[str, Any],
        base_obs:   Dict[str, Any],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Obs layout (70 chiều):
          rays        (48) = 16 tia × 3 kênh [opponent, teammate, target_zone]
          energy       (1) = energy field value của agent
          ball_owned   (1) = 1 nếu agent đang cầm bóng
          sticky      (10) = trạng thái 10 sticky action
          role        (10) = one-hot vị trí sở trường
        """
        # Tách danh sách điểm mục tiêu từ goals (near_post, far_post, penalty_spot)
        target_positions: List[NDArray[np.float32]] = [
            np.array(g["position"] if isinstance(g, dict) else g, dtype=np.float32)
            for g in goals
        ]
        # Fallback: nếu không có goals (ngoài corner), dùng vị trí khung thành đối phương
        if not target_positions:
            target_sign = 1.0 if ball[0] > 0 else -1.0
            target_positions = [np.array([target_sign * 0.9, 0.0], dtype=np.float32)]

        ray_info: NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, target_positions, max_distance=0.3
        )

        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)

        #  Cờ sở hữu bóng (Boolean flag)
        ball_owned_team   = base_obs.get("ball_owned_team", -1)
        ball_owned_player = base_obs.get("ball_owned_player", -1)
        is_ball_owned     = 1.0 if (ball_owned_team == 0 and ball_owned_player == agent_idx) else 0.0
        is_ball_owned_arr: NDArray[np.float32] = np.array([is_ball_owned], dtype=np.float32)

        # Trạng thái nút bấm (Sticky actions)
        sticky_actions_raw = raw_obs_i.get("sticky_actions", [0]*10)
        sticky_actions: NDArray[np.float32] = np.array(sticky_actions_raw, dtype=np.float32)

        # Vị trí sở trường (Left team roles)
        roles_raw = base_obs.get("left_team_roles", [0]*11)
        role_idx  = roles_raw[agent_idx] if agent_idx < len(roles_raw) else 0
        role_onehot: NDArray[np.float32] = np.zeros(10, dtype=np.float32)
        if 0 <= role_idx < 10:
            role_onehot[role_idx] = 1.0

        obs_vec: NDArray[np.float32] = np.concatenate([
            ray_info,           # 48  (16 tia × 3 kênh)
            energy_field_info,  #  1
            is_ball_owned_arr,  #  1
            sticky_actions,     # 10
            role_onehot,        # 10
        ])                      # Tổng: 70

        return obs_vec, ray_info
    
    def _build_global_state(self, base_obs: Dict[str, Any]) -> NDArray[np.float32]:
        """Trả về state toàn cục (46,) = left(22) + right(22) + ball(2)."""
        return np.concatenate([
            base_obs["left_team"].flatten(),
            base_obs["right_team"].flatten(),
            base_obs["ball"][:2],
        ])
    
    def _map_global_actions(
        self,
        agent_actions: NDArray[np.int64],
        raw_obs:       Optional[RawObs] = None,
    ) -> NDArray[np.int64]:
        
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            act = int(agent_actions[i])
            if 0 <  act <= 12:
                mapped_actions[i] = act   # GRF action 1-8 (8 hướng di chuyển)
            elif act == 0:
                mapped_actions[i] = 0
            else:
                mapped_actions[i] = 14     # GRF 14: release direction → đứng im
        return mapped_actions
    
    def _get_all_obses_and_state(
        self,
        raw_obs: RawObs,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Trả về state toàn cức (46,) + obses (11, 54) + energy_fields (11,)
        Returns:
            state  (46,)    — global state vector
            obses  (11, 54) — per-agent GAgent observations
            energy_fields (11,) — energy field values (cũng cập nhật self.last_energy_fields)
        """
        left_team, right_team, ball, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )
        base_obs  = raw_obs[0]
        state:      NDArray[np.float32] = self._build_global_state(base_obs)

        results = [
            self._process_single_obs(
                i, left_team[i], left_team, right_team, ball, goals, energy_fields[i],
                raw_obs[i], base_obs
            )
            for i in range(self.num_agents)
        ]
        obses: NDArray[np.float32] = np.array([r[0] for r in results])
        return state, obses, energy_fields
    
    def reset(self, **kwargs: Any) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        raw_obs: RawObs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        self._kicker_id   = None
        self._aim_frames  = 0
        self._kick_frames = 0
        self._has_kicked  = False

        state, obses, energy_fields = self._get_all_obses_and_state(raw_obs)
        self.last_energy_fields = energy_fields

        return state, obses
    
    def step(self, actions):
        ENV_SCALE:         float = 1.0 / self.num_agents
        ENERGY_SCALE:    float = 0.05
        PASSING_REWARD:    float =  0.5   # Kicker tạt bóng thành công
        ASSIST_REWARD:     float =  1.0   # Kiến tạo dẫn đến bàn thắng
        BALL_APPROACH_R:   float =  0.3   # Tiến về điểm rơi sau khi bóng vào vùng cấm
        # Role-based (§1.4): thưởng theo vùng chiến thuật khi ghi bàn
        ROLE_NEAR_POST:    float =  1.0
        ROLE_FAR_POST:     float =  1.0
        ROLE_PENALTY_SPOT: float =  1.5   
        # ── Cơ chế ưu tiên sút trong vòng cấm ────────────────────────────────
        BOX_X_THRESHOLD:   float =  0.83  # GRF pitch x ∈ [-1, 1]
        BOX_Y_THRESHOLD:   float =  0.20  # GRF pitch |y| ∈ [0, 0.42]
        SHOT_IN_BOX_R:     float =  2.0   # Thưởng cực mạnh khi sút trong vòng cấm
        PASS_IN_BOX_P:     float = -1.5   # Phạt nặng khi chuyền trong vòng cấm (triệt tiêu chuyền quẩn)
        # ── Cơ chế đánh giá đường chuyền của Kicker ──────────────────────────
        DANGER_ZONE_X:       float =  0.75
        DANGER_ZONE_Y:       float =  0.35
        BACKPASS_PENALTY:    float = -0.8
        OUTSIDE_BOX_PENALTY: float = -0.3

        # ── Đọc trạng thái trước bước ─────────────────────────────────────────
        last_obs  = self.last_raw_obs[0] if self.last_raw_obs is not None else None
        last_ball_owned:   int = last_obs["ball_owned_team"]           if last_obs else -1
        last_ball_owned_player: int = last_obs.get("ball_owned_player", -1) if last_obs else -1
        last_score: list[int]  = list(last_obs["score"])               if last_obs else [0, 0]

        # ── Corner-kick state machine (mirror _map_global_actions) ───────────
        kicker_id: Optional[int] = 1

        safe_action = self._map_global_actions(actions)
        raw_obs, rewards_list, done, info = self.env.step(safe_action)
        current_obs = raw_obs[0]

        self.last_raw_obs = raw_obs
        state, obses, energy_fields = self._get_all_obses_and_state(raw_obs)

        rewards_env = np.array(rewards_list, dtype=np.float32) * ENV_SCALE
        rewards_energy = (-energy_fields) * ENERGY_SCALE
        shaped_rewards: NDArray[np.float32] = rewards_env.copy() + rewards_energy

        # ── Đọc trạng thái sau bước ──────────────────────────────────────────
        current_ball_owned:        int       = current_obs["ball_owned_team"]
        current_ball_owned_player: int       = current_obs.get("ball_owned_player", -1)
        current_score: list[int]             = list(current_obs["score"])
        left_team:     NDArray[np.float32]   = np.array(current_obs["left_team"])

        # ── R_passing: Đánh giá chất lượng đường chuyền của Kicker ───────────
        rewards_passing: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)

        # 1. Thưởng Ý Định (Intent Reward): Ép Kicker chọn High Pass
        # TACTIC_MAP = [9(long), 10(high), 11(short), 12(shot)]
        kicker_just_got_ball: bool = (
            kicker_id is not None
            and last_ball_owned_player != kicker_id   # bước trước chưa có bóng
            and last_ball_owned_player == -1          # bóng vừa rời tay người chuyền
        )
        if kicker_just_got_ball:
            kicker_tactic = int(safe_action[kicker_id])
            if kicker_tactic == 10:   # high_pass (index 1 = GRF 10)
                rewards_passing[kicker_id] += 1.0
                shaped_rewards[kicker_id] += 1.0
            else:                    # short_pass / long_pass / shot
                rewards_passing[kicker_id] -= 1.0
                shaped_rewards[kicker_id] -= 1.0

        # 2. Thưởng Kết Quả (Outcome Reward): Bóng đến vùng nguy hiểm
        if (
            kicker_id is not None
            and current_ball_owned == 0       # sau đội ta có bóng
            and current_ball_owned_player != kicker_id  # người khác nhận
            and last_ball_owned_player != current_ball_owned_player # vừa mới nhận bóng ở step này
        ):
            receiver_idx = current_ball_owned_player
            receiver_x = float(left_team[receiver_idx][0])
            receiver_y = float(left_team[receiver_idx][1])
            
            # Xác định hướng tấn công dựa trên vị trí nhận bóng hoặc bóng đầu
            attack_sign: float = 1.0 if (last_obs and last_obs["ball"][0] > 0) else -1.0
            rx_eval = attack_sign * receiver_x
            
            if rx_eval > DANGER_ZONE_X and abs(receiver_y) < DANGER_ZONE_Y:
                # [Tốt] Chuyền vào vùng nguy hiểm
                rewards_passing[kicker_id] += PASSING_REWARD
                shaped_rewards[kicker_id] += PASSING_REWARD
            elif rx_eval < 0.0:
                # [Tồi] Chuyền về sân nhà
                rewards_passing[kicker_id] += BACKPASS_PENALTY
                shaped_rewards[kicker_id] += BACKPASS_PENALTY
            else:
                # [Kém] Chuyền ra ngoài vòng cấm nhưng vẫn ở phần sân đối phương
                rewards_passing[kicker_id] += OUTSIDE_BOX_PENALTY
                shaped_rewards[kicker_id] += OUTSIDE_BOX_PENALTY

        # ── R_in_box: Soft reward ưu tiên sút trong vòng cấm 
        last_left_team: NDArray[np.float32] = np.array(last_obs["left_team"])
        rewards_in_box: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if last_obs is not None:
            last_ball_owned_check: int = last_obs.get("ball_owned_team", -1)
            last_ball_pos: NDArray[np.float32] = np.array(last_obs["ball"][:2])
            attack_sign: float = 1.0 if last_ball_pos[0] > 0 else -1.0
            if last_ball_owned_check == 0 and last_ball_owned_player != -1:
                idx = last_ball_owned_player
                player_x: float = float(last_left_team[idx][0])
                player_y: float = float(last_left_team[idx][1])
                in_box: bool = (
                    attack_sign * player_x > BOX_X_THRESHOLD
                    and abs(player_y) < BOX_Y_THRESHOLD
                )
                if in_box and idx != kicker_id:
                    tactic_chosen: int = int(safe_action[idx])
                    if tactic_chosen == 12:   # Shot
                        rewards_in_box[idx]  = SHOT_IN_BOX_R
                        shaped_rewards[idx] += SHOT_IN_BOX_R
                    else:
                        rewards_in_box[idx]  = PASS_IN_BOX_P
                        shaped_rewards[idx] += PASS_IN_BOX_P

        # ── R_assist: Kiến tạo dẫn đến bàn thắng (+1.0) ─────────────────────
        rewards_assist: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        scored_this_step: bool = current_score[0] > last_score[0]
        if scored_this_step and kicker_id is not None:
            rewards_assist[kicker_id] = ASSIST_REWARD
            shaped_rewards[kicker_id] += ASSIST_REWARD

        # ── R_role_based: Dense positioning reward + Sparse goal bonus
        rewards_role: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if last_obs is not None:
            ball_init: NDArray[np.float32] = np.array(last_obs["ball"][:2])
            sign: float = 1.0 if ball_init[0] > 0 else -1.0

            # Vùng chiến thuật (theo phân tích §1.4 Reward_logic.md)
            near_post_zone:    NDArray[np.float32] = np.array([sign * 0.9,  ball_init[1] * 0.1])
            far_post_zone:     NDArray[np.float32] = np.array([sign * 0.9, -ball_init[1] * 0.1])
            penalty_spot_zone: NDArray[np.float32] = np.array([sign * 0.8,  0.0])

            for i in range(self.num_agents):
                pos: NDArray[np.float32] = left_team[i]
                if scored_this_step:
                    # Sparse goal bonus (giữ nguyên)
                    if float(np.linalg.norm(pos - near_post_zone)) < 0.15:
                        rewards_role[i] = ROLE_NEAR_POST
                        shaped_rewards[i] += ROLE_NEAR_POST
                    elif float(np.linalg.norm(pos - far_post_zone)) < 0.15:
                        rewards_role[i] = ROLE_FAR_POST
                        shaped_rewards[i] += ROLE_FAR_POST
                    elif float(np.linalg.norm(pos - penalty_spot_zone)) < 0.20:
                        rewards_role[i] = ROLE_PENALTY_SPOT
                        shaped_rewards[i] += ROLE_PENALTY_SPOT
                elif current_ball_owned == -1 and i != kicker_id:
                    DENSE_POS_R: float = 0.05
                    if float(np.linalg.norm(pos - near_post_zone)) < 0.15:
                        rewards_role[i] += DENSE_POS_R
                        shaped_rewards[i] += DENSE_POS_R
                    elif float(np.linalg.norm(pos - far_post_zone)) < 0.15:
                        rewards_role[i] += DENSE_POS_R
                        shaped_rewards[i] += DENSE_POS_R
                    elif float(np.linalg.norm(pos - penalty_spot_zone)) < 0.20:
                        rewards_role[i] += DENSE_POS_R * 1.5  # Penalty spot khó hơn → thưởng cao hơn
                        shaped_rewards[i] += DENSE_POS_R * 1.5

        # ── R_ball_approach: Thưởng Sparse khi chạm cắt bóng thành công ─────
        rewards_approach: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if current_ball_owned == 0 and current_ball_owned_player != -1:
            if last_ball_owned != 0 or last_ball_owned_player != current_ball_owned_player:
                # Chỉ thưởng 1 lần khi giành được quyền kiểm soát bóng
                rewards_approach[current_ball_owned_player] = BALL_APPROACH_R
                shaped_rewards[current_ball_owned_player] += BALL_APPROACH_R

       

        # ── Tổng hợp các phần thưởng thành phần (sum trên tất cả agent) ──────
        reward_info: dict = {
            'R_env':        float(np.sum(rewards_env)),
            'R_energy':     float(np.sum(rewards_energy)),
            'R_passing':    float(np.sum(rewards_passing)),
            'R_in_box':     float(np.sum(rewards_in_box)),
            'R_assist':     float(np.sum(rewards_assist)),
            'R_role':       float(np.sum(rewards_role)),
            'R_approach':   float(np.sum(rewards_approach)),
        }

        return state, obses, shaped_rewards, done, reward_info