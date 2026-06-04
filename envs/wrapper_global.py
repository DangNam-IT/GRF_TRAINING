from __future__ import annotations

from operator import is_
import random
from typing import Any, Optional, Tuple, Dict, List

import gym
import numpy as np
from numpy.typing import NDArray

from utils.energy_field import EnergyFieldDefiner, FieldItem

# Kiểu raw observation của GRF: list of dict (1 dict per agent, nhưng shared state)
RawObs = List[Dict[str, Any]]


class GFootballGlobalWrapper(gym.Wrapper):
    """
    Wrapper Phase 1 (Global Spatial Hierarchy) của HES-COMA.

    Xử lý:
    - Raycast 8 hướng × 4 object types.
    - Làm giàu quan sát (Obs Enrichment): ball_owned_player, sticky_actions, left_team_roles.
    - Lực hút dự đoán (Anticipatory Attractors) và Lực đẩy định hướng (Directional Repulsors).
    - Phân cấp (Hierarchical): Phase 1 (GAgent) lo chạy chỗ không gian và xuất ra lệnh STOP. 
      Khi lệnh STOP được xuất ra, Phase 2 (LAgent) mới được kích hoạt để Sút/Chuyền.
    - Reward shaping tối ưu chuẩn hóa [-1, 1].
    """

    def __init__(self, env: gym.Env, num_agents:int) -> None:
        super().__init__(env)
        self.num_agents: int = num_agents
        self.action_space = gym.spaces.MultiDiscrete([9] * num_agents)
        self.energy_definer: EnergyFieldDefiner = EnergyFieldDefiner()
        
        self.state_dim: int = 46   # left(22) + right(22) + ball(2)
        # obs_dim = rays(32) + energy(1) + ball_owned(1) + sticky_actions(10) + role(10) = 54
        self.obs_dim: int = 54   

        self.last_raw_obs: Optional[RawObs] = None
        self.last_energy_fields: Optional[NDArray[np.float32]] = None

        self._kicker_id:   Optional[int] = None
        self._aim_frames:  int  = 0       # Số frame đã xoay hướng
        self._kick_frames: int  = 0       # Số frame đã nhấn nút chuyền (gồng lực)
        self._has_kicked:  bool = False
        self._AIM_DURATION: int = 5       # Số frame cần để GRF engine xoay mặt kicker
        self._KICK_DURATION: int = 4      # Số frame gồng lực High Pass (chuyền sâu vào cấm)

    # =========================================================================
    def _extract_positions(
        self,
        raw_obs: RawObs,
    ) -> Tuple[
        NDArray[np.float32],   # left_team  (11, 2)
        List[FieldItem],       # goals
        List[FieldItem],       # obstacles
    ]:
        """
        Tạo goals và obstacles cho Energy Field.
        [YÊu CẦU 1]: Nâng cấp Trường năng lượng Động (Dynamic Energy Fields)

        [MODULE 2 - FIX 0] Bug gốc rễ: ball_direction phải đọc từ key riêng
        'ball_direction' chứ KHÔNG phải từ base_obs['ball'][3:5].
        base_obs['ball'] chỉ có 3 phần tử [x, y, z], không có index 3,4.
        """
        base_obs = raw_obs[0]
        left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        # ball_z: float = float(base_obs["ball"][2])  # chiều cao bóng (z-coordinate)

        # [MODULE 2 - FIX 0]: Dùng key 'ball_direction' riêng (theo observation.md)
        # ball_direction là [vx, vy, vz] — chỉ lấy 2 thành phần x,y mặt đất
        # ball_dir_raw = base_obs.get("ball_direction", [0.0, 0.0, 0.0])
        # ball_direction: NDArray[np.float32] = np.array(ball_dir_raw[:2], dtype=np.float32)

        # right_team_direction_raw = base_obs.get("right_team_direction", [[0.0, 0.0]] * len(right_team))
        # right_team_direction: NDArray[np.float32] = np.array(right_team_direction_raw, dtype=np.float32)

        # [MODULE 2 - FIX 2]: Phát hiện bóng đang bay bổng (ball in flight)
        # Khi bóng bay bổng (z > 0.08m), KHÔNG nhân velocity vì sẽ đẩy
        # attractor ra ngoài sân. Khóa cứng vào Landing Zone cố định.
        # ball_in_flight: bool = ball_z > 0.08

        base_obs = raw_obs[0]
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )

        # # game_mode == 4 là trạng thái đặt bóng chết ở góc sân của GRF
        # is_set_piece_corner: bool = (base_obs.get("game_mode", -1) == 4)
        # is_ball_flying_in_air: bool = self._has_kicked and (ball_z > 0.06)

        # # if is_set_piece_corner or is_ball_flying_in_air:
        # if 3 ==2:
        #     target_sign: float = 1.0 if ball[0] > 0 else -1.0

        #     if ball_in_flight:
        #         # [MODULE 2 - FIX 2]: Bóng đang bay bổng — Landing Zone cố định
        #         # Khóa tuyệt đối, KHÔNG nhân với ball_direction.
        #         near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  0.05])
        #         far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -0.05])
        #         penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])
        #     else:
        #         # Bóng trên mặt đất — dùng Anticipatory Attractors
        #         # [YÊU CẦU 1.1]: Lực hút đoán trước (Anticipatory Attractors)
        #         k_ball: float = 5.0
        #         predicted_ball_pos: NDArray[np.float32] = ball + ball_direction * k_ball
        #         near_post    = np.array([target_sign * 0.9,  predicted_ball_pos[1] * 0.1])
        #         far_post     = np.array([target_sign * 0.9, -predicted_ball_pos[1] * 0.1])
        #         penalty_spot = np.array([target_sign * 0.8,  0.0])

        #     goals = [
        #         {"position": near_post,    "sigma": 0.15, "scale": -0.5},
        #         {"position": far_post,     "sigma": 0.20, "scale": -0.3},
        #         {"position": penalty_spot, "sigma": 0.25, "scale": -0.4},
        #     ]

        #     # [YÊU CẦU 1.2]: Lực đẩy định hướng (Directional Repulsors)
        #     # [MODULE 2 - FIX 3]: sigma=0.06 để giữ kẽ hở phòng ngự (Anti-pressing gaps)
        #     k_opp: float = 3.0
        #     obstacles = []
        #     for i, pos in enumerate(right_team):
        #         opp_dir: NDArray[np.float32] = (
        #             right_team_direction[i] if i < len(right_team_direction)
        #             else np.zeros(2, dtype=np.float32)
        #         )
        #         predicted_opp_pos: NDArray[np.float32] = pos + opp_dir * k_opp
        #         obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 0.3})

        #     # Chỉ thêm obstacle bóng khi bóng trên mặt đất (không push attractor khi bay)
        #     if not ball_in_flight:
        #         predicted_ball_pos_obs: NDArray[np.float32] = ball + ball_direction * k_opp
        #         obstacles.append({"position": predicted_ball_pos_obs, "sigma": 0.25, "scale": -0.5})

        # else:
        #     k_ball = 5.0
        #     predicted_ball_pos = ball + ball_direction * k_ball
        #     goals = [{"position": predicted_ball_pos, "sigma": 0.25, "scale": -0.5}]

        #     k_opp = 3.0
        #     obstacles = []
        #     for i, pos in enumerate(right_team):
        #         opp_dir = (
        #             right_team_direction[i] if i < len(right_team_direction)
        #             else np.zeros(2, dtype=np.float32)
        #         )
        #         predicted_opp_pos = pos + opp_dir * k_opp
        #         obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 0.3})

        if is_corner_kick:
            target_sign:  float = 1.0 if ball[0] > 0 else -1.0
            near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  ball[1] * 0.1])
            far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -ball[1] * 0.1])
            penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])

            # THAY ĐỔI: Thu hẹp Sigma, Đào sâu Scale để tạo khoảng trống rõ rệt
            goals = [
                {"position": near_post,    "sigma": 0.15, "scale": -0.5},
                {"position": far_post,     "sigma": 0.20, "scale": -0.3},
                {"position": penalty_spot, "sigma": 0.25, "scale": -0.4},
            ]
            # THAY ĐỔI: Thu hẹp Sigma của hậu vệ để Agent có kẽ hở luồn lách
            obstacles = [
                {"position": pos, "sigma": 0.06, "scale": 0.2} for pos in right_team
            ]
            obstacles.append({"position": ball, "sigma": 0.25, "scale": 0.5})

        else:
            goals = [{"position": ball, "sigma": 0.25, "scale": -2.5}]
            obstacles = [{"position": pos, "sigma": 0.06, "scale": 0.2} for pos in right_team]

        return left_team, goals, obstacles

    # =========================================================================
    # PHẦN 2: RAY-CAST
    # =========================================================================

    def _raycast_from_agent(
        self,
        agent_pos:    NDArray[np.float32],
        left_team:    NDArray[np.float32],
        right_team:   NDArray[np.float32],
        ball:         NDArray[np.float32],
        goals:        List[FieldItem],
        max_distance: float = 0.5,
    ) -> NDArray[np.float32]:
        """Ray-cast từ agent theo 8 hướng, detect closest objects."""
        angles: NDArray[np.float32] = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        detection_radius: float = 0.025
        ray_distances: List[float] = []

        for angle in angles:
            direction: NDArray[np.float32] = np.array([np.cos(angle), np.sin(angle)])
            dists: Dict[str, float] = {
                "ball":     max_distance,
                "opponent": max_distance,
                "teammate": max_distance,
                "goal":     max_distance,
            }

            # BALL
            ball_vec: NDArray[np.float32] = ball - agent_pos
            dist: float = float(np.linalg.norm(ball_vec))
            if dist > 0:
                proj: float = float(np.dot(ball_vec, direction))
                if proj > 0:
                    perp: float = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                    if perp < detection_radius:
                        dists["ball"] = dist

            # OPPONENTS
            min_opp: float = max_distance
            for opp in right_team:
                ov: NDArray[np.float32] = opp - agent_pos
                dist = float(np.linalg.norm(ov))
                if dist > 0:
                    proj = float(np.dot(ov, direction))
                    if proj > 0:
                        perp = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                        if perp < detection_radius:
                            min_opp = min(min_opp, dist)
            dists["opponent"] = min_opp

            # TEAMMATES
            min_tm: float = max_distance
            for teammate in left_team:
                if not np.allclose(teammate, agent_pos):
                    tv: NDArray[np.float32] = teammate - agent_pos
                    dist = float(np.linalg.norm(tv))
                    if dist > 0:
                        proj = float(np.dot(tv, direction))
                        if proj > 0:
                            perp = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                            if perp < detection_radius:
                                min_tm = min(min_tm, dist)
            dists["teammate"] = min_tm

            # GOALS
            min_goal: float = max_distance
            for goal in goals:
                gp: NDArray[np.float32] = (
                    np.array(goal["position"]) if isinstance(goal, dict) else np.array(goal)
                )
                gv: NDArray[np.float32] = gp - agent_pos
                dist = float(np.linalg.norm(gv))
                if dist > 0:
                    proj = float(np.dot(gv, direction))
                    if proj > 0:
                        perp = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                        if perp < detection_radius:
                            min_goal = min(min_goal, dist)
            dists["goal"] = min_goal

            ray_distances.extend([dists["ball"], dists["opponent"], dists["teammate"], dists["goal"]])

        arr: NDArray[np.float32] = np.array(ray_distances, dtype=np.float32)
        return np.minimum(arr / max_distance, 1.0)  # (32,)

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
        [YÊU CẦU 2]: Làm giàu Không gian Quan sát (Obs Enrichment)
        """
        ray_info: NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, ball, goals, max_distance=1.0
        )
        
        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)
        
        # [YÊU CẦU 2.1]: Cờ sở hữu bóng (Boolean flag)
        ball_owned_team = base_obs.get("ball_owned_team", -1)
        ball_owned_player = base_obs.get("ball_owned_player", -1)
        is_ball_owned = 1.0 if (ball_owned_team == 0 and ball_owned_player == agent_idx) else 0.0
        is_ball_owned_arr: NDArray[np.float32] = np.array([is_ball_owned], dtype=np.float32)
        
        # Trạng thái nút bấm (Sticky actions)
        sticky_actions_raw = raw_obs_i.get("sticky_actions", [0]*10)
        sticky_actions: NDArray[np.float32] = np.array(sticky_actions_raw, dtype=np.float32)
        
        # Vị trí sở trường (Left team roles)
        roles_raw = base_obs.get("left_team_roles", [0]*11)
        role_idx = roles_raw[agent_idx] if agent_idx < len(roles_raw) else 0
        role_onehot: NDArray[np.float32] = np.zeros(10, dtype=np.float32)
        if 0 <= role_idx < 10:
            role_onehot[role_idx] = 1.0
            
        obs_vec: NDArray[np.float32] = np.concatenate([
            ray_info,           # 32
            energy_field_info,  # 1
            is_ball_owned_arr,  # 1
            sticky_actions,     # 10
            role_onehot         # 10
        ])                      # Tổng: 54
        
        return obs_vec, ray_info

    def _build_global_state(self, base_obs: Dict[str, Any]) -> NDArray[np.float32]:
        """Trả về state toàn cục (46,) = left(22) + right(22) + ball(2)."""
        return np.concatenate([
            base_obs["left_team"].flatten(),
            base_obs["right_team"].flatten(),
            base_obs["ball"][:2],
        ])

    # =========================================================================
    # PHẦN 4: CORNER-KICK STATE MACHINE
    # =========================================================================

    def _get_corner_kicker_id(
        self, raw_obs: Optional[RawObs]
    ) -> Optional[int]:
        """Xác định ai là người đá phạt góc. Trả về None nếu không phải phạt góc."""
        if raw_obs is None:
            return None

        base_obs = raw_obs[0]
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )
        if is_corner_kick:
            left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
            distances: NDArray[np.float32] = np.sum((left_team - ball) ** 2, axis=1)
            return int(np.argmin(distances))

        return None

    def _get_aim_action(
        self,
        kicker_pos: NDArray[np.float32],
        target_pos: NDArray[np.float32],
    ) -> int:
        delta: NDArray[np.float32] = target_pos - kicker_pos
        norm: float = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return 0  

        angle: float = float(np.arctan2(float(delta[1]), float(delta[0]))) 
        sector: int = int(round(angle / (np.pi / 4))) % 8
        # GRF coordinate system: y increases DOWNWARDS.
        # sector 0 (angle 0): Right (5)
        # sector 1 (angle pi/4): BottomRight (6)
        # sector 2 (angle pi/2): Bottom (7)
        # sector 3 (angle 3pi/4): BottomLeft (8)
        # sector 4 (angle pi): Left (1)
        # sector 5 (angle -3pi/4): TopLeft (2)
        # sector 6 (angle -pi/2): Top (3)
        # sector 7 (angle -pi/4): TopRight (4)
        action_map: List[int] = [5, 6, 7, 8, 1, 2, 3, 4]
        return action_map[sector]

    # =========================================================================
    # PHẦN 5: ÁNH XẠ HÀNH ĐỘNG
    # =========================================================================

    def _map_global_actions(
        self,
        agent_actions: NDArray[np.int64],
        raw_obs:       Optional[RawObs] = None,
    ) -> NDArray[np.int64]:
        kicker_id: Optional[int] = self._kicker_id

        if kicker_id is not None and self._has_kicked and raw_obs is not None:
            base_obs = raw_obs[0]
            ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
            # [MODULE 2 - FIX 0]: Dùng key 'ball_direction' đúng, KHÔNG phải ball[3:5]
            ball_dir_raw = base_obs.get("ball_direction", [0.0, 0.0, 0.0])
            ball_speed: float = float(np.linalg.norm(ball_dir_raw[:2]))
            kicker_pos: NDArray[np.float32] = np.array(base_obs["left_team"][kicker_id])
            ball_dist_from_kicker: float = float(np.linalg.norm(ball - kicker_pos))

            if ball_speed > 0.1 or ball_dist_from_kicker > 0.05:
                self._kicker_id   = None
                self._aim_frames  = 0
                self._kick_frames = 0
                self._has_kicked  = False
                kicker_id         = None

        # Tính aim_action cho PHASE_AIM (dùng lại nhiều frame)
        aim_action: int = 0
        if kicker_id is not None and self._aim_frames < self._AIM_DURATION and raw_obs is not None:
            _b: NDArray[np.float32] = np.array(raw_obs[0]["ball"][:2])
            _s: float = 1.0 if _b[0] > 0 else -1.0
            kp: NDArray[np.float32] = np.array(raw_obs[0]["left_team"][kicker_id])
            
            # Ta tìm đồng đội ĐANG ĐỨNG TRONG VÒNG CẤM để nhắm chuẩn xác.
            left_team_pos = np.array(raw_obs[0]["left_team"])
            best_target: Optional[NDArray[np.float32]] = None
            min_dist_to_goal = float('inf')
            
            for j, pos in enumerate(left_team_pos):
                if j == kicker_id:
                    continue
                # Kiểm tra xem cầu thủ có trong/gần vòng cấm không (x > 0.65, |y| < 0.25)
                if _s * pos[0] > 0.7 and abs(pos[1]) < 0.1:
                    dist = float(np.linalg.norm(pos - np.array([_s * 1.0, 0.0])))
                    if dist < min_dist_to_goal:
                        min_dist_to_goal = dist
                        best_target = pos
            
            if best_target is None:
                # Fallback nếu không có ai trong vòng cấm
                best_target = np.array([_s * 0.85, 0.0], dtype=np.float32)
                
            aim_action = self._get_aim_action(kp, best_target)

        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        base_obs = raw_obs[0] if raw_obs is not None else None
        ball_owned_team = base_obs.get("ball_owned_team", -1) if base_obs is not None else -1
        ball_owned_player = base_obs.get("ball_owned_player", -1) if base_obs is not None else -1
        
        for i in range(self.num_agents):
            if i == kicker_id:
                if self._aim_frames < self._AIM_DURATION:
                    # PHASE_AIM: Lặp lại hành động xoay hướng trong nhiều frame
                    # để GRF engine kịp rotate cầu thủ quay mặt vào vòng cấm.
                    mapped_actions[i] = aim_action
                    self._aim_frames += 1
                elif self._kick_frames < self._KICK_DURATION:
                    # PHASE_KICK: Nhấn giữ nút High Pass (action 10) trong nhiều frame.
                    # 1 frame = đường chuyền rất nhẹ (sẽ rơi vào chân người gần nhất).
                    # 4-5 frame = tạt bổng sâu vào trong vòng cấm địa.
                    mapped_actions[i] = 10
                    self._kick_frames += 1
                else:
                    self._has_kicked  = True
                    mapped_actions[i] = 0
            else:
                # ĐỐI VỚI 10 CẦU THỦ CHẠY CHỖ: Ánh xạ chuẩn từ không gian [0 -> 9] của mạng nơ-ron
                # [FIX] Act 0-7 → GRF 1-8 (di chuyển 8 hướng), Act 8-9 → GRF 14 (STAY/release direction)
                act = int(agent_actions[i])
                if 0 <= act < 8:
                    mapped_actions[i] = act + 1   # GRF action 1-8 (8 hướng di chuyển)
                else:
                    mapped_actions[i] = 14        # GRF 14: release direction → đứng im
        return mapped_actions

    # =========================================================================
    # PHẦN 6: RESET & STEP
    # =========================================================================

    def reset(self, **kwargs: Any) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        raw_obs: RawObs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        self._kicker_id   = None
        self._aim_frames  = 0
        self._kick_frames = 0
        self._has_kicked  = False

        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )
        self.last_energy_fields = energy_fields

        base_obs = raw_obs[0]
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        state: NDArray[np.float32] = self._build_global_state(base_obs)

        results = [
            self._process_single_obs(
                i, left_team[i], left_team, right_team, ball, goals, energy_fields[i],
                raw_obs[i], base_obs
            )
            for i in range(self.num_agents)
        ]
        obses: NDArray[np.float32] = np.array([r[0] for r in results])

        return state, obses

    def step(
        self,
        actions: NDArray[np.int64],
    ) -> Tuple[
        NDArray[np.float32],                    # state           (46,)
        NDArray[np.float32],                    # obses           (11, 54)
        NDArray[np.float32],                    # shaped_rewards  (11,)
        Dict[str, NDArray[np.float32]],         # rewards_view
        bool,                                   # done
    ]:
        kicker_id: Optional[int] = self._get_corner_kicker_id(self.last_raw_obs)
        if kicker_id is not None and not self._has_kicked:
            self._kicker_id = kicker_id

        safe_actions: NDArray[np.int64] = self._map_global_actions(actions, raw_obs=self.last_raw_obs)
        raw_obs, rewards_list, done, info = self.env.step(safe_actions)

        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )

        base_obs = raw_obs[0]
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])

        # =========================================================================
        # [YÊU CẦU 3]: Thiết kế Hàm Phần Thưởng Chuẩn hóa [-1, 1] Tối ưu cho Phase 1
        # =========================================================================

        # Hằng số chuẩn hóa
        ENV_SCALE:       float = 0.1
        ENERGY_SCALE:    float = 0.08   # [FIX] Hạ từ 0.7 → 0.08 để tích lũy mịn, không lấn át bàn thắng
        POSSESSION_GAIN: float = 0.3
        POSSESSION_LOSS: float = -0.1
        HANDOVER_GAIN:   float = 1.0    # [FIX] Hệ số = 1.0 vì giá trị ±0.6 đã được tính toán chính xác

        # 1. R_env
        rewards_env: NDArray[np.float32] = np.array(rewards_list, dtype=np.float32) * ENV_SCALE

        # 2. R_energy
        """
        [Triết lý HES-COMA - R_energy]
        Khuyến khích chiếm lĩnh không gian trống. Chênh lệch (E_t - E_t+1) nhân hệ số
        tạo động lực cho các đặc vụ di chuyển liên tục vào các "hố năng lượng" 
        (vùng mục tiêu) đồng thời tránh xa các "đỉnh năng lượng" (đối thủ).
        """
        rewards_energy: NDArray[np.float32] = (
            (self.last_energy_fields - energy_fields) * ENERGY_SCALE  # type: ignore[operator]
        )

        current_ball_owned_team: int = base_obs.get("ball_owned_team", -1)
        current_ball_owned_player: int = base_obs.get("ball_owned_player", -1)

        # 4. R_possession
        rewards_possession: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        last_ball_owned: int = (
            self.last_raw_obs[0].get("ball_owned_team", -1) if self.last_raw_obs is not None else -1
        )
        
        if current_ball_owned_team == 0 and last_ball_owned != 0:
            rewards_possession[:] = POSSESSION_GAIN / self.num_agents
        elif current_ball_owned_team == 1 and last_ball_owned != 1:
            rewards_possession[:] = POSSESSION_LOSS / self.num_agents


        rewards_handover: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        # [FIX] Chỉ thưởng/phạt cầu thủ ĐANG CẦM BÓNG — không phạt đồng đội
        # vì phạt đồng đội STAY khi không có bóng là sai logic:
        # → Ép TẤT CẢ phải MOVE → không có mục tiêu rõ → chọn hướng energy cao nhất → bầy đàn 1 hướng
        if current_ball_owned_team == 0 and current_ball_owned_player != -1:
            idx = current_ball_owned_player
            original_agent_action = int(actions[idx])
            if original_agent_action >= 8:    # Actor chọn STAY (action 8-9) → đúng, giữ bóng để chuyền
                rewards_handover[idx] =  0.6
            elif 0 <= original_agent_action < 8:  # Actor chọn MOVE → phạt vì di chuyển khi đang cầm bóng
                rewards_handover[idx] = -0.6
        rewards_handover = rewards_handover * HANDOVER_GAIN
        # Tổng hợp và chuẩn hóa [-1.0, 1.0]
        shaped_rewards: NDArray[np.float32] = (
            rewards_env + rewards_energy + rewards_possession + rewards_handover 
        )
          # Chuẩn hóa riêng để dễ phân tích tác động
        shaped_rewards = np.clip(shaped_rewards, -1.0, 1.0)

        # ── Cập nhật buffer cho bước kế tiếp ───────────────────────────────
        self.last_raw_obs       = raw_obs
        self.last_energy_fields = energy_fields

        state: NDArray[np.float32] = self._build_global_state(base_obs)

        results = [
            self._process_single_obs(
                i, left_team[i], left_team, right_team, ball, goals, energy_fields[i],
                raw_obs[i], base_obs
            )
            for i in range(self.num_agents)
        ]
        obses: NDArray[np.float32] = np.array([r[0] for r in results])

        rewards_view: Dict[str, NDArray[np.float32]] = {
            "rewards_env":        rewards_env,
            "rewards_energy":     rewards_energy,
            "rewards_possession": rewards_possession,
            "rewards_handover":   rewards_handover,
        }

        return state, obses, shaped_rewards, rewards_view, done