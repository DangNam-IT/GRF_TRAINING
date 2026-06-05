from __future__ import annotations

import random
from typing import Any, Optional, Tuple, List

import gym
import numpy as np
from numpy.typing import NDArray

from utils.energy_field import EnergyFieldDefiner, FieldItem

# Kiểu raw observation của GRF
RawObs = list[dict[str, Any]]


class GFootballLocalWrapper(gym.Wrapper):
    """
    Wrapper Phase 2 (Local Spatial Hierarchy) của HES-COMA.

    Theo paper:
    - GAgent (frozen từ Phase 1) quyết định di chuyển (0-7) hoặc STOP (8).
    - Chỉ khi GAgent chọn STOP, LAgent mới thực thi tactical action.
    - Energy field CHỈ dùng ở global hierarchy → obs_l KHÔNG chứa energy.
    - obs_g = 32 rays + 1 energy = 33 chiều (giữ nguyên chuẩn Phase 1).
    - obs_l = 32 rays (ray-info thuần, không có energy).
    """

    # ── Hằng số hành động ──────────────────────────────────────────────────────
    # GAgent output [0..8]: 0-7 → di chuyển (GRF 1-8), 8 → STOP (GRF 14)
    STOP_ACTIONS: frozenset[int] = frozenset({8})

    # LAgent output [0..n_tactic_actions-1] → GRF tactical action
    # 0: Short Pass (11), 1: Shot (12)
    TACTIC_MAP: list[int] = [11, 12]

    def __init__(self, env: gym.Env, num_agents: int = 11) -> None:
        super().__init__(env)
        self.num_agents:       int = num_agents
        self.n_tactic_actions: int = 2

        # ── Không gian quan sát ───────────────────────────────────────────────
        self.state_dim:  int = 46  # left(22) + right(22) + ball(2)
        # obs_dim_g đồng bộ Phase 1: rays(32) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 54
        self.obs_dim_g:  int = 54
        self.obs_dim_l:  int = 32  # Ray-info thuần — KHÔNG có energy

        # Energy field vẫn cần để xây dựng obs_g
        self.energy_definer: EnergyFieldDefiner = EnergyFieldDefiner()

        # ── State machine phạt góc (mirror Global) ───────────────────────────
        self.last_raw_obs: Optional[RawObs] = None
        self._kicker_id:   Optional[int]    = None
        self._has_kicked:  bool             = False
        self._aim_frames:  int              = 0
        self._kick_frames: int              = 0
        self._AIM_DURATION: int             = 5
        self._KICK_DURATION: int            = 4

    # =========================================================================
    # PHẦN 1: TRÍCH XUẤT VỊ TRÍ (cho Energy Field của GAgent)
    # =========================================================================

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
        Tích hợp Dynamic Energy Fields:
        - Anticipatory Attractors (ball_direction)
        - Directional Repulsors (right_team_direction)
        Trả về thêm right_team và ball để tái sử dụng ở các hàm khác.

        [MODULE 2 - FIX 0]: ball_direction đọc từ key riêng 'ball_direction',
        KHÔNG phải từ base_obs['ball'][3:5] (ball chỉ có 3 phần tử [x,y,z]).
        [MODULE 2 - FIX 2]: Phát hiện ball_in_flight — khóa attractor cố định
        khi bóng bay bổng trên không (ball_z > 0.08).
        """
        base_obs   = raw_obs[0]
        left_team:  NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )

        # ball_z:     float               = float(base_obs["ball"][2])
        # # [MODULE 2 - FIX 0]: Dùng key 'ball_direction' riêng (observation.md)
        # ball_dir_raw = base_obs.get("ball_direction", [0.0, 0.0, 0.0])
        # ball_direction: NDArray[np.float32] = np.array(ball_dir_raw[:2], dtype=np.float32)

        # right_team_direction_raw = base_obs.get("right_team_direction", [[0.0, 0.0]] * len(right_team))
        # right_team_direction: NDArray[np.float32] = np.array(right_team_direction_raw, dtype=np.float32)


        # if is_corner_kick:
        #     target_sign: float = 1.0 if ball[0] > 0 else -1.0

        #     if ball_in_flight:
        #         # [MODULE 2 - FIX 2]: Bóng bay bổng — Landing Zone cố định
        #         near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  0.05])
        #         far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -0.05])
        #         penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])
        #     else:
        #         # Anticipatory Attractors: dịch chuyển mục tiêu theo quỹ đạo bóng dự kiến
        #         k_ball: float = 5.0
        #         predicted_ball_pos: NDArray[np.float32] = ball + ball_direction * k_ball
        #         near_post    = np.array([target_sign * 0.9,  predicted_ball_pos[1] * 0.1])
        #         far_post     = np.array([target_sign * 0.9, -predicted_ball_pos[1] * 0.1])
        #         penalty_spot = np.array([target_sign * 0.8,  0.0])

        #     goals = [
        #         {"position": near_post,    "sigma": 0.15, "scale": -3.0},
        #         {"position": far_post,     "sigma": 0.20, "scale": -2.0},
        #         {"position": penalty_spot, "sigma": 0.25, "scale": -2.5},
        #     ]

        #     # Directional Repulsors: đẩy tâm chướng ngại vật theo hướng chạy hậu vệ
        #     # [MODULE 2 - FIX 3]: sigma=0.06 giữ kẽ hở phòng ngự
        #     k_opp: float = 3.0
        #     obstacles = []
        #     for j, pos in enumerate(right_team):
        #         opp_dir: NDArray[np.float32] = (
        #             right_team_direction[j] if j < len(right_team_direction)
        #             else np.zeros(2, dtype=np.float32)
        #         )
        #         predicted_opp_pos: NDArray[np.float32] = pos + opp_dir * k_opp
        #         obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 1.5})

        #     if not ball_in_flight:
        #         predicted_ball_obs: NDArray[np.float32] = ball + ball_direction * k_opp
        #         obstacles.append({"position": predicted_ball_obs, "sigma": 0.15, "scale": 2.5})

        # else:
        #     k_ball = 5.0
        #     predicted_ball_pos = ball + ball_direction * k_ball
        #     goals = [{"position": predicted_ball_pos, "sigma": 0.25, "scale": -2.0}]

        #     k_opp = 3.0
        #     obstacles = []
        #     for j, pos in enumerate(right_team):
        #         opp_dir = (
        #             right_team_direction[j] if j < len(right_team_direction)
        #             else np.zeros(2, dtype=np.float32)
        #         )
        #         predicted_opp_pos = pos + opp_dir * k_opp
        #         obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 1.2})

        obstacles = []

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
            # obstacles = [
            #     {"position": pos, "sigma": 0.06, "scale": 0.2} for pos in right_team
            # ]
            obstacles = [{"position": ball, "sigma": 0.25, "scale": 0.5}]

        else:
            goals = [{"position": ball, "sigma": 0.25, "scale": -2.5}]
            obstacles = [{"position": pos, "sigma": 0.06, "scale": -0.1} for pos in right_team]

        return left_team, right_team, ball, goals, obstacles

    # =========================================================================
    # PHẦN 2: RAY-CAST (cho obs_g và obs_l — mirror wrapper_global)
    # =========================================================================

    def _raycast_from_agent(
        self,
        agent_pos:    NDArray[np.float32],
        left_team:    NDArray[np.float32],
        right_team:   NDArray[np.float32],
        ball:         NDArray[np.float32],
        goals:        list[FieldItem],
        max_distance: float = 0.5,
    ) -> NDArray[np.float32]:
        """8 hướng × 4 object types → (32,) normalized [0, 1]."""
        angles:           NDArray[np.float32] = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        detection_radius: float               = 0.025
        ray_distances:    list[float]         = []

        for angle in angles:
            direction: NDArray[np.float32] = np.array([np.cos(angle), np.sin(angle)])
            dists: dict[str, float] = {
                "ball": max_distance, "opponent": max_distance,
                "teammate": max_distance, "goal": max_distance,
            }

            # BALL
            bv:   NDArray[np.float32] = ball - agent_pos
            dist: float = float(np.linalg.norm(bv))
            if dist > 0:
                proj: float = float(np.dot(bv, direction))
                if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                    dists["ball"] = dist

            # OPPONENTS
            min_opp: float = max_distance
            for opp in right_team:
                ov:   NDArray[np.float32] = opp - agent_pos
                dist = float(np.linalg.norm(ov))
                if dist > 0:
                    proj = float(np.dot(ov, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                        min_opp = min(min_opp, dist)
            dists["opponent"] = min_opp

            # TEAMMATES
            min_tm: float = max_distance
            for tm in left_team:
                if not np.allclose(tm, agent_pos):
                    tv:   NDArray[np.float32] = tm - agent_pos
                    dist = float(np.linalg.norm(tv))
                    if dist > 0:
                        proj = float(np.dot(tv, direction))
                        if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                            min_tm = min(min_tm, dist)
            dists["teammate"] = min_tm

            # GOALS
            min_goal: float = max_distance
            for g in goals:
                gp:   NDArray[np.float32] = np.array(g["position"]) if isinstance(g, dict) else np.array(g)
                gv:   NDArray[np.float32] = gp - agent_pos
                dist = float(np.linalg.norm(gv))
                if dist > 0:
                    proj = float(np.dot(gv, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < detection_radius:
                        min_goal = min(min_goal, dist)
            dists["goal"] = min_goal

            ray_distances.extend([dists["ball"], dists["opponent"], dists["teammate"], dists["goal"]])

        arr: NDArray[np.float32] = np.array(ray_distances, dtype=np.float32)
        return np.minimum(arr / max_distance, 1.0)  # (32,)

    # =========================================================================
    # PHẦN 3: XÂY DỰNG QUAN SÁT
    # =========================================================================

    def _process_global_obs(
        self,
        agent_idx:  int,
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      list[FieldItem],
        energy_val: float,
        raw_obs_i:  dict[str, Any],
        base_obs:   dict[str, Any],
    ) -> NDArray[np.float32]:
        """
        obs_g: Mirror wrapper_global._process_single_obs (54 chiều).
        rays(32) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 54.
        Giữ nguyên chuẩn Phase 1 để load model đúng shape.
        """
        ray_info: NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, ball, goals
        )
        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)

        # Cờ sở hữu bóng (Boolean flag)
        ball_owned_team = base_obs.get("ball_owned_team", -1)
        ball_owned_player = base_obs.get("ball_owned_player", -1)
        is_ball_owned = 1.0 if (ball_owned_team == 0 and ball_owned_player == agent_idx) else 0.0
        is_ball_owned_arr: NDArray[np.float32] = np.array([is_ball_owned], dtype=np.float32)

        # Trạng thái nút bấm (Sticky actions)
        sticky_actions_raw = raw_obs_i.get("sticky_actions", [0] * 10)
        sticky_actions: NDArray[np.float32] = np.array(sticky_actions_raw, dtype=np.float32)

        # Vị trí sở trường (Left team roles — one-hot)
        roles_raw = base_obs.get("left_team_roles", [0] * 11)
        role_idx = roles_raw[agent_idx] if agent_idx < len(roles_raw) else 0
        role_onehot: NDArray[np.float32] = np.zeros(10, dtype=np.float32)
        if 0 <= role_idx < 10:
            role_onehot[role_idx] = 1.0

        return np.concatenate([
            ray_info,           # 32
            energy_field_info,  # 1
            is_ball_owned_arr,  # 1
            sticky_actions,     # 10
            role_onehot,        # 10
        ])  # (54,)

    def _process_local_obs(
        self,
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      list[FieldItem],
    ) -> NDArray[np.float32]:
        """
        obs_l: Ray-info thuần (32 chiều) — KHÔNG có energy field.

        Theo paper §4.2: "energy field is used exclusively within the global
        spatial hierarchy". LAgent nhận thức không gian cục bộ qua ray-cast
        để quyết định tactical action (pass/shoot/steal/...).

        Returns:
            (32,) = 8 hướng × 4 object types, normalized [0, 1].
        """
        return self._raycast_from_agent(agent_pos, left_team, right_team, ball, goals)

    def _build_global_state(self, base_obs: dict[str, Any]) -> NDArray[np.float32]:
        """Trả về state toàn cục (46,) = left(22) + right(22) + ball(2)."""
        return np.concatenate([
            base_obs["left_team"].flatten(),
            base_obs["right_team"].flatten(),
            base_obs["ball"][:2],
        ])

    # =========================================================================
    # PHẦN 4: CORNER-KICK STATE MACHINE (mirror wrapper_global)
    # =========================================================================

    def _get_corner_kicker_id(
        self, raw_obs: Optional[RawObs]
    ) -> Optional[int]:
        """Trả về ID người đứng gần bóng góc nhất, hoặc None nếu không phải phạt góc."""
        if raw_obs is None:
            return None
        base_obs = raw_obs[0]
        # Thống nhất: sử dụng game_mode == 4 để nhận diện phạt góc
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )
        if is_corner_kick:
            ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
            left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
            return int(np.argmin(np.sum((left_team - ball) ** 2, axis=1)))
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

    def _get_facing_cosine(
        self,
        agent_idx:   int,
        player_pos:  NDArray[np.float32],   # (2,) vị trí cầu thủ
        ball_pos:    NDArray[np.float32],   # (2,) vị trí bóng
        raw_obs_0:   dict,                  # raw_obs[0] — base obs
    ) -> float:
        """
        Tính cos θ giữa hướng mặt cầu thủ (suy ra từ sticky_actions)
        và vector cầu thủ → bóng.

        sticky_actions[0..7] tương ứng:
          0=Left(-1,0), 1=TopLeft(-1,-1), 2=Top(0,-1), 3=TopRight(1,-1),
          4=Right(1,0), 5=BottomRight(1,1), 6=Bottom(0,1), 7=BottomLeft(-1,1)
        (GRF: y tăng xuống dưới)

        Trả về:
          cos θ ∈ [-1, 1]  — 1.0 nếu nhìn thẳng vào bóng.
          0.0              — nếu không giữ phím hướng nào (đứng im).
        """
        # 8 direction vectors tương ứng sticky_actions bit 0-7
        DIR_VECTORS: List[NDArray[np.float32]] = [
            np.array([-1.0,  0.0]),   # 0: Left
            np.array([-1.0, -1.0]),   # 1: TopLeft
            np.array([ 0.0, -1.0]),   # 2: Top
            np.array([ 1.0, -1.0]),   # 3: TopRight
            np.array([ 1.0,  0.0]),   # 4: Right
            np.array([ 1.0,  1.0]),   # 5: BottomRight
            np.array([ 0.0,  1.0]),   # 6: Bottom
            np.array([-1.0,  1.0]),   # 7: BottomLeft
        ]
        # GRF lưu sticky_actions per-agent trong raw_obs[agent_idx]
        try:
            sticky = raw_obs_0.get("left_team_active", None)
            # sticky_actions nằm trong raw_obs mỗi agent — dùng index trực tiếp
            agent_obs = self.last_raw_obs[agent_idx] if self.last_raw_obs else None
            sticky_bits = agent_obs["sticky_actions"] if agent_obs else None
        except (IndexError, KeyError, TypeError):
            return 0.0

        if sticky_bits is None:
            return 0.0

        # Tổng hợp direction vector từ tất cả các bit hướng đang được giữ
        facing_vec: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        for bit_idx in range(8):
            if sticky_bits[bit_idx]:
                facing_vec += DIR_VECTORS[bit_idx]

        face_norm: float = float(np.linalg.norm(facing_vec))
        if face_norm < 1e-6:
            return 0.0   # Không giữ phím hướng nào → không xác định được mặt

        facing_vec /= face_norm

        # Vector cầu thủ → bóng
        to_ball: NDArray[np.float32] = ball_pos - player_pos
        ball_norm: float = float(np.linalg.norm(to_ball))
        if ball_norm < 1e-6:
            return 1.0   # Đứng ngay trên bóng → coi như facing

        to_ball /= ball_norm
        return float(np.dot(facing_vec, to_ball))

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
                if _s * pos[0] > 0.65 and abs(pos[1]) < 0.1:
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
        # Tìm 5 cầu thủ đội nhà gần khung thành đối phương nhất (nhóm tấn công chủ chốt)
        left_team = np.array(raw_obs[0]['left_team'])
        dist_to_goal = np.sum((left_team - np.array([1.0, 0.0]))**2, axis=1)
        
        # Bốc ra ID của 5 người gần gôn nhất (trừ thủ môn và kicker)
        key_attacker_ids = np.argsort(dist_to_goal)[:5]
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
            elif i in key_attacker_ids:
                # ĐỐI VỚI 10 CẦU THỦ CHẠY CHỖ: Ánh xạ chuẩn từ không gian [0 -> 9] của mạng nơ-ron
                act = int(agent_actions[i])
                if 0 <= act < 8:
                    mapped_actions[i] = act+1
                else:
                    mapped_actions[i] = 14  # Lệnh giải phóng phím dính, đứng im rình rập khoảng trống
            else: 
                mapped_actions[i] = agent_actions[i]
        return mapped_actions

    # =========================================================================
    # PHẦN 6: HÀM TIỆN ÍCH CHUNG
    # =========================================================================

    def _get_all_obses_and_state(
        self, raw_obs: RawObs
    ) -> Tuple[
        NDArray[np.float32],  # state  (46,)
        NDArray[np.float32],  # obs_g  (11, 54)
        NDArray[np.float32],  # obs_l  (11, 32)
    ]:
        """
        Trích xuất (state, obs_g, obs_l) từ raw_obs.
          obs_g: rays(32) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 54  — Global hierarchy
          obs_l: 32 rays (không energy)   — Local hierarchy
        """
        left_team, right_team, ball, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )

        base_obs = raw_obs[0]
        state: NDArray[np.float32] = self._build_global_state(base_obs)

        obs_g: NDArray[np.float32] = np.array([
            self._process_global_obs(
                i, left_team[i], left_team, right_team, ball, goals,
                energy_fields[i], raw_obs[i], base_obs
            )
            for i in range(self.num_agents)
        ])  # (11, 54)

        obs_l: NDArray[np.float32] = np.array([
            self._process_local_obs(left_team[i], left_team, right_team, ball, goals)
            for i in range(self.num_agents)
        ])  # (11, 32)

        return state, obs_g, obs_l

    # =========================================================================
    # PHẦN 7: RESET & STEP
    # =========================================================================

    def reset(
        self, **kwargs: Any
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Returns:
            state:  (46,)
            obs_g:  (11, 54)
            obs_l:  (11, 32)
        """
        raw_obs: RawObs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        self._kicker_id   = None
        self._has_kicked  = False
        self._aim_frames  = 0
        self._kick_frames = 0
        return self._get_all_obses_and_state(raw_obs)

    def step_global(
        self,
        global_actions: NDArray[np.int64],
    ) -> Tuple[
        NDArray[np.float32],  # state  (46,)
        NDArray[np.float32],  # obs_g  (11, 54)
        NDArray[np.float32],  # obs_l  (11, 32)
        NDArray[np.float32],  # rewards (11,) — zeros (LAgent không hành động)
        bool,                 # done
    ]:
        """Bước do GAgent quyết định — LAgent không hành động."""
        kicker_id: Optional[int] = self._get_corner_kicker_id(self.last_raw_obs)
        if kicker_id is not None and not self._has_kicked:
            self._kicker_id = kicker_id

        safe_actions: NDArray[np.int64] = self._map_global_actions(
            global_actions, raw_obs=self.last_raw_obs
        )
        raw_obs, rewards, done, info = self.env.step(safe_actions)

        self.last_raw_obs = raw_obs
        
        state, obs_g, obs_l = self._get_all_obses_and_state(raw_obs)
        return state, obs_g, obs_l, np.zeros(self.num_agents, dtype=np.float32), done

    def step_local(
        self,
        local_actions:  NDArray[np.int64],   # (11,) ∈ [0..n_tactic_actions-1]
        active_mask:    NDArray[np.bool_],    # (11,) True nếu GAgent chọn STOP
        global_actions: NDArray[np.int64],   # (11,) output gốc của GAgent
    ) -> Tuple[
        NDArray[np.float32],  # state   (46,)
        NDArray[np.float32],  # obs_g   (11, 54)
        NDArray[np.float32],  # obs_l   (11, 32)
        NDArray[np.float32],  # shaped_rewards (11,)
        bool,                 # done
        dict,                 # reward_info — các phần thưởng thành phần (sum theo agent)
    ]:
        """
        Bước khi ít nhất một GAgent chọn STOP.

        Reward shaping theo Reward_logic.md §2:
          R_i^l = R_env + 0.5·I(Passing) + 1.0·I(Assist) + R_role_based + R_ball_approach + R_possession

        - R_env:          GRF sparse reward (normalized).
        - R_passing:      +0.5 khi kicker chuyền thành công (ball_owned chuyển sang agent khác).
        - R_assist:       +1.0 khi kicker tạo ra assist (bàn thắng sau pha kiến tạo).
        - R_role_based:   Thưởng/phạt theo vùng chiến thuật (near_post, far_post, penalty_spot).
        - R_ball_approach: Sau khi bóng vào vòng cấm, khuyến khích agent chủ động áp sát.
        - R_possession:   Possession change (+/-).
        """
        # ── Hằng số reward (Table 13 §2) ─────────────────────────────────────
        ENV_SCALE:         float = 1.0 / self.num_agents
        PASSING_REWARD:    float =  0.5   # Kicker tạt bóng thành công
        ASSIST_REWARD:     float =  1.0   # Kiến tạo dẫn đến bàn thắng
        POSSESSION_LOSS:   float = -2.0
        POSSESSION_GAIN:   float =  1.0
        BALL_APPROACH_R:   float =  0.3   # Tiến về điểm rơi sau khi bóng vào vùng cấm
        # Role-based (§1.4): thưởng theo vùng chiến thuật khi ghi bàn
        ROLE_NEAR_POST:    float =  1.0
        ROLE_FAR_POST:     float =  1.0
        ROLE_PENALTY_SPOT: float =  1.5   # Khó hơn → thưởng thêm hệ số khuyến khích
        # ── Cơ chế chuyền có chủ đích ─────────────────────────────────────────
        FACING_THRESHOLD:  float =  0.7   # cos θ tối thiểu để pass được coi là "trực diện"
        BAD_FACING_PENALTY: float = -0.4  # [RL trial-and-error] Phạt pass khi chưa nhìn thẳng bóng
        # ── Cơ chế ưu tiên sút trong vòng cấm ────────────────────────────────
        BOX_X_THRESHOLD:   float =  0.83  # GRF pitch x ∈ [-1, 1]
        BOX_Y_THRESHOLD:   float =  0.20  # GRF pitch |y| ∈ [0, 0.42]
        SHOT_IN_BOX_R:     float =  1.5   # Thưởng khi sút trong vòng cấm
        PASS_IN_BOX_P:     float = -0.8   # Phạt khi chuyền trong vòng cấm (mất cơ hội)

        # ── Đọc trạng thái trước bước ─────────────────────────────────────────
        last_obs  = self.last_raw_obs[0] if self.last_raw_obs is not None else None
        last_ball_owned:   int = last_obs["ball_owned_team"]           if last_obs else -1
        last_ball_owned_player: int = last_obs.get("ball_owned_player", -1) if last_obs else -1
        last_score: list[int]  = list(last_obs["score"])               if last_obs else [0, 0]

        # ── Corner-kick state machine (mirror _map_global_actions) ───────────
        kicker_id: Optional[int] = self._get_corner_kicker_id(self.last_raw_obs)

        if kicker_id is not None and self._has_kicked and last_obs is not None:
            b_prev:  NDArray[np.float32] = np.array(last_obs["ball"][:2])
            # [MODULE 2 - FIX 0]: Dùng key 'ball_direction' đúng
            spd_raw = last_obs.get("ball_direction", [0.0, 0.0, 0.0])
            spd:     float = float(np.linalg.norm(spd_raw[:2]))
            kp: NDArray[np.float32] = np.array(last_obs["left_team"][kicker_id])
            if spd > 0.1 or float(np.linalg.norm(b_prev - kp)) > 0.05:
                self._kicker_id   = None
                self._aim_frames  = 0
                self._kick_frames = 0
                self._has_kicked  = False
                kicker_id         = None

        # Tính aim_action cho PHASE_AIM (dùng lại nhiều frame)
        aim_action: int = 0
        if kicker_id is not None and self._aim_frames < self._AIM_DURATION and last_obs is not None:
            _b: NDArray[np.float32] = np.array(last_obs["ball"][:2])
            _s: float = 1.0 if _b[0] > 0 else -1.0
            kp: NDArray[np.float32] = np.array(last_obs["left_team"][kicker_id])
            
            left_team_pos = np.array(last_obs["left_team"])
            best_target: Optional[NDArray[np.float32]] = None
            min_dist_to_goal = float('inf')
            
            for j, pos in enumerate(left_team_pos):
                if j == kicker_id:
                    continue
                if _s * pos[0] > 0.7 and abs(pos[1]) < 0.1:
                    dist = float(np.linalg.norm(pos - np.array([_s * 1.0, 0.0])))
                    if dist < min_dist_to_goal:
                        min_dist_to_goal = dist
                        best_target = pos
            
            if best_target is None:
                best_target = np.array([_s * 0.85, 0.0], dtype=np.float32)
                
            aim_action = self._get_aim_action(kp, best_target)

        # ── Ánh xạ hành động ─────────────────────────────────────────────────
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            if active_mask[i]:
                tactic_idx        = int(local_actions[i]) % self.n_tactic_actions
                mapped_actions[i] = self.TACTIC_MAP[tactic_idx]
            else:
                if i == kicker_id:
                    if self._aim_frames < self._AIM_DURATION:
                        mapped_actions[i] = aim_action
                        self._aim_frames += 1
                    elif self._kick_frames < self._KICK_DURATION:
                        mapped_actions[i] = 9
                        self._kick_frames += 1
                    else:
                        self._has_kicked  = True
                        mapped_actions[i] = 0
                else:
                    act: int = int(global_actions[i])
                    mapped_actions[i] = (act + 1) if 0 <= act < 8 else 14

        raw_obs, rewards_list, done, info = self.env.step(mapped_actions)
        current_obs = raw_obs[0]

        # ── R_env: GRF sparse reward (normalized) ────────────────s────────────
        rewards_env: NDArray[np.float32] = (
            np.array(rewards_list, dtype=np.float32) * ENV_SCALE
        )

        shaped_rewards: NDArray[np.float32] = rewards_env.copy()

        # ── Đọc trạng thái sau bước ──────────────────────────────────────────
        current_ball_owned:        int       = current_obs["ball_owned_team"]
        current_ball_owned_player: int       = current_obs.get("ball_owned_player", -1)
        current_score: list[int]             = list(current_obs["score"])
        left_team:     NDArray[np.float32]   = np.array(current_obs["left_team"])
        ball:          NDArray[np.float32]   = np.array(current_obs["ball"][:2])

        # ── R_passing: Kicker chuyền thành công (+0.5) ───────────────────────
        # Điều kiện: kicker_id đã được đặt, và quyền sở hữu bóng chuyển sang
        # đồng đội khác (ball_owned_team=0, ball_owned_player != kicker_id).
        rewards_passing: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if (
            kicker_id is not None
            and last_ball_owned != 0          # trước chưa phải đội ta sở hữu
            and current_ball_owned == 0       # sau đội ta có bóng
            and current_ball_owned_player != kicker_id  # người khác nhận
        ):
            rewards_passing[kicker_id] = PASSING_REWARD
            shaped_rewards[kicker_id] += PASSING_REWARD

        # ── R_facing: Phạt chuyền khi chưa "trực diện" bóng (RL trial-and-error) ──
        # Nguyên lý: Vẫn thực thi action để agent nhận tín hiệu thật từ môi trường,
        # nhưng cộng thêm reward âm để agent học rằng pass lúc này là sai lầm.
        # Agent sẽ học: "đợi quay mặt vào bóng (cosθ > 0.7) rồi mới chuyền".
        rewards_facing: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if last_obs is not None:
            last_left_team: NDArray[np.float32] = np.array(last_obs["left_team"])
            last_ball: NDArray[np.float32]       = np.array(last_obs["ball"][:2])
            for i in range(self.num_agents):
                # Chỉ kiểm tra agent đang active (LAgent vừa thực thi tactical action)
                # và đã chọn PASS (local action 0 = TACTIC_MAP[0] = Short Pass GRF 11)
                if active_mask[i] and int(local_actions[i]) % self.n_tactic_actions == 0:
                    cos_theta: float = self._get_facing_cosine(
                        i, last_left_team[i], last_ball, last_obs
                    )
                    if cos_theta <= FACING_THRESHOLD:
                        rewards_facing[i]   = BAD_FACING_PENALTY
                        shaped_rewards[i]  += BAD_FACING_PENALTY

        # ── R_in_box: Soft reward ưu tiên sút trong vòng cấm ─────────────────
        # Nếu agent đang cầm bóng VÀ đứng trong vùng nguy hiểm:
        #   - Chọn Shot (action 1) → thưởng thêm: khuyến khích dứt điểm ngay
        #   - Chọn Pass (action 0) → phạt thêm: mất cơ hội vàng
        # Vùng vòng cấm GRF: x > 0.83, |y| < 0.20 (sign theo phía tấn công)
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
                if in_box and active_mask[idx]:
                    tactic_chosen: int = int(local_actions[idx]) % self.n_tactic_actions
                    if tactic_chosen == 1:  # Shot (TACTIC_MAP[1] = GRF 12)
                        rewards_in_box[idx]  = SHOT_IN_BOX_R
                        shaped_rewards[idx] += SHOT_IN_BOX_R
                    elif tactic_chosen == 0:  # Pass (TACTIC_MAP[0] = GRF 11)
                        rewards_in_box[idx]  = PASS_IN_BOX_P
                        shaped_rewards[idx] += PASS_IN_BOX_P

        # ── R_assist: Kiến tạo dẫn đến bàn thắng (+1.0) ─────────────────────
        rewards_assist: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        scored_this_step: bool = current_score[0] > last_score[0]
        if scored_this_step and kicker_id is not None and self._has_kicked:
            rewards_assist[kicker_id] = ASSIST_REWARD
            shaped_rewards[kicker_id] += ASSIST_REWARD

        # ── R_role_based: Thưởng theo vùng chiến thuật khi ghi bàn ──────────
        # Xác định target_sign từ vị trí bóng ban đầu (trước reset)
        rewards_role: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if scored_this_step and last_obs is not None:
            ball_init: NDArray[np.float32] = np.array(last_obs["ball"][:2])
            sign: float = 1.0 if ball_init[0] > 0 else -1.0

            # Vùng chiến thuật (theo phân tích §1.4 Reward_logic.md)
            near_post_zone:    NDArray[np.float32] = np.array([sign * 0.9,  ball_init[1] * 0.1])
            far_post_zone:     NDArray[np.float32] = np.array([sign * 0.9, -ball_init[1] * 0.1])
            penalty_spot_zone: NDArray[np.float32] = np.array([sign * 0.8,  0.0])

            for i in range(self.num_agents):
                pos: NDArray[np.float32] = left_team[i]
                if float(np.linalg.norm(pos - near_post_zone)) < 0.15:
                    rewards_role[i] = ROLE_NEAR_POST
                    shaped_rewards[i] += ROLE_NEAR_POST
                elif float(np.linalg.norm(pos - far_post_zone)) < 0.15:
                    rewards_role[i] = ROLE_FAR_POST
                    shaped_rewards[i] += ROLE_FAR_POST
                elif float(np.linalg.norm(pos - penalty_spot_zone)) < 0.20:
                    rewards_role[i] = ROLE_PENALTY_SPOT
                    shaped_rewards[i] += ROLE_PENALTY_SPOT

        # ── R_ball_approach: Thưởng Sparse khi chạm cắt bóng thành công ─────
        rewards_approach: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if current_ball_owned == 0 and current_ball_owned_player != -1:
            if last_ball_owned != 0 or last_ball_owned_player != current_ball_owned_player:
                # Chỉ thưởng 1 lần khi giành được quyền kiểm soát bóng
                rewards_approach[current_ball_owned_player] = BALL_APPROACH_R
                shaped_rewards[current_ball_owned_player] += BALL_APPROACH_R

        # ── R_possession: Kiểm soát bóng (Table 13 §1) ───────────────────────
        rewards_possession: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if current_ball_owned == 1 and last_ball_owned != 1:
            rewards_possession[:] = POSSESSION_LOSS / self.num_agents
            shaped_rewards       += rewards_possession
        elif current_ball_owned == 0 and last_ball_owned != 0:
            rewards_possession[:] = POSSESSION_GAIN / self.num_agents
            shaped_rewards       += rewards_possession

        self.last_raw_obs = raw_obs
        state, obs_g, obs_l = self._get_all_obses_and_state(raw_obs)

        # ── Tổng hợp các phần thưởng thành phần (sum trên tất cả agent) ──────
        reward_info: dict = {
            'R_env':        float(np.sum(rewards_env)),
            'R_passing':    float(np.sum(rewards_passing)),
            'R_facing':     float(np.sum(rewards_facing)),
            'R_in_box':     float(np.sum(rewards_in_box)),
            'R_assist':     float(np.sum(rewards_assist)),
            'R_role':       float(np.sum(rewards_role)),
            'R_approach':   float(np.sum(rewards_approach)),
            'R_possession': float(np.sum(rewards_possession)),
        }

        return state, obs_g, obs_l, shaped_rewards, done, reward_info