from __future__ import annotations

from operator import is_
import random
from typing import Any, Optional, Tuple, Dict, List

import gym
import numpy as np
from numpy.typing import NDArray

from utils.energy_field import EnergyFieldDefiner, FieldItem
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
        # obs_dim = rays(48) + energy(1) + ball_owned(1) + sticky_actions(10) + role(10) = 70
        # rays = 16 tia × 3 kênh (opponent, teammate, target_zone)
        self.N_RAYS:   int = 16
        self.obs_dim:  int = 70

        self.last_raw_obs: Optional[RawObs] = None
        self.last_energy_fields: Optional[NDArray[np.float32]] = None

        self._kicker_id:   Optional[int] = 1
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
        """
        base_obs = raw_obs[0]
        left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        obstacles = []

        target_sign:  float = 1.0 if ball[0] > 0 else -1.0
        near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  ball[1] * 0.1])
        far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -ball[1] * 0.1])
        penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])

        # THAY ĐỔI: Thu hẹp Sigma, Đào sâu Scale để tạo khoảng trống rõ rệt
        # Tăng mạnh sigma để lực hút lan tỏa ra xa, giải quyết Vanishing Gradient
        goals = [
            {"position": near_post,    "sigma": 0.3, "scale": -2.0},
            {"position": far_post,     "sigma": 0.3, "scale": -1.5},
            {"position": penalty_spot, "sigma": 0.6, "scale": -2.5},
        ]
        # THAY ĐỔI: Thu hẹp Sigma của hậu vệ để Agent có kẽ hở luồn lách
        # Đảm bảo scale dương (lực đẩy). Không để âm (âm là hút).
        obstacles = [
            {"position": pos, "sigma": 0.05, "scale": 0.1} for pos in right_team
        ]
        obstacles.extend([{"position": ball, "sigma": 0.25, "scale": 0.1}])

        
        return left_team, goals, obstacles

    def _raycast_from_agent(
        self,
        agent_pos:    NDArray[np.float32],
        left_team:    NDArray[np.float32],
        right_team:   NDArray[np.float32],
        targets:      List[NDArray[np.float32]],   # [near_post, far_post, penalty_spot]
        max_distance: float = 1.0,
    ) -> NDArray[np.float32]:
        """
        Ray-cast từ agent theo 16 hướng, 3 kênh mỗi tia.

        Kênh:
          [0] opponent  — khoảng cách tia chạm cầu thủ đối phương gần nhất
          [1] teammate  — khoảng cách tia chạm đồng đội gần nhất
          [2] target    — khoảng cách tia chạm điểm mục tiêu (near_post / far_post / penalty_spot) gần nhất
        Output: (48,) = 16 tia × 3 kênh, chuẩn hóa [0, 1] (0 = gần, 1 = xa/không có)
        """
        angles: NDArray[np.float32] = np.linspace(0, 2 * np.pi, self.N_RAYS, endpoint=False)
        detection_radius: float = 0.02
        ray_distances: List[float] = []

        for angle in angles:
            direction: NDArray[np.float32] = np.array([np.cos(angle), np.sin(angle)])

            # ── Kênh 0: OPPONENTS ───────────────────────────────────────────
            min_opp: float = max_distance
            for opp in right_team:
                ov: NDArray[np.float32] = opp - agent_pos
                dist: float = float(np.linalg.norm(ov))
                if dist > 0:
                    proj: float = float(np.dot(ov, direction))
                    if proj > 0:
                        perp: float = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                        if perp < detection_radius:
                            min_opp = min(min_opp, dist)

            # ── Kênh 1: TEAMMATES ────────────────────────────────────────────
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

            # ── Kênh 2: TARGET ZONES (near_post, far_post, penalty_spot) ────
            min_target: float = max_distance
            for tgt in targets:
                tv2: NDArray[np.float32] = tgt - agent_pos
                dist = float(np.linalg.norm(tv2))
                if dist > 0:
                    proj = float(np.dot(tv2, direction))
                    if proj > 0:
                        perp = float(np.sqrt(max(0.0, dist**2 - proj**2)))
                        if perp < detection_radius:
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


    # =========================================================================
    # ÁNH XẠ HÀNH ĐỘNG
    # =========================================================================

    def _map_global_actions(
        self,
        agent_actions: NDArray[np.int64],
        raw_obs:       Optional[RawObs] = None,
    ) -> NDArray[np.int64]:
        
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            act = int(agent_actions[i])
            if 0 < act <= 8:
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
        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )
        base_obs  = raw_obs[0]
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])
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
        state, obses, energy_fields = self._get_all_obses_and_state(raw_obs)
        self.last_energy_fields = energy_fields

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
        kicker_id: Optional[int] = 1

        safe_actions: NDArray[np.int64] = self._map_global_actions(actions, raw_obs=self.last_raw_obs)
        raw_obs, rewards, done, info = self.env.step(safe_actions)
        state, obses, energy_fields = self._get_all_obses_and_state(raw_obs)
        base_obs = raw_obs[0]

        # Hằng số chuẩn hóa
        ENERGY_SCALE:    float = 0.05   
        HANDOVER_GAIN:   float = 1.0 

        # 1. R_energy
        rewards_energy: NDArray[np.float32] = (-energy_fields) * ENERGY_SCALE 
        
        # 4. R_possession
        current_ball_owned_team: int = base_obs.get("ball_owned_team", -1)
        current_ball_owned_player: int = base_obs.get("ball_owned_player", -1)

        rewards_handover: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if current_ball_owned_team == 0 and current_ball_owned_player != -1:
            idx = current_ball_owned_player
            original_agent_action = int(actions[idx])
            if original_agent_action > 8 or original_agent_action == 0:    # Actor chọn STAY (action 8-9) → đúng, giữ bóng để chuyền
                rewards_handover[idx] =  1.0
            elif 0 < original_agent_action <= 8:
                rewards_handover[idx] = -0.6
        rewards_handover = rewards_handover * HANDOVER_GAIN

        shaped_rewards: NDArray[np.float32] = rewards_energy + rewards_handover

        # ── Cập nhật buffer và trạng thái cho bước kế tiếp ──────────────────
        self.last_raw_obs = raw_obs
        self.last_energy_fields = energy_fields

        rewards_view: Dict[str, NDArray[np.float32]] = {
            "r_energy":     rewards_energy,
            "r_handover":   rewards_handover,
        }

        return state, obses, shaped_rewards, rewards_view, done