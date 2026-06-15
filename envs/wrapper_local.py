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
    - GAgent (frozen từ Phase 1) quyết định di chuyển (1-8) hoặc STOP (0, 9).
    - Chỉ khi GAgent chọn STOP, LAgent mới thực thi tactical action.
    - Energy field CHỈ dùng ở global hierarchy → obs_l KHÔNG chứa energy.
    - obs_g = 48 rays + energy(1) + ball_owned(1) + sticky(10) + role(10) = 70 — mirror Phase 1.
    - obs_l = 48 rays (ray-info thuần, 16 tia × 3 kênh, không có energy).
    """

    # ── Hằng số hành động ──────────────────────────────────────────────────────
    # GAgent output [0..9]: 1-8 → di chuyển (GRF 1-8), 0 9 → STOP (GRF 0, 14)
    STOP_ACTIONS: frozenset[int] = frozenset({0, 9})

    # LAgent output [0..n_tactic_actions-1] → GRF tactical action
    TACTIC_MAP: list[int] = [9, 10, 11, 12]
    def __init__(self, env: gym.Env, num_agents: int = 11) -> None:
        super().__init__(env)
        self.num_agents:       int = num_agents
        
        self.n_tactic_actions: int = 4

        # ── Không gian quan sát ───────────────────────────────────────────────
        self.state_dim:  int = 46  # left(22) + right(22) + ball(2)
        # obs_dim_g đồng bộ Phase 1: rays(48) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 70
        # rays = 16 tia × 3 kênh (opponent, teammate, target_zone) — mirror wrapper_global
        self.N_RAYS:    int = 16
        self.obs_dim_g: int = 70
        self.obs_dim_l: int = 51  # Ray-info(48) + ball_owned(1) + ball_rel_pos(2) — KHÔNG có energy

        # Energy field vẫn cần để xây dựng obs_g
        self.energy_definer: EnergyFieldDefiner = EnergyFieldDefiner()

        # ── State machine phạt góc (mirror Global) ───────────────────────────
        self.last_raw_obs: Optional[RawObs] = None
        self._kicker_id:   Optional[int]    = 1
        # [FIX STICKY] Lưu tactic đang chờ được thực thi sau khi direction đã được set
        self._pending_kicker_tactic: Optional[int] = None  # GRF action id cần gửi ở step kế

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
        - Anticipatory Attractors (ball_direction)
        - Directional Repulsors (right_team_direction)
        Trả về thêm right_team và ball để tái sử dụng ở các hàm khác.
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

    # =========================================================================
    # PHẦN 2: RAY-CAST (cho obs_g và obs_l — mirror wrapper_global)
    # =========================================================================

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
        detection_radius: float               = 0.02  # Mirror wrapper_global
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
        obs_g: Mirror wrapper_global._process_single_obs (70 chiều).
        rays(48) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 70.
        Giữ nguyên chuẩn Phase 1 để load model đúng shape.
        """
        # Tách target positions từ goals (near_post, far_post, penalty_spot)
        target_positions: list[NDArray[np.float32]] = [
            np.array(g["position"] if isinstance(g, dict) else g, dtype=np.float32)
            for g in goals
        ]
        if not target_positions:
            target_sign = 1.0 if ball[0] > 0 else -1.0
            target_positions = [np.array([target_sign * 0.9, 0.0], dtype=np.float32)]

        ray_info: NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, target_positions
        )
        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)

        # Cờ sở hữu bóng (Boolean flag)
        ball_owned_team   = base_obs.get("ball_owned_team", -1)
        ball_owned_player = base_obs.get("ball_owned_player", -1)
        is_ball_owned     = 1.0 if (ball_owned_team == 0 and ball_owned_player == agent_idx) else 0.0
        is_ball_owned_arr: NDArray[np.float32] = np.array([is_ball_owned], dtype=np.float32)

        # Trạng thái nút bấm (Sticky actions)
        sticky_actions_raw = raw_obs_i.get("sticky_actions", [0] * 10)
        sticky_actions: NDArray[np.float32] = np.array(sticky_actions_raw, dtype=np.float32)

        # Vị trí sở trường (Left team roles — one-hot)
        roles_raw = base_obs.get("left_team_roles", [0] * 11)
        role_idx  = roles_raw[agent_idx] if agent_idx < len(roles_raw) else 0
        role_onehot: NDArray[np.float32] = np.zeros(10, dtype=np.float32)
        if 0 <= role_idx < 10:
            role_onehot[role_idx] = 1.0

        return np.concatenate([
            ray_info,           # 48  (16 tia × 3 kênh)
            energy_field_info,  #  1
            is_ball_owned_arr,  #  1
            sticky_actions,     # 10
            role_onehot,        # 10
        ])  # (70,)

    def _process_local_obs(
        self,
        agent_idx:  int,
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      list[FieldItem],
        base_obs:   dict,
    ) -> NDArray[np.float32]:
        """
        obs_l: Ray-info(48) + ball_owned(1) + ball_rel_pos(2) = 51 chiều.

        [FIX #1] Thêm ball_owned_flag và ball_rel_pos để LAgent biết:
          - Nó có đang cầm bóng không.
          - Bóng đang ở đâu so với vị trí hiện tại.
        Không có energy field (chỉ dùng cho GAgent).

        Returns:
            (51,) = ray_info(48) + ball_owned(1) + ball_rel_pos(2).
        """
        target_positions: list[NDArray[np.float32]] = [
            np.array(g["position"] if isinstance(g, dict) else g, dtype=np.float32)
            for g in goals
        ]
        if not target_positions:
            target_sign = 1.0 if ball[0] > 0 else -1.0
            target_positions = [np.array([target_sign * 0.9, 0.0], dtype=np.float32)]

        ray_info: NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, target_positions
        )

        # [FIX #1] Cờ sở hữu bóng — LAgent cần biết mình có bóng để ra lệnh pass/shot
        ball_owned_team   = base_obs.get("ball_owned_team", -1)
        ball_owned_player = base_obs.get("ball_owned_player", -1)
        ball_owned_flag: NDArray[np.float32] = np.array(
            [1.0 if (ball_owned_team == 0 and ball_owned_player == agent_idx) else 0.0],
            dtype=np.float32
        )

        # [FIX #1] Vị trí bóng tương đối (chuẩn hóa về [-1, 1])
        ball_rel: NDArray[np.float32] = np.clip(
            (ball - agent_pos) / 0.5, -1.0, 1.0
        ).astype(np.float32)

        return np.concatenate([ray_info, ball_owned_flag, ball_rel])  # (51,)

    def _build_global_state(self, base_obs: dict[str, Any]) -> NDArray[np.float32]:
        """Trả về state toàn cục (46,) = left(22) + right(22) + ball(2)."""
        return np.concatenate([
            base_obs["left_team"].flatten(),
            base_obs["right_team"].flatten(),
            base_obs["ball"][:2],
        ])

    # =========================================================================
    # PHẦN 5: ÁNH XẠ HÀNH ĐỘNG
    # =========================================================================

    def _map_global_actions(
        self,
        agent_actions: NDArray[np.int64],
        raw_obs:       Optional[RawObs] = None,
    ) -> NDArray[np.int64]:
        
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)

        for i in range(self.num_agents):
            act = agent_actions[i]
            if 0 < act <= 8:
                mapped_actions[i] = act   # GRF action 1-8 (8 hướng di chuyển)
            elif act == 0:
                mapped_actions[i] = 0
            else:
                mapped_actions[i] = 14     # GRF 14: release direction → đứng im
        return mapped_actions

    # =========================================================================
    # PHẦN 6: HÀM TIỆN ÍCH CHUNG
    # =========================================================================

    def _get_all_obses_and_state(
        self, raw_obs: RawObs
    ) -> Tuple[
        NDArray[np.float32],  # state  (46,)
        NDArray[np.float32],  # obs_g  (11, 70)
        NDArray[np.float32],  # obs_l  (11, 51)
    ]:
        """
        Trích xuất (state, obs_g, obs_l) từ raw_obs.
          obs_g: rays(48) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 70 — Global hierarchy
          obs_l: ray_info(48) + ball_owned(1) + ball_rel_pos(2) = 51 — Local hierarchy [FIX #1]
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
        ])  # (11, 70)

        obs_l: NDArray[np.float32] = np.array([
            self._process_local_obs(i, left_team[i], left_team, right_team, ball, goals, base_obs)
            for i in range(self.num_agents)
        ])  # (11, 51)

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
        self._kicker_id   = 1
        self._pending_kicker_tactic = None   # [FIX STICKY] Reset pending tactic
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
        # ── Hằng số reward
        ENV_SCALE:         float = 1.0 / self.num_agents
        ASSIST_REWARD:     float =  1.0    # Kiến tạo dẫn đến bàn thắng
        BALL_APPROACH_R:   float =  0.3    # Chạm cắt bóng thành công
        # ── Role 1: Kicker (player 1) → CHỈ high_pass ──────────────────────────
        # high_pass đưa bóng vào vòng cấm, các cầu thủ khác sẽ dứt điểm
        KICKER_HIGH_PASS_R:  float =  0.5   # Thưởng khi kicker chọn high_pass
        KICKER_WRONG_TACTIC_P: float = -1.0  # Phạt mạnh khi kicker chọn tạc tiᲠ khác
        # Bonus khi bóng đến tay người đăng trong vùng nguy hiểm
        PASSING_REWARD:      float =  0.5   # Kicker tung bóng vào vùng nguy hiểm
        BACKPASS_PENALTY:    float = -0.8
        OUTSIDE_BOX_PENALTY: float = -0.3
        DANGER_ZONE_X:       float =  0.75
        DANGER_ZONE_Y:       float =  0.35
        # ── Role 2: Cầu thủ trong vòng cấm khi nhận bóng → DỨT ĐIỂM ────────────
        BOX_X_THRESHOLD:   float =  0.83
        BOX_Y_THRESHOLD:   float =  0.20
        SHOT_IN_BOX_R:     float =  4.0    # Thưởng rất lọn khi dứt điểm trong vòng cấm
        PASS_IN_BOX_P:     float = -2.0    # Phạt nặng khi bỏ qua cơ hội sút
        # Role-based (positioning bonus khi góp phần vào bàn thắng)
        ROLE_NEAR_POST:    float =  1.0
        ROLE_FAR_POST:     float =  1.0
        ROLE_PENALTY_SPOT: float =  1.5
        # ── Dense positioning & anticipate ──────────────────────────────────────
        ANTICIPATE_R:        float =  0.12

        # ── Đọc trạng thái trước bước ─────────────────────────────────────────
        last_obs  = self.last_raw_obs[0] if self.last_raw_obs is not None else None
        last_ball_owned:   int = last_obs["ball_owned_team"]           if last_obs else -1
        last_ball_owned_player: int = last_obs.get("ball_owned_player", -1) if last_obs else -1
        last_score: list[int]  = list(last_obs["score"])               if last_obs else [0, 0]

        # ── Corner-kick state machine (mirror _map_global_actions) ───────────
        kicker_id: Optional[int] = 1
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        ball_holder: int = last_obs.get("ball_owned_player", -1) if last_obs else -1
        ball_team:   int = last_obs.get("ball_owned_team",   -1) if last_obs else -1
        # Chỉ xét ball_holder hợp lệ thuộc đội ta
        valid_ball_holder: bool = (ball_team == 0 and ball_holder != -1)

        # [FIX STICKY] Helper: tính GRF direction action (1–8) từ vector hướng
        def _direction_from_vec(dx: float, dy: float) -> int:
            """GRF Action Map: 1=left, 2=top_left, 3=top, 4=top_right,
                               5=right, 6=bottom_right, 7=bottom, 8=bottom_left.
               (Note: +x is right, +y is bottom in GRF pitch coords)"""
            angle = float(np.degrees(np.arctan2(dy, dx))) % 360
            # Right is 0 deg. Bottom is 90 deg. Left is 180 deg. Top is 270 deg.
            if angle < 22.5 or angle >= 337.5:
                return 5  # right
            elif angle < 67.5:
                return 6  # bottom_right
            elif angle < 112.5:
                return 7  # bottom
            elif angle < 157.5:
                return 8  # bottom_left
            elif angle < 202.5:
                return 1  # left
            elif angle < 247.5:
                return 2  # top_left
            elif angle < 292.5:
                return 3  # top
            else:
                return 4  # top_right

        for i in range(self.num_agents):
            if active_mask[i] and valid_ball_holder and i == ball_holder:
                # [ROLE 1] Kicker luôn được ép dùng high_pass (GRF 10)
                # Ignore lựa chọn của LAgent để đảm bảo kết cấu chiến thuật
                grf_tactic = self.TACTIC_MAP[1]   # Index 1 = high_pass (GRF 10)

                # [FIX STICKY] Kiểm tra sticky direction của kicker
                sticky_raw = (self.last_raw_obs[i].get("sticky_actions", [0] * 10)
                              if self.last_raw_obs is not None else [0] * 10)
                has_direction: bool = any(int(sticky_raw[d]) for d in range(8))

                if self._pending_kicker_tactic is not None:
                    # Step 2: đã set direction bước trước → gửi tactic thực sự
                    mapped_actions[i]           = self._pending_kicker_tactic
                    self._pending_kicker_tactic = None
                elif not has_direction:
                    # Step 1: chưa có sticky direction → setup hướng về penalty_spot trước
                    if last_obs is not None:
                        _b      = np.array(last_obs["ball"][:2], dtype=np.float32)
                        _sign   = 1.0 if float(_b[0]) > 0 else -1.0
                        _kpos   = np.array(last_obs["left_team"][i], dtype=np.float32)
                        _target = np.array([_sign * 0.8, 0.0], dtype=np.float32)
                        _vec    = _target - _kpos
                        mapped_actions[i] = _direction_from_vec(float(_vec[0]), float(_vec[1]))
                    else:
                        mapped_actions[i] = 7   # Fallback: right
                    # Lưu tactic để gửi ở step tiếp theo
                    self._pending_kicker_tactic = grf_tactic
                else:
                    # Đã có sticky direction → gửi tactic trực tiếp
                    mapped_actions[i]           = grf_tactic
                    self._pending_kicker_tactic = None
            elif active_mask[i] and valid_ball_holder and i != ball_holder:
                # [ROLE 2] Cầu thủ khác có bóng khi GAgent STOP → LAgent quyết định
                # (ví dụ: nhận bóng từ kicker và đang trong vòng cấm)
                tactic_idx        = int(local_actions[i]) % self.n_tactic_actions
                mapped_actions[i] = self.TACTIC_MAP[tactic_idx]
            elif active_mask[i]:
                # Agent STOP nhưng không có bóng → đứng giữ vị trí
                mapped_actions[i] = 0
            else:
                act = global_actions[i]
                if 0 < act <= 8:
                    mapped_actions[i] = act
                elif act == 0:
                    mapped_actions[i] = 0
                else:
                    mapped_actions[i] = 14

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

        # ── R_passing: [ROLE 1] Kicker phải chọn HIGH_PASS ──────────────────────
        # TACTIC_MAP = [9(long), 10(high), 11(short), 12(shot)]
        # Kicker được ép dùng high_pass trong action mapping,
        # nhưng vẫn thưởng/phạt theo lựa chọn của LAgent để gradient học được signal.
        rewards_passing: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)

        kicker_just_got_ball: bool = (
            kicker_id is not None
            and last_ball_owned_player != kicker_id
            and last_ball_owned_player == -1
            and active_mask[kicker_id]
        )
        if kicker_just_got_ball:
            kicker_tactic = int(local_actions[kicker_id]) % self.n_tactic_actions
            if kicker_tactic == 1:   # high_pass → ĐÚNG
                rewards_passing[kicker_id] += KICKER_HIGH_PASS_R
                shaped_rewards[kicker_id]  += KICKER_HIGH_PASS_R
            else:                    # long/short/shot → SAI vai trò
                rewards_passing[kicker_id] += KICKER_WRONG_TACTIC_P
                shaped_rewards[kicker_id]  += KICKER_WRONG_TACTIC_P

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

        # ── R_in_box: [ROLE 2] Cầu thủ trong vòng cấm PHẢI DỨT ĐIỂM ─────────────
        # Chỉ áp dụng cho cầu thủ không phải kicker (kicker ở corner, không trong vòng cấm)
        last_left_team: NDArray[np.float32] = np.array(last_obs["left_team"])
        rewards_in_box: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if last_obs is not None:
            last_ball_owned_check: int = last_obs.get("ball_owned_team", -1)
            last_ball_pos: NDArray[np.float32] = np.array(last_obs["ball"][:2])
            attack_sign: float = 1.0 if last_ball_pos[0] > 0 else -1.0
            if last_ball_owned_check == 0 and last_ball_owned_player != -1:
                idx = last_ball_owned_player
                # Kicker (ở corner) không áp dụng rule này
                if idx != kicker_id:
                    player_x: float = float(last_left_team[idx][0])
                    player_y: float = float(last_left_team[idx][1])
                    in_box: bool = (
                        attack_sign * player_x > BOX_X_THRESHOLD
                        and abs(player_y) < BOX_Y_THRESHOLD
                    )
                    if in_box and active_mask[idx]:
                        tactic_chosen: int = int(local_actions[idx]) % self.n_tactic_actions
                        if tactic_chosen == 3:   # Shot → ĐÚNG vai trò
                            rewards_in_box[idx]  = SHOT_IN_BOX_R
                            shaped_rewards[idx] += SHOT_IN_BOX_R
                        else:                    # Pass/idle trong vòng cấm → SAI
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
                elif i != kicker_id:
                    # [FIX #3] Thay ball_owned==-1 bằng ball_z>0.1 (bóng đang bay)
                    # corner kick: ball_owned thường là 0 (player 1 giữ bóng)
                    # nên điều kiện cũ hầu như không kích hoạt
                    _ball_z = float(current_obs.get('ball', [0, 0, 0])[2])
                    ball_is_airborne: bool = _ball_z > 0.1
                    if ball_is_airborne:
                        DENSE_POS_R: float = 0.08   # Tăng từ 0.05 → 0.08
                        if float(np.linalg.norm(pos - near_post_zone)) < 0.15:
                            rewards_role[i] += DENSE_POS_R
                            shaped_rewards[i] += DENSE_POS_R
                        elif float(np.linalg.norm(pos - far_post_zone)) < 0.15:
                            rewards_role[i] += DENSE_POS_R
                            shaped_rewards[i] += DENSE_POS_R
                        elif float(np.linalg.norm(pos - penalty_spot_zone)) < 0.20:
                            rewards_role[i] += DENSE_POS_R * 1.5
                            shaped_rewards[i] += DENSE_POS_R * 1.5

        # ── R_ball_approach: Thưởng Sparse khi chạm cắt bóng thành công ─────
        rewards_approach: NDArray[np.float32] = np.zeros(self.num_agents, dtype=np.float32)
        if current_ball_owned == 0 and current_ball_owned_player != -1:
            if last_ball_owned != 0 or last_ball_owned_player != current_ball_owned_player:
                # Chỉ thưởng 1 lần khi giành được quyền kiểm soát bóng
                rewards_approach[current_ball_owned_player] = BALL_APPROACH_R
                shaped_rewards[current_ball_owned_player] += BALL_APPROACH_R

        # ── R_anticipate: Thưởng khi agent tiến gần điểm rơi dự đoán của bóng ─
        # [FIX #4] Khuyến khích chạy đón bóng thay vì đứng yên khi bóng bay
        _ball_arr = np.array(current_obs.get('ball', [0, 0, 0]), dtype=np.float32)
        _ball_dir = np.array(current_obs.get('ball_direction', [0, 0, 0]), dtype=np.float32)
        _ball_z_now = float(_ball_arr[2])
        if _ball_z_now > 0.1:   # Bóng đang bay
            # Ước tính điểm rơi tuyến tính (10 bước tới)
            est_landing: NDArray[np.float32] = _ball_arr[:2] + _ball_dir[:2] * 10
            for i in range(self.num_agents):
                if i == kicker_id:
                    continue
                dist_to_landing = float(np.linalg.norm(left_team[i] - est_landing))
                if dist_to_landing < 0.2:   # Agent gần điểm rơi dự kiến
                    rewards_approach[i] += ANTICIPATE_R
                    shaped_rewards[i]   += ANTICIPATE_R

        self.last_raw_obs = raw_obs
        state, obs_g, obs_l = self._get_all_obses_and_state(raw_obs)

        # ── Tổng hợp các phần thưởng thành phần (sum trên tất cả agent) ──────
        reward_info: dict = {
            'R_env':        float(np.sum(rewards_env)),
            'R_passing':    float(np.sum(rewards_passing)),
            'R_in_box':     float(np.sum(rewards_in_box)),
            'R_assist':     float(np.sum(rewards_assist)),
            'R_role':       float(np.sum(rewards_role)),
            'R_approach':   float(np.sum(rewards_approach)),
        }

        return state, obs_g, obs_l, shaped_rewards, done, reward_info