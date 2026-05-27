from __future__ import annotations

import random
from typing import Any, Optional, Tuple

import gym
import numpy as np
from numpy.typing import NDArray

from utils.energy_field import EnergyFieldDefiner, FieldItem

# Kiểu raw observation của GRF: list of dict (1 dict per agent, nhưng shared state)
RawObs = list[dict[str, Any]]


class GFootballGlobalWrapper(gym.Wrapper):
    """
    Wrapper Phase 1 (Global Spatial Hierarchy) của HES-COMA.

    Xử lý:
    - Raycast 8 hướng × 4 object types → obs_dim = 32 + 1 energy = 33.
    - Corner-kick state machine (Positioning → Pass trigger → Physics follow-through).
    - Reward shaping: env reward (normalized) + energy delta + ball possession.
    """

    def __init__(self, env: gym.Env, num_agents: int = 11) -> None:
        super().__init__(env)
        self.num_agents:    int   = num_agents
        self.action_space         = gym.spaces.MultiDiscrete([10] * num_agents)
        self.energy_definer:      EnergyFieldDefiner = EnergyFieldDefiner()
        self.state_dim:     int   = 46   # left(22) + right(22) + ball(2)
        self.obs_dim:       int   = 33   # 32 rays + 1 energy

        self.last_raw_obs:      Optional[RawObs]          = None
        self.last_energy_fields: Optional[NDArray[np.float32]] = None

        self._kicker_id:  Optional[int] = None
        self._has_kicked: bool          = False

    # =========================================================================
    # PHẦN 1: TRÍCH XUẤT VỊ TRÍ
    # =========================================================================

    def _extract_positions(
        self,
        raw_obs: RawObs,
    ) -> Tuple[
        NDArray[np.float32],   # left_team  (11, 2)
        list[FieldItem],       # goals
        list[FieldItem],       # obstacles
    ]:
        """
        Tạo goals và obstacles cho Energy Field.
        Hoàn toàn tách biệt 2 trường hợp để tránh lẫn lộn mục tiêu.
        """
        base_obs   = raw_obs[0]
        left_team:  NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])

        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )

        if is_corner_kick:
            target_sign:  float = 1.0 if ball[0] > 0 else -1.0
            near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  ball[1] * 0.1])
            far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -ball[1] * 0.1])
            penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])

            # THAY ĐỔI: Thu hẹp Sigma, Đào sâu Scale để tạo khoảng trống rõ rệt
            goals = [
                {"position": near_post,    "sigma": 0.15, "scale": -3.0},
                {"position": far_post,     "sigma": 0.20, "scale": -2.0},
                {"position": penalty_spot, "sigma": 0.25, "scale": -2.5},
            ]
            # THAY ĐỔI: Thu hẹp Sigma của hậu vệ để Agent có kẽ hở luồn lách
            obstacles = [
                {"position": pos, "sigma": 0.06, "scale": 1.5} for pos in right_team
            ]
            obstacles.append({"position": ball, "sigma": 0.15, "scale": 2.5})

        else:
            goals = [{"position": ball, "sigma": 0.25, "scale": -2.0}]
            obstacles = [{"position": pos, "sigma": 0.06, "scale": 1.2} for pos in right_team]

        return left_team, goals, obstacles

    # =========================================================================
    # PHẦN 2: RAY-CAST
    # =========================================================================

    def _raycast_from_agent(
        self,
        agent_pos:    NDArray[np.float32],  # (2,)
        left_team:    NDArray[np.float32],  # (11, 2)
        right_team:   NDArray[np.float32],  # (11, 2)
        ball:         NDArray[np.float32],  # (2,)
        goals:        list[FieldItem],
        max_distance: float = 1.0,
    ) -> NDArray[np.float32]:
        """
        Ray-cast từ agent theo 8 hướng, detect closest objects.

        Returns:
            ray_distances: (32,) = 8 rays × 4 object types, normalized [0, 1].
        """
        angles:           NDArray[np.float32] = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        detection_radius: float = 0.025
        ray_distances:    list[float] = []

        for angle in angles:
            direction: NDArray[np.float32] = np.array([np.cos(angle), np.sin(angle)])

            dists: dict[str, float] = {
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
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      list[FieldItem],
        energy_val: float,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Returns:
            obs_vec:  (33,) = ray_info (32,) + energy (1,)
            ray_info: (32,) — raw ray distances (for debug)
        """
        ray_info:          NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, ball, goals, max_distance=2.0
        )
        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)
        return np.concatenate([ray_info, energy_field_info]), ray_info

    def _build_global_state(self, base_obs: dict[str, Any]) -> NDArray[np.float32]:
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

    # =========================================================================
    # PHẦN 5: ÁNH XẠ HÀNH ĐỘNG
    # =========================================================================

    def _map_global_actions(
        self,
        agent_actions: NDArray[np.int64],
        raw_obs:       Optional[RawObs] = None,
    ) -> NDArray[np.int64]:
        """
        Bộ chuyển đổi hành động cho GAgent (tích hợp corner-kick state machine).

        GAgent output [0..9]:
          0-7 → GRF 1-8  (8 hướng di chuyển)
          8-9 → GRF 0    (Idle/Stop)
        Kicker được override bởi state machine phạt góc.
        """
        kicker_id: Optional[int] = self._kicker_id

        # Kiểm tra thoát Phase 3 (bóng đã rời góc)
        if kicker_id is not None and self._has_kicked and raw_obs is not None:
            base_obs = raw_obs[0]
            ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])
            ball_speed: float = (
                float(np.linalg.norm(base_obs["ball"][3:5]))
                if len(base_obs["ball"]) >= 5
                else 0.0
            )
            kicker_pos:           NDArray[np.float32] = np.array(base_obs["left_team"][kicker_id])
            ball_dist_from_kicker: float = float(np.linalg.norm(ball - kicker_pos))

            if ball_speed > 0.1 or ball_dist_from_kicker > 0.05:
                self._kicker_id  = None
                self._has_kicked = False
                kicker_id        = None

        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            if i == kicker_id:
                if not self._has_kicked:
                    # PHASE 2: Trigger duy nhất 1 frame High Pass
                    mapped_actions[i] = random.choice([9, 10])
                    self._has_kicked  = True
                else:
                    # PHASE 3: Idle cho đến khi bóng rời góc
                    mapped_actions[i] = 0
            else:
                act: int = int(agent_actions[i])
                if 0 < act <= 8:
                    mapped_actions[i] = act
                else:
                    mapped_actions[i] = random.choice([0, 14])

        return mapped_actions

    # =========================================================================
    # PHẦN 6: RESET & STEP
    # =========================================================================

    def reset(self, **kwargs: Any) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Returns:
            state: (46,)
            obses: (11, 33)
        """
        raw_obs: RawObs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs

        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )
        self.last_energy_fields = energy_fields

        base_obs   = raw_obs[0]
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])
        state:      NDArray[np.float32] = self._build_global_state(base_obs)

        results = [
            self._process_single_obs(left_team[i], left_team, right_team, ball, goals, energy_fields[i])
            for i in range(self.num_agents)
        ]
        obses: NDArray[np.float32] = np.array([r[0] for r in results])

        return state, obses

    def step(
        self,
        actions: NDArray[np.int64],
    ) -> Tuple[
        NDArray[np.float32],           # state          (46,)
        NDArray[np.float32],           # obses          (11, 33)
        NDArray[np.float32],           # shaped_rewards (11,)
        bool,                          # done
    ]:
        kicker_id: Optional[int] = self._get_corner_kicker_id(self.last_raw_obs)
        if kicker_id is not None and not self._has_kicked:
            self._kicker_id = kicker_id

        safe_actions: NDArray[np.int64] = self._map_global_actions(actions, raw_obs=self.last_raw_obs)
        raw_obs, rewards, done, info = self.env.step(safe_actions)

        left_team, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )

        # ── Reward shaping ──────────────────────────────────────────────────
        ENERGY_SCALE:      float = 0.1

        rewards_energy:     NDArray[np.float32] = (self.last_energy_fields - energy_fields) * ENERGY_SCALE  # type: ignore[operator]
        shaped_rewards: NDArray[np.float32] =  rewards_energy

        # Cập nhật buffer cho bước kế tiếp
        self.last_raw_obs       = raw_obs
        self.last_energy_fields = energy_fields

        base_obs   = raw_obs[0]
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])
        state:      NDArray[np.float32] = self._build_global_state(base_obs)

        results = [
            self._process_single_obs(left_team[i], left_team, right_team, ball, goals, energy_fields[i])
            for i in range(self.num_agents)
        ]
        obses: NDArray[np.float32] = np.array([r[0] for r in results])

        return state, obses, shaped_rewards, done