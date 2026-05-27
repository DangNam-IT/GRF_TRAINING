from __future__ import annotations

import random
from typing import Any, Optional, Tuple

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
    - GAgent (frozen từ Phase 1) quyết định di chuyển (0-7) hoặc STOP (8/9).
    - Chỉ khi GAgent chọn STOP, LAgent mới thực thi tactical action.
    - Energy field CHỈ dùng ở global hierarchy → obs_l KHÔNG chứa energy.
    - obs_g = 32 rays + 1 energy = 33 chiều (giữ nguyên chuẩn Phase 1).
    - obs_l = 32 rays (ray-info thuần, không có energy).
    """

    # ── Hằng số hành động ──────────────────────────────────────────────────────
    # GAgent output [0..9]: 0-7 → di chuyển (GRF 1-8), 8-9 → STOP (GRF 0)
    STOP_ACTIONS: frozenset[int] = frozenset({8, 9})

    # LAgent output [0..n_tactic_actions-1] → GRF tactical action
    # 0: Short Pass (11), 1: High Pass (12)
    TACTIC_MAP: list[int] = [11, 12]

    def __init__(self, env: gym.Env, num_agents: int = 11) -> None:
        super().__init__(env)
        self.num_agents:       int = num_agents
        self.n_tactic_actions: int = 2

        # ── Không gian quan sát ───────────────────────────────────────────────
        self.state_dim:  int = 46  # left(22) + right(22) + ball(2)
        self.obs_dim_g:  int = 33  # 32 rays + 1 energy  (đồng bộ Phase 1)
        self.obs_dim_l:  int = 32  # Ray-info thuần — KHÔNG có energy

        # Energy field vẫn cần để xây dựng obs_g
        self.energy_definer: EnergyFieldDefiner = EnergyFieldDefiner()

        # ── State machine phạt góc (mirror Global) ───────────────────────────
        self.last_raw_obs: Optional[RawObs] = None
        self._kicker_id:   Optional[int]    = None
        self._has_kicked:  bool             = False

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
        Tạo goals/obstacles cho Energy Field — giống hệt wrapper_global.
        Trả về thêm right_team và ball để tái sử dụng ở các hàm khác.
        """
        base_obs   = raw_obs[0]
        left_team:  NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])

        is_corner_kick: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )

        if is_corner_kick:
            target_sign:  float                = 1.0 if ball[0] > 0 else -1.0
            near_post:    NDArray[np.float32]  = np.array([target_sign * 0.9,  ball[1] * 0.1])
            far_post:     NDArray[np.float32]  = np.array([target_sign * 0.9, -ball[1] * 0.1])
            penalty_spot: NDArray[np.float32]  = np.array([target_sign * 0.8,  0.0])
            goals: list[FieldItem] = [
                {"position": near_post,    "sigma": 0.35, "scale": -1.5},
                {"position": far_post,     "sigma": 0.35, "scale": -1.0},
                {"position": penalty_spot, "sigma": 0.40, "scale": -1.2},
            ]
            obstacles: list[FieldItem] = [
                {"position": pos, "sigma": 0.18, "scale": 1.2} for pos in right_team
            ]
            obstacles.append({"position": ball, "sigma": 0.20, "scale": 1.8})
        else:
            goals = [{"position": ball, "sigma": 0.40, "scale": -1.0}]
            obstacles = [{"position": pos, "sigma": 0.18, "scale": 1.0} for pos in right_team]

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
        max_distance: float = 2.0,
    ) -> NDArray[np.float32]:
        """8 hướng × 4 object types → (32,) normalized [0, 1]."""
        angles:           NDArray[np.float32] = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        detection_radius: float               = 0.15
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
        agent_pos:  NDArray[np.float32],
        left_team:  NDArray[np.float32],
        right_team: NDArray[np.float32],
        ball:       NDArray[np.float32],
        goals:      list[FieldItem],
        energy_val: float,
    ) -> NDArray[np.float32]:
        """
        obs_g: 32 rays + 1 energy = 33 chiều.
        Giữ nguyên chuẩn Phase 1 để load model đúng shape.
        """
        ray_info:          NDArray[np.float32] = self._raycast_from_agent(
            agent_pos, left_team, right_team, ball, goals
        )
        energy_field_info: NDArray[np.float32] = np.array([energy_val], dtype=np.float32)
        return np.concatenate([ray_info, energy_field_info])  # (33,)

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
        base_obs  = raw_obs[0]
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        is_corner: bool = (
            abs(abs(ball[0]) - 1.0) < 0.05 and abs(abs(ball[1]) - 0.42) < 0.05
        )
        if is_corner:
            left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
            return int(np.argmin(np.sum((left_team - ball) ** 2, axis=1)))
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
        Bộ lọc hành động GAgent (giữ corner-kick state machine từ Phase 1).
        GAgent output [0..9]:
          0-7 → GRF 1-8  (di chuyển)
          8-9 → GRF 0    (idle/stop)
        """
        kicker_id: Optional[int] = self._kicker_id

        # Kiểm tra thoát Phase 3 (bóng đã rời góc)
        if kicker_id is not None and self._has_kicked and raw_obs is not None:
            base_obs              = raw_obs[0]
            ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
            ball_speed: float = (
                float(np.linalg.norm(base_obs["ball"][3:5]))
                if len(base_obs["ball"]) >= 5
                else 0.0
            )
            kicker_pos:            NDArray[np.float32] = np.array(base_obs["left_team"][kicker_id])
            ball_dist_from_kicker: float               = float(np.linalg.norm(ball - kicker_pos))
            if ball_speed > 0.1 or ball_dist_from_kicker > 0.05:
                self._kicker_id  = None
                self._has_kicked = False
                kicker_id        = None

        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            if i == kicker_id:
                if not self._has_kicked:
                    mapped_actions[i] = random.choice([9, 10])  # High Pass
                    self._has_kicked  = True
                else:
                    mapped_actions[i] = 0  # Idle
            else:
                act: int = int(agent_actions[i])
                if 0 < act <= 8:
                    mapped_actions[i] = act
                else:
                    mapped_actions[i] = 0     # STOP → idle

        return mapped_actions

    # =========================================================================
    # PHẦN 6: HÀM TIỆN ÍCH CHUNG
    # =========================================================================

    def _get_all_obses_and_state(
        self, raw_obs: RawObs
    ) -> Tuple[
        NDArray[np.float32],  # state  (46,)
        NDArray[np.float32],  # obs_g  (11, 33)
        NDArray[np.float32],  # obs_l  (11, 32)
    ]:
        """
        Trích xuất (state, obs_g, obs_l) từ raw_obs.
          obs_g: 32 rays + 1 energy = 33  — Global hierarchy
          obs_l: 32 rays (không energy)   — Local hierarchy
        """
        left_team, right_team, ball, goals, obstacles = self._extract_positions(raw_obs)
        energy_fields: NDArray[np.float32] = self.energy_definer.calculate_field_for_agents(
            left_team, goals, obstacles
        )

        base_obs = raw_obs[0]
        state: NDArray[np.float32] = self._build_global_state(base_obs)

        obs_g: NDArray[np.float32] = np.array([
            self._process_global_obs(left_team[i], left_team, right_team, ball, goals, energy_fields[i])
            for i in range(self.num_agents)
        ])  # (11, 33)

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
            obs_g:  (11, 33)
            obs_l:  (11, 32)
        """
        raw_obs: RawObs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        self._kicker_id   = None
        self._has_kicked  = False
        return self._get_all_obses_and_state(raw_obs)

    def step_global(
        self,
        global_actions: NDArray[np.int64],
    ) -> Tuple[
        NDArray[np.float32],  # state  (46,)
        NDArray[np.float32],  # obs_g  (11, 33)
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
        NDArray[np.float32],  # obs_g   (11, 33)
        NDArray[np.float32],  # obs_l   (11, 32)
        NDArray[np.float32],  # rewards (11,)
        bool,                 # done
    ]:
        """
        Bước khi ít nhất một GAgent chọn STOP.

        Với agent i:
          - active_mask[i] == True  → TACTIC_MAP[local_actions[i] % n]
          - active_mask[i] == False → global_actions[i] + 1 (di chuyển tiếp)
        """
        kicker_id: Optional[int] = self._kicker_id

        # Kiểm tra thoát Phase 3 (bóng đã rời góc)
        if kicker_id is not None and self._has_kicked and self.last_raw_obs is not None:
            base_obs = self.last_raw_obs[0]
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
            if active_mask[i]:
                tactic_idx:        int = int(local_actions[i]) % self.n_tactic_actions
                mapped_actions[i]      = self.TACTIC_MAP[tactic_idx]
            else:
                act: int = int(global_actions[i])
                if i == kicker_id:
                    if not self._has_kicked:
                        # PHASE 2: Trigger duy nhất 1 frame High Pass
                        mapped_actions[i] = random.choice([9, 10])
                        self._has_kicked  = True
                    else:
                        # PHASE 3: Idle cho đến khi bóng rời góc
                        mapped_actions[i] = 0
                else:
                    if 0 < act <= 8:
                        mapped_actions[i] = act
                    else:
                        mapped_actions[i] = random.choice([0, 14])

        raw_obs, rewards, done, info = self.env.step(mapped_actions)

         # Ball possession change
        current_ball_owned:  int = raw_obs[0]["ball_owned_team"]
        last_ball_owned:     int = (
            self.last_raw_obs[0]["ball_owned_team"] if self.last_raw_obs is not None else -1
        )
        BALL_OWNED_REWARD: float = 0.1

        if current_ball_owned == 1 and last_ball_owned != 1:
            ball_possess: NDArray[np.float32] = np.full(self.num_agents, -BALL_OWNED_REWARD / self.num_agents, dtype=np.float32)
            rewards     += ball_possess
        elif current_ball_owned == 0 and last_ball_owned != 0:
            ball_possess = np.full(self.num_agents, BALL_OWNED_REWARD / self.num_agents, dtype=np.float32)
            rewards     += ball_possess
        
        self.last_raw_obs = raw_obs
        
        state, obs_g, obs_l = self._get_all_obses_and_state(raw_obs)
        return state, obs_g, obs_l, np.array(rewards), done