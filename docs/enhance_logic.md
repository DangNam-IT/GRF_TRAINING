```python
with open("wrapper_local.py", "r", encoding="utf-8") as f:
    print("--- wrapper_local.py ---")
    print(f.read()[:2000])

with open("energy_field.py", "r", encoding="utf-8") as f:
    print("\n--- energy_field.py ---")
    print(f.read()[:2000])


```

```text
--- wrapper_local.py ---
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
        # obs_dim_g đồng bộ Phase 1: rays(32) + energy(1) + ball_owned(1) + sticky(10) + role(10) = 54
        self.obs_dim_g:  int = 54
        self.obs_dim_l:  int = 32  # Ray-info thuần — KHÔNG có energy

        # Energy field vẫn cần để xây dựng obs_g
        self.energy_definer: EnergyFieldDefiner = EnergyFieldDefiner()

        # ── State machine phạt góc (mirror Global) ───────────────────────────
        self.last_raw_obs: Optional[RawObs] = None
        self._kicker_id:   Optional[int]    = None
        self._has_kicked:  bool

--- energy_field.py ---
from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

# Một item goal/obstacle có thể là:
#   - dict: {'position': [...], 'sigma': float, 'scale': float}
#   - ndarray/list: chỉ vị trí, dùng tham số mặc định
FieldItem = Union[dict[str, Any], NDArray[np.float32]]


class EnergyFieldDefiner:
    """Tính toán trường năng lượng (Energy Field) cho các tác tử trên sân."""

    def __init__(
        self,
        sigma_attract:  float = 0.9,
        sigma_repel:    float = 0.3,
        scale_attract:  float = -0.9,
        scale_repel:    float = -0.5,
    ) -> None:
        self.sigma_attract:  float = sigma_attract
        self.sigma_repel:    float = sigma_repel
        self.scale_attract:  float = scale_attract
        self.scale_repel:    float = scale_repel

    def _gaussian_kernel(
        self,
        pos:    NDArray[np.float32],
        target: NDArray[np.float32],
        sigma:  float,
        scale:  float,
    ) -> NDArray[np.float32]:
        """Tính giá trị phân phối Gaussian cho một điểm ảnh hưởng."""
        distance_sq: NDArray[np.float32] = np.sum((pos - target) ** 2, axis=-1)
        return scale * np.exp(-distance_sq / (sigma ** 2))  # type: ignore[return-value]

    def _extract_position_and_params(
        self,
        item:          FieldItem,
        default_sigma: float,
        default_scale: float,
    ) -> tuple[NDArray[np.float32], float, float]:
        """
        Trích xuất vị trí và tham số từ item.

        - Nếu item là dict: {'position': [...], 'sigma': ..., 'scale': ...}
        - Nếu item là ndarray/list: chỉ vị trí, dùng tham số mặc định.
        """
        if isinstance(item, dict):
            position: NDArray[np.float32] = np.array(item["position"])
            sigma: float = item.get("sigma", default_sigma)
            scale: float = item.get("scale", default_scale)
        else:
            position = np.array(item)
            sigma = default_sigma
         


```

```python
with open("wrapper_local.py", "r", encoding="utf-8") as f:
    text = f.read()
print(text[2000:5000])
if len(text) > 5000:
    print(text[5000:])


```

```text
             = False

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
        """
        base_obs   = raw_obs[0]
        left_team:  NDArray[np.float32] = np.array(base_obs["left_team"])
        right_team: NDArray[np.float32] = np.array(base_obs["right_team"])
        ball:       NDArray[np.float32] = np.array(base_obs["ball"][:2])

        # Dynamic Energy Fields: dự đoán quỹ đạo bóng và hướng chạy hậu vệ
        ball_direction: NDArray[np.float32] = np.array(
            base_obs["ball"][3:5]) if len(base_obs["ball"]) >= 5 else np.zeros(2, dtype=np.float32)
        right_team_direction_raw = base_obs.get("right_team_direction", [[0.0, 0.0]] * len(right_team))
        right_team_direction: NDArray[np.float32] = np.array(right_team_direction_raw, dtype=np.float32)

        # Thống nhất: sử dụng game_mode == 4 để nhận diện phạt góc
        is_corner_kick: bool = (base_obs.get("game_mode", 0) == 4)

        if is_corner_kick:
            target_sign: float = 1.0 if ball[0] > 0 else -1.0

            # Anticipatory Attractors: dịch chuyển mục tiêu theo quỹ đạo bóng dự kiến
            k_ball: float = 5.0
            predicted_ball_pos: NDArray[np.float32] = ball + ball_direction * k_ball

            near_post:    NDArray[np.float32] = np.array([target_sign * 0.9,  predicted_ball_pos[1] * 0.1])
            far_post:     NDArray[np.float32] = np.array([target_sign * 0.9, -predicted_ball_pos[1] * 0.1])
            penalty_spot: NDArray[np.float32] = np.array([target_sign * 0.8,  0.0])

            goals = [
                {"position": near_post,    "sigma": 0.15, "scale": -3.0},
                {"position": far_post,     "sigma": 0.20, "scale": -2.0},
                {"position": penalty_spot, "sigma": 0.25, "scale": -2.5},
            ]

            # Directional Repulsors: đẩy tâm chướng ngại vật theo hướng chạy hậu vệ
            k_opp: float = 3.0
            obstacles = []
            for j, pos in enumerate(right_team):
                opp_dir: NDArray[np.float32] = (
                    right_team_direction[j] if j < len(right_team_direction)
                    else np.zeros(2, dtype=np.float32)
                )
                predicted_opp_pos: NDArray
[np.float32] = pos + opp_dir * k_opp
                obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 1.5})
            obstacles.append({"position": predicted_ball_pos, "sigma": 0.15, "scale": 2.5})

        else:
            k_ball = 5.0
            predicted_ball_pos = ball + ball_direction * k_ball
            goals = [{"position": predicted_ball_pos, "sigma": 0.25, "scale": -2.0}]

            k_opp = 3.0
            obstacles = []
            for j, pos in enumerate(right_team):
                opp_dir = (
                    right_team_direction[j] if j < len(right_team_direction)
                    else np.zeros(2, dtype=np.float32)
                )
                predicted_opp_pos = pos + opp_dir * k_opp
                obstacles.append({"position": predicted_opp_pos, "sigma": 0.06, "scale": 1.2})

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
        if base_obs.get("game_mode", 0) != 4:
            return None
        ball: NDArray[np.float32] = np.array(base_obs["ball"][:2])
        left_team: NDArray[np.float32] = np.array(base_obs["left_team"])
        return int(np.argmin(np.sum((left_team - ball) ** 2, axis=1)))

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
        NDArray[np.float32],  # shaped_rewards (11,)
        bool,                 # done
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

        # ── Đọc trạng thái trước bước ─────────────────────────────────────────
        last_obs  = self.last_raw_obs[0] if self.last_raw_obs is not None else None
        last_ball_owned:   int = last_obs["ball_owned_team"]           if last_obs else -1
        last_ball_owned_player: int = last_obs.get("ball_owned_player", -1) if last_obs else -1
        last_score: list[int]  = list(last_obs["score"])               if last_obs else [0, 0]

        # ── Corner-kick state machine (mirror _map_global_actions) ───────────
        kicker_id: Optional[int] = self._kicker_id

        if kicker_id is not None and self._has_kicked and last_obs is not None:
            b_prev:  NDArray[np.float32] = np.array(last_obs["ball"][:2])
            spd:     float = (
                float(np.linalg.norm(last_obs["ball"][3:5]))
                if len(last_obs["ball"]) >= 5 else 0.0
            )
            kp: NDArray[np.float32] = np.array(last_obs["left_team"][kicker_id])
            if spd > 0.1 or float(np.linalg.norm(b_prev - kp)) > 0.05:
                self._kicker_id  = None
                self._has_kicked = False
                kicker_id        = None

        # ── Ánh xạ hành động ─────────────────────────────────────────────────
        mapped_actions: NDArray[np.int64] = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            if active_mask[i]:
                tactic_idx        = int(local_actions[i]) % self.n_tactic_actions
                mapped_actions[i] = self.TACTIC_MAP[tactic_idx]
            else:
                if i == kicker_id:
                    if not self._has_kicked:
                        mapped_actions[i] = random.choice([9, 10])
                        self._has_kicked  = True
                    else:
                        mapped_actions[i] = 0
                else:
                    act: int = int(global_actions[i])
                    mapped_actions[i] = act if 0 < act <= 8 else 0

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
             ...

```

Dựa trên việc đối chiếu nghiêm ngặt giữa đặc tả kỹ thuật động học trong `observation.md` và mã nguồn các tệp tin hệ thống của bạn (`wrapper_global.py`, `wrapper_local.py`, `energy_field.py`, `HES_COMA.py`, `networks.py`), dưới đây là báo cáo nghiên cứu chi tiết về tính đúng đắn, rủi ro tiềm ẩn và các phương án hiệu chỉnh tối ưu cho Đồ án tốt nghiệp của bạn.

---

## 1. Kiểm tra tính hợp lý của Hàm phần thưởng (Reward Reasoning)

Hệ thống phần thưởng phân cấp của bạn hiện tại đã thiết lập rất tốt cấu trúc thứ bậc, tuy nhiên vẫn tồn tại hai điểm nghẽn toán học có thể gây chết hành vi (Policy Exploitation) hoặc làm chậm tốc độ hội tụ.

### Giai đoạn 1 (Phase 1 — `wrapper_global.py`)

* **Trạng thái hiện tại:** `ENV_SCALE = 0.2 / 11` (Bàn thắng chỉ đóng góp $\approx 0.018$ mỗi bước), `ENERGY_SCALE = 0.6`.
* **Điểm bất hợp lý:** Việc tăng tỷ trọng năng lượng để ép đặc vụ chạy chỗ là đúng đắn, nhưng `ENERGY_SCALE = 0.6` kết hợp với hiệu số thế năng đơn bước ($E_t - E_{t+1}$) có thể sinh ra phần thưởng dao động từ $0.05 \rightarrow 0.2$ **tại mỗi step**.
* **Hệ quả (Exploitation Bug):** Qua một quỹ đạo (trajectory) dài 100-150 steps, phần thưởng năng lượng tích lũy sẽ đạt tới hàng chục điểm, áp đảo hoàn toàn mốc Suggestion nền của bàn thắng ($0.018$). Mạng nơ-ron GAgent sẽ sinh ra hành vi "tha hóa": Các cầu thủ chỉ thích chạy lượn lờ quanh mép hố thế năng để "farm" điểm chênh lệch năng lượng liên tục thay vì đứng im tại vị trí tối ưu để chờ bóng.
* **Khắc phục:** Hạ `ENERGY_SCALE` xuống mốc **`0.05`** hoặc **`0.08`**. Điều này đảm bảo điểm thưởng chạy chỗ mịn màng qua từng bước nhưng tổng tích lũy cả episode không được vượt quá phần thưởng chuyển giao chiến thuật hoặc ghi bàn ròng.

### Giai đoạn 2 (Phase 2 — `wrapper_local.py`)

* **Trạng thái hiện tại:** Thưởng dứt điểm cá nhân rất lớn (`ROLE_PENALTY_SPOT = 1.5`, `ROLE_NEAR_POST = 1.0`) kết hợp phần thưởng áp sát bóng liên tục (`BALL_APPROACH_R = 0.3`).
* **Điểm bất hợp lý:** Phần thưởng `BALL_APPROACH_R` tính toán khoảng cách hình học đơn thuần tới bóng ở mỗi bước sau khi bóng vào vòng cấm.
* **Hệ quả (Ball Chasing Bug):** Khi quả bóng được tạt vào, chỉ số này vô tình tạo ra một lực hút cơ học kéo tuột toàn bộ các đặc vụ đang ở trạng thái `active` lao xúm vào quả bóng (hiện tượng bầy đàn), làm vỡ vụn sơ đồ chiến thuật giữ cự ly đón bóng hai mà mạng GAgent đã mất công học ở Phase 1.
* **Khắc phục:** Loại bỏ hoàn toàn `BALL_APPROACH_R`, hoặc chỉ áp dụng nó như một phần thưởng thưa thớt (Sparse) **chỉ cộng duy nhất một lần** khi đặc vụ thực hiện hành động tranh chấp/chạm bóng thành công (`current_ball_owned_player == i`).

---

## 2. Kiểm tra tính chính xác của Toán học Trường Năng lượng (`energy_field.py`)

### Biểu thức Kernel và Lập trình ma trận (Vectorization)

Hàm tính toán lõi thế năng Gaussian của bạn được viết rất xuất sắc:

```python
distance_sq: NDArray[np.float32] = np.array(np.sum((pos - target) ** 2, axis=-1))
return scale * np.exp(-distance_sq / (sigma ** 2))

```

* **Đánh giá:** Công thức toán học hoàn toàn chính xác theo lý thuyết trường thế năng phân phối. Cơ chế tính toán tận dụng tối đa sức mạnh xử lý mảng của Numpy (SIMD Alignment). Khi truyền `pos` dạng `(11, 2)` và `target` dạng `(2,)`, Numpy tự động thực hiện phép tính Broadcast và trả về mảng năng lượng `(11,)` của 11 người trong 1 chu kỳ xung nhịp. **Phần này chuẩn xác 100%.**

### 🚨 Lỗi Vật lý Động học nghiêm trọng trong Dự đoán Điểm rơi (`_extract_positions`)

Trong cả hai wrapper, bạn đang sử dụng phép toán dự đoán tuyến tính:

```python
predicted_ball_pos: NDArray[np.float32] = ball + ball_direction * k_ball  # với k_ball = 5.0

```

* **Sai lệch bản chất:** Theo đặc tả của `observation.md`, `ball_direction` trả về vector vận tốc 3D ($v_x, v_y, v_z$). Bạn cắt lấy `[3:5]` tức là trích xuất vận tốc sệt mặt đất ($v_x, v_y$).
* **Hệ quả (Trajectory Mismatch Bug):** Quả tạt phạt góc chiến thuật của bạn phát ra lệnh `action_high_pass = 10` (Chuyền bổng). Quả bóng sẽ bay theo quỹ đạo **Parabol 3D** chịu gia tốc trọng lực ($g$) và lực cản không khí rất nặng của GRF Engine.
Việc nhân tuyến tính `ball_direction * 5.0` giả định bóng lăn thẳng trên cỏ. Khi bóng đang bay bổng vòng cung, vector vận tốc $2D$ sẽ hướng thẳng ra biên dọc đối diện. Tâm hố năng lượng (Goals) của bạn sẽ bị phép toán này đẩy văng... ra ngoài khán đài hoặc xuyên qua lưới, khiến 10 cầu thủ chạy chỗ chạy hỗn loạn đuổi theo một "bóng ma vị trí" sai thực tế.
* **Khắc phục:** Trong kịch bản phạt góc cố định, bóng bay bổng luôn có một điểm rơi cố định trong vòng cấm. Bạn **không nên nhân với vector vận tốc bóng khi bóng đang bay trên không**, hãy giữ tâm hố năng lượng cố định tại các tọa độ vùng cấm (`near_post`, `far_post`, `penalty_spot`) như thiết lập tĩnh ban đầu. Chỉ áp dụng logic nhân `ball_direction` khi bóng đã tiếp đất (bóng sống bình thường).

---

## 3. Đánh giá Thiết kế Hành động & Lưu trữ Observations (Obs & Action Design)

### 🚨 Lỗi Bất đồng bộ State Machine giữa 2 Giai đoạn (Chí mạng)

Hãy đối chiếu hàm ánh xạ hành động an toàn `_map_global_actions` ở hai file wrapper:

* **Tại `wrapper_global.py` (Phase 1):** Kicker tuân thủ quy trình 3 bước nghiêm ngặt: Quay mặt nhắm bắn vào tâm vòng cấm `penalty_spot` $\rightarrow$ Phát lệnh tạt bổng gán cứng `10` (`action_high_pass`) $\rightarrow$ Dừng lại gán cứng `0`.
* **Tại `wrapper_local.py` (Phase 2):** Bạn lại lập trình gán ngẫu nhiên:
```python
mapped_actions[i] = random.choice([9, 10])  # Chọn ngẫu nhiên giữa Long Pass và High Pass

```


Đồng thời, Phase 2 hoàn toàn **bỏ qua bước quay mặt nhắm hướng** (`_get_aim_action`).
* **Hệ quả hệ thống (Convergence Failure):** Đây là nguyên nhân khiến mô hình Phase 2 của bạn rất khó hội tụ. Ở Phase 1, 10 đặc vụ chạy chỗ đã được học một Policy hoàn hảo dựa trên giả định: *Quả bóng luôn được tạt bổng chuẩn xác hướng về chấm 11m*. Sang Phase 2, bạn đóng băng GAgent (Frozen), nhưng lại để Kicker đá ngẫu nhiên (lúc chuyền sệt mặt đất dính chân hậu vệ, lúc tạt bổng) và không thèm ngắm hướng. Quả bóng rơi sai vị trí học tập khiến các GAgent bị rơi vào trạng thái "ảo giác vị trí", dẫn đến việc LAgent hiếm khi kích hoạt được hành động dứt điểm chiến thuật.
* **Khắc phục:** Bắt buộc phải copy sao chép nguyên vẹn 100% logic State Machine của Kicker (bao gồm cả hàm nhắm hướng `aim_target` và khóa cứng hành động tạt bổng `10`) từ `wrapper_global.py` sang `wrapper_local.py`. Hai Phase phải đồng nhất về mặt vật lý môi trường.

### Đánh giá cấu trúc làm giàu thông tin quan sát (Obs Enrichment)

Việc bạn nén vector quan sát lên `obs_dim = 54` bằng cách bổ sung `is_ball_owned_arr`, `sticky_actions`, và `role_onehot` là **hoàn toàn chính xác, cực kỳ tường minh**.

* Nó giải quyết triệt để lỗi "nhân bản hành vi" (Clone behavior) của Parameter Sharing nhờ dấu vết vai trò `role_onehot`.
* Nó cung cấp vector vận tốc ẩn cho mạng MLP tĩnh thông qua dữ liệu nút bấm `sticky_actions`.
* **Lỗi nhỏ cần sửa (Docstring Typo):** Trong file `wrapper_local.py`, tại hàm `reset` và `step_local`, docstring của bạn ghi chú đầu ra là `obs_g: (11, 33)`. Tuy nhiên, code thực tế bên trong hàm `_process_global_obs` lại trả về ma trận chuẩn hóa 54 chiều để khớp với mạng Phase 1. Bạn nên sửa lại thông tin text của docstring thành `obs_g: (11, 54)` để tránh gây nhầm lẫn khi hội đồng đọc code phản biện.

---

## 4. Tổng hợp bảng điều chỉnh thông số tối ưu cho Đồ án

| Thành phần tệp tin | Vị trí đoạn mã | Lỗi hiện tại | Giải pháp hiệu chỉnh chuẩn khoa học |
| --- | --- | --- | --- |
| **`wrapper_global.py`** | Hàm `step()` | `ENERGY_SCALE = 0.6` quá lớn, lấn át bàn thắng. | Hạ xuống `ENERGY_SCALE = 0.05` để mịn hóa điểm thế năng. |
| **`wrapper_global.py`** | Hàm `_extract_positions` | Nhân tuyến tính vận tốc bóng $2D$ khi bóng đang bay bổng. | Khóa cứng tọa độ Landing Zones cố định khi bóng đang bay bổng trên không. |
| **`wrapper_local.py`** | Hàm `_map_global_actions` | Kicker đá ngẫu nhiên `random.choice([9,10])`, không ngắm hướng. | Sao chép 100% State Machine nhắm bắn và tạt bổng cố định từ Phase 1 sang. |
| **`wrapper_local.py`** | Hàm `step_local()` | Thưởng áp sát bóng liên tục `BALL_APPROACH_R` gây vỡ đội hình. | Loại bỏ, chuyển thành thưởng Sparse duy nhất 1 lần khi đặc vụ chạm cắt bóng thành công. |

Tiến hành đồng bộ hóa vật lý giữa 2 Phase và thu nhỏ dải điểm thưởng chạy chỗ đơn bước sẽ giúp hệ thống HES-COMA của bạn vận hành mượt mà, tốc độ hội tụ đồ thị trên TensorBoard sẽ nhanh hơn rõ rệt và đạt độ tin cậy cao để bảo vệ đồ án!