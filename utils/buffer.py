from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class RolloutBuffer:
    """Bộ đệm rollout lưu trữ transition của toàn bộ 11 agents theo từng step.

    Lưu đủ tuple (s_t, o_t, a_t, r_t, s_{t+1}, o_{t+1}, d) theo algo_logic.md §4.2.1 Bước 9.
    next_obses cần thiết để tính TD-target đúng on-policy:
      a'_t = π(o_{t+1})  →  Q(s', a') — Bước 14.
    """

    def __init__(self) -> None:
        self.states:       list[NDArray[np.float32]] = []
        self.obses:        list[NDArray[np.float32]] = []
        self.actions:      list[NDArray[np.int64]]   = []
        self.rewards:      list[NDArray[np.float32]] = []
        self.next_states:  list[NDArray[np.float32]] = []
        self.next_obses:   list[NDArray[np.float32]] = []  # o_{t+1} cho TD-target
        self.dones:        list[bool]                = []

    def store(
        self,
        state:       NDArray[np.float32],
        obses:       NDArray[np.float32],
        actions:     NDArray[np.int64],
        rewards:     NDArray[np.float32],
        next_state:  NDArray[np.float32],
        next_obses:  NDArray[np.float32],
        done:        bool,
    ) -> None:
        """Lưu trữ transition (s_t, o_t, a_t, r_t, s_{t+1}, o_{t+1}, d) cho 11 agents."""
        self.states.append(state)
        self.obses.append(obses)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.next_obses.append(next_obses)
        self.dones.append(done)

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu trong buffer."""
        self.states.clear()
        self.obses.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_obses.clear()
        self.dones.clear()

    def get_data(self) -> Tuple[
        NDArray[np.float32],  # states        (T, state_dim)
        NDArray[np.float32],  # obses         (T, n_agents, obs_dim)
        NDArray[np.int64],    # actions       (T, n_agents)
        NDArray[np.float32],  # rewards       (T, n_agents)
        NDArray[np.float32],  # next_states   (T, state_dim)
        NDArray[np.float32],  # next_obses    (T, n_agents, obs_dim)
        NDArray[np.float32],  # dones         (T,)
    ]:
        return (
            np.array(self.states,      dtype=np.float32),
            np.array(self.obses,       dtype=np.float32),
            np.array(self.actions,     dtype=np.int64),
            np.array(self.rewards,     dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.next_obses,  dtype=np.float32),
            np.array(self.dones,       dtype=np.float32),
        )