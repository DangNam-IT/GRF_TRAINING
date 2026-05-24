from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class RolloutBuffer:
    """Bộ đệm rollout lưu trữ transition của toàn bộ 11 agents theo từng step."""

    def __init__(self) -> None:
        self.states:      list[NDArray[np.float32]] = []
        self.obses:       list[NDArray[np.float32]] = []
        self.actions:     list[NDArray[np.int64]]   = []
        self.rewards:     list[NDArray[np.float32]] = []
        self.next_states: list[NDArray[np.float32]] = []
        self.dones:       list[bool]                = []

    def store(
        self,
        state:      NDArray[np.float32],
        obses:      NDArray[np.float32],
        actions:    NDArray[np.int64],
        rewards:    NDArray[np.float32],
        next_state: NDArray[np.float32],
        done:       bool,
    ) -> None:
        """Lưu trữ transition của toàn bộ 11 agents trong 1 step."""
        self.states.append(state)
        self.obses.append(obses)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu trong buffer."""
        self.states.clear()
        self.obses.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def get_data(self) -> Tuple[
        NDArray[np.float32],  # states       (T, state_dim)
        NDArray[np.float32],  # obses        (T, n_agents, obs_dim)
        NDArray[np.int64],    # actions      (T, n_agents)
        NDArray[np.float32],  # rewards      (T, n_agents)
        NDArray[np.float32],  # next_states  (T, state_dim)
        NDArray[np.float32],  # dones        (T,)
    ]:
        return (
            np.array(self.states,      dtype=np.float32),
            np.array(self.obses,       dtype=np.float32),
            np.array(self.actions,     dtype=np.int64),
            np.array(self.rewards,     dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
            np.array(self.dones,       dtype=np.float32),
        )