from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gfootball.env as football_env
from gym import Env

from envs.wrapper_global import GFootballGlobalWrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger


def create_env() -> Env:
    """Tạo môi trường GRF cho Phase 1 (không render để tăng tốc training)."""
    return football_env.create_environment(
        env_name="academy_corner",
        number_of_left_players_agent_controls=11,
        representation="raw",
        rewards="scoring",
        render=False,
    )


def main() -> None:
    base_env: Env                   = create_env()
    env:      GFootballGlobalWrapper = GFootballGlobalWrapper(base_env, num_agents=11)
    agent:    HES_COMA_Agent         = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim,
        n_actions=10,
        n_agents=11,
    )
    buffer: RolloutBuffer = RolloutBuffer()
    logger: CSVLogger     = CSVLogger(
        "experiments/phase1_training_test.csv",
        ["Episode", "Total_reward_buffer", "Reward_env", "Reward_energy", "Reward_ball_owned"],
    )

    n_episodes: int = 1000
    max_steps:  int = 150

    print("Bắt đầu huấn luyện Phase 1 (Global Agent)...")
    for episode in range(1, n_episodes + 1):
        state: NDArray[np.float32]
        obses: NDArray[np.float32]
        state, obses = env.reset()
        buffer.clear()

        total_reward:          float = 0.0
        view_rewards_energy:   float = 0.0
        view_rewards_ball_owned: float = 0.0

        for t in range(max_steps):
            actions: NDArray[np.int64]
            actions, _ = agent.get_actions(obses)

            next_state: NDArray[np.float32]
            next_obses: NDArray[np.float32]
            rewards:    NDArray[np.float32]
            view_rewards: dict[str, NDArray[np.float32]]
            done: bool
            next_state, next_obses, rewards, view_rewards, done = env.step(actions)

            buffer.store(state, obses, actions, rewards, next_state, done)

            total_reward            += float(np.sum(rewards))
            view_rewards_energy     += float(np.sum(view_rewards["rewards_energy"]))
            view_rewards_ball_owned += float(np.sum(view_rewards["rewards_ball_owned"]))

            state, obses = next_state, next_obses
            if done:
                break

        agent.update(buffer)
        logger.log([
            episode,
            total_reward,
            view_rewards_energy,
            view_rewards_ball_owned,
        ])
        print(
            f"Episode: {episode}/{n_episodes} | "
            f"Total_reward: {total_reward:.2f} | "
            f"Reward_energy: {view_rewards_energy:.2f} | "
            f"Reward_ball_owned: {view_rewards_ball_owned:.2f}"
        )

    os.makedirs("experiments/models", exist_ok=True)
    agent.save_model("experiments/models/gagent_model.pth")
    print("Huấn luyện hoàn tất. Đã lưu mô hình GAgent cùng cấu hình môi trường.")


if __name__ == "__main__":
    main()