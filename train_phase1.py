from __future__ import annotations

import os
import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gfootball.env as football_env
from gym import Env

from envs.wrapper_global import GFootballGlobalWrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger


def create_env(args: argparse.Namespace) -> Env:
    """Tạo môi trường GRF cho Phase 1 (không render để tăng tốc training)."""
    return football_env.create_environment(
        env_name="academy_corner",
        number_of_left_players_agent_controls=11,
        representation="raw",
        rewards="scoring",
        render=args.render,  # Bật render để xem quá trình huấn luyện
        write_full_episode_dumps=(args.dump_freq > 0),
        dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,  # Lưu video mỗi 100 episode
        logdir=args.video_dir
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 (GAgent)")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=50, help="Model save frequency")
    parser.add_argument("--model_path", type=str, default="experiments/models/gagent_model", help="Path to save the model")
    parser.add_argument("--log_file", type=str, default="experiments/phase1_training.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
    parser.add_argument("--load_model", type=str, default="", help="Path to load a pre-trained model")
    parser.add_argument("--load_episode", type=int, default=0, help="Episode number of the loaded model")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_env: Env                   = create_env(args)
    env:      GFootballGlobalWrapper = GFootballGlobalWrapper(base_env, num_agents=args.number_agents)
    agent:    HES_COMA_Agent         = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim,
        n_actions=9,
        n_agents=args.number_agents,
    )
    
    if args.load_model and args.load_episode > 0:
        agent.load_model(args.load_model, episode=args.load_episode)

    buffer: RolloutBuffer = RolloutBuffer()
    logger: CSVLogger     = CSVLogger(
        args.log_file,
        [
            "Episode", 
            "Reward_total", 
            "Reward_env", 
            "Reward_energy", 
            "Reward_possession",
            "Reward_handover"
        ],
    )

    n_episodes: int = args.episodes
    max_steps:  int = args.max_steps

    print("Bắt đầu huấn luyện Phase 1 (Global Agent)...")
    for episode in range(1, n_episodes + 1):
        state: NDArray[np.float32]
        obses: NDArray[np.float32]
        state, obses = env.reset()
        buffer.clear()

        total_reward:       float = 0.0
        total_env:          float = 0.0
        total_energy:       float = 0.0
        total_possession:   float = 0.0
        total_handover:     float = 0.0

        for t in range(max_steps):
            actions: NDArray[np.int64]
            actions, _ = agent.get_actions(obses)

            next_state:   NDArray[np.float32]
            next_obses:   NDArray[np.float32]
            rewards:      NDArray[np.float32]
            rewards_view: dict[str, NDArray[np.float32]]
            done:         bool

            next_state, next_obses, rewards, rewards_view, done = env.step(actions)

            buffer.store(state, obses, actions, rewards, next_state, next_obses, done)

            total_reward     += float(np.sum(rewards))
            total_env        += float(np.sum(rewards_view["rewards_env"]))
            total_energy     += float(np.sum(rewards_view["rewards_energy"]))
            total_possession += float(np.sum(rewards_view["rewards_possession"]))
            total_handover   += float(np.sum(rewards_view["rewards_handover"]))

            state, obses = next_state, next_obses
            if done:
                break

        agent.update(buffer)
        logger.log([episode, total_reward, total_env, total_energy, total_possession, total_handover])
        print(
            f"Episode: {episode}/{n_episodes} | "
            f"Total: {total_reward:.3f} | "
            f"Env: {total_env:.3f} | "
            f"Energy: {total_energy:.3f} | "
            f"Possession: {total_possession:.3f} | "
            f"Handover: {total_handover:.3f}"
        )
        if episode % args.save_freq_model == 0:
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            agent.save_model(args.model_path, episode=episode)
    print("Huấn luyện hoàn tất. Đã lưu mô hình GAgent cùng cấu hình môi trường.")


if __name__ == "__main__":
    main()