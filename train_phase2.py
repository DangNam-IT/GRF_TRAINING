import os
import argparse

import numpy as np
import gfootball.env as football_env

from envs.wrapper_local import GFootballLocalWrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger


def create_env(args):
    return football_env.create_environment(
        env_name="academy_corner",
        number_of_left_players_agent_controls=args.number_agents,
        representation='raw',
        rewards='scoring',
        render=args.render,
        write_full_episode_dumps=(args.dump_freq > 0),
        dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,
        logdir=args.video_dir
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 (LAgent)")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent_model_test2", help="Path to load GAgent model")
    parser.add_argument("--lagent_model", type=str, default="experiments/models/lagent_model_test2", help="Path to save LAgent model")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq", type=int, default=500, help="Model save frequency")
    parser.add_argument("--log_file", type=str, default="experiments/phase2_training_test2.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase2", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_env = create_env(args)
    env      = GFootballLocalWrapper(base_env, num_agents=args.number_agents)

    # ── GAgent: tải model đã frozen từ Phase 1 ───────────────────────────────
    model_path = args.gagent_model
    gagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_g,   # 54 chiều
        n_actions=9,
        n_agents=args.number_agents
    )
    n_episodes = args.episodes
    max_steps  = args.max_steps


    if os.path.exists(model_path + '_ep_' + str(n_episodes) +'.pth'):
        gagent.load_model(model_path, n_episodes)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {model_path}. GAgent chạy với trọng số ngẫu nhiên!")

    # Freeze GAgent hoàn toàn — Phase 2 CHỈ train LAgent
    for param in gagent.actor.parameters():
        param.requires_grad = False
    for param in gagent.critic.parameters():
        param.requires_grad = False

    # ── LAgent: khởi tạo mới, obs_dim = 7 (tactical context) ─────────────────
    lagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_l,          # 32 chiều (ray-info)
        n_actions=env.n_tactic_actions,  # 2 tactical actions
        n_agents=args.number_agents
    )
    buffer = RolloutBuffer()
    logger = CSVLogger(
        args.log_file,
        ['Episode', 'Total_Local_Reward']
    )

    # ── GAgent "stop" actions — output 8 hoặc 9 → GRF idle ──────────────────
    # Trong wrapper_global: actions 0-7 → GRF 1-8 (di chuyển), 8-9 → GRF 0, 14 (stop).
    STOP_ACTIONS = set(env.STOP_ACTIONS)

    print("Bắt đầu huấn luyện Phase 2 (Local Agent)...")
    for episode in range(1, n_episodes + 1):
        state, obs_g, obs_l = env.reset()
        buffer.clear()
        total_reward  = 0.0


        for t in range(max_steps):
            # ── Bước 1: GAgent (frozen) ra quyết định ────────────────────────
            actions_g, _ = gagent.get_actions(obs_g)

            # ── Bước 2: Xác định agent nào chọn STOP ─────────────────────────
            # active_mask[i] = True → GAgent[i] dừng → LAgent[i] được kích hoạt
            active_mask = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)

            if np.any(active_mask):
                # ── Bước 3a: LAgent quyết định tactical action ───────────────
                actions_l, _ = lagent.get_actions(obs_l)

                next_state, next_obs_g, next_obs_l, rewards, done = env.step_local(
                    actions_l, active_mask, actions_g
                )

                # Chỉ lưu vào buffer khi LAgent thực sự active
                # Buffer lưu (s_t, o_t^l, a_t^l, r_t^l, s_{t+1}, o_{t+1}^l, d)
                buffer.store(state, obs_l, actions_l, rewards, next_state, next_obs_l, done)
                total_reward += np.sum(rewards)

            else:
                # ── Bước 3b: Tất cả di chuyển → LAgent không hành động ───────
                next_state, next_obs_g, next_obs_l, _, done = env.step_global(actions_g)

            state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
            if done:
                break

        # ── Cập nhật LAgent sau mỗi episode ─────────────────────────────────
        lagent.update(buffer)

        logger.log([episode, total_reward])
        print(f"Episode: {episode:4d}/{n_episodes} | "
            #   f"Steps_Local: {steps_local:3d} | "
              f"Reward: {total_reward:.3f}")

        if episode % args.save_freq == 0:
            # Lưu model định kỳ
            os.makedirs(os.path.dirname(args.lagent_model), exist_ok=True)
            lagent.save_model(args.lagent_model, episode=episode)

    print("Huấn luyện Phase 2 hoàn tất. Model LAgent đã được lưu.")


if __name__ == '__main__':
    main()