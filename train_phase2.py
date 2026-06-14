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
        logdir=args.video_dir,
        other_config_options={'action_set': 'v2'},
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 (LAgent)")
    parser.add_argument("--eps", type=int, default=3000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent/test1/g_model", help="Path to load GAgent model")
    parser.add_argument("--lagent_model", type=str, default="experiments/models/lagent/l_model", help="Path to save LAgent model")
    parser.add_argument("--pre_lagent", type=str, default="experiments/models/lagent/test5/l_model", help="Path to load a pre-trained LAgent model")

    parser.add_argument("--g_eps",     type=int,   default=400,                               help="Number eps to run global agent")
    parser.add_argument("--l_eps",     type=int,   default=0,                               help="Number episodes to run local agent")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=100, help="Model save frequency")
    parser.add_argument("--log_file", type=str, default="experiments/train_reward/one/l_train.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase2", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")

    # # Colab-specific arguments
    # parser.add_argument("--gagent_model", type=str, default="content/drive/MyDrive/experiments/models/gagent/g_model", help="Path to load GAgent model")
    # parser.add_argument("--lagent_model", type=str, default="content/drive/MyDrive/experiments/models/lagent/test6/l_model", help="Path to save LAgent model")
    # parser.add_argument("--pre_lagent", type=str, default="content/drive/MyDrive/experiments/models/lagent/test5/l_model", help="Path to load a pre-trained LAgent model")
    # parser.add_argument("--log_file", type=str, default="content/drive/MyDrive/experiments/l_train.csv", help="Path to the log CSV file")
    # parser.add_argument("--video_dir", type=str, default="content/drive/MyDrive/experiments/videos/phase2/test6", help="Directory to save videos")
    return parser.parse_args()


def main():
    args = parse_args()
    base_env = create_env(args)
    env      = GFootballLocalWrapper(base_env, num_agents=args.number_agents)

    # ── GAgent: tải model đã frozen từ Phase 1 ───────────────────────────────
    model_path = args.gagent_model
    g_eps_model   = args.g_eps
    gagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_g,   # 70 chiều (48 rays + energy + ball_owned + sticky + role)
        n_actions=10,
        n_agents=args.number_agents
    )

       # ── LAgent: khởi tạo mới, obs_dim = 7 (tactical context) ─────────────────
    lagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_l,          # 48 chiều (ray-info 16×3, không energy)
        n_actions=env.n_tactic_actions,  # 2 tactical actions
        n_agents=args.number_agents
    )

    n_episodes = args.eps
    max_steps  = args.max_steps

    for param in gagent.actor.parameters():
        param.requires_grad = False
    for param in gagent.critic.parameters():
        param.requires_grad = False

    if os.path.exists(model_path + '_ep_' + str(g_eps_model) +'.pth'):
        gagent.load_model(model_path, g_eps_model)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {model_path}. GAgent chạy với trọng số ngẫu nhiên!")

    if os.path.exists(args.pre_lagent + '_ep_' + str(args.l_eps) +'.pth') and args.l_eps > 0:
        lagent.load_model(args.pre_lagent, episode=args.l_eps)
        print(f"Loaded LAgent từ ep {args.l_eps} của {args.pre_lagent}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy pre-trained {args.pre_lagent}. LAgent chạy với trọng số ngẫu nhiên!")

 
    buffer = RolloutBuffer()
    logger = CSVLogger(
        args.log_file,
        [
            'Episode',
            'Total_Local_Reward',
            'R_env',
            'R_passing',
            # 'R_facing',
            'R_in_box',
            'R_assist',
            'R_role',
            'R_approach',
            # 'R_possession',
        ]
    )

    # ── GAgent "stop" actions — output 8 hoặc 9 → GRF idle ──────────────────
    # Trong wrapper_global: actions 0-7 → GRF 1-8 (di chuyển), 8-9 → GRF 0, 14 (stop).
    STOP_ACTIONS = set(env.STOP_ACTIONS)

    print("Bắt đầu huấn luyện Phase 2 (Local Agent)...")
    for episode in range(1, n_episodes + 1):
        state, obs_g, obs_l = env.reset()
        buffer.clear()
        total_reward = 0.0
        # Biến tích lũy phần thưởng thành phần theo episode
        ep_R_env        = 0.0
        ep_R_passing    = 0.0
        # ep_R_facing     = 0.0
        ep_R_in_box     = 0.0
        ep_R_assist     = 0.0
        ep_R_role       = 0.0
        ep_R_approach   = 0.0
        # ep_R_possession = 0.0


        for t in range(max_steps):
            # ── Bước 1: GAgent (frozen) ra quyết định ────────────────────────
            actions_g, _ = gagent.get_actions(obs_g)

            # ── Bước 2: Xác định agent nào chọn STOP ─────────────────────────
            # active_mask[i] = True → GAgent[i] dừng → LAgent[i] được kích hoạt
            active_mask = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)

            # ── Bước 3a: LAgent quyết định tactical action ───────────────
            actions_l, _ = lagent.get_actions(obs_l)

            next_state, next_obs_g, next_obs_l, rewards, done, reward_info = env.step_local(
                actions_l, active_mask, actions_g
            )

            # Lưu vào buffer kèm active_mask
            # Buffer lưu (s_t, o_t^l, a_t^l, r_t^l, s_{t+1}, o_{t+1}^l, d, mask)
            buffer.store(state, obs_l, actions_l, rewards, next_state, next_obs_l, done, active_mask)
            total_reward += np.sum(rewards)

            # Tích lũy từng phần thưởng thành phần
            ep_R_env        += reward_info['R_env']
            ep_R_passing    += reward_info['R_passing']
            # ep_R_facing     += reward_info['R_facing']
            ep_R_in_box     += reward_info['R_in_box']
            ep_R_assist     += reward_info['R_assist']
            ep_R_role       += reward_info['R_role']
            ep_R_approach   += reward_info['R_approach']
            # ep_R_possession += reward_info['R_possession']

            state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
            if done:
                break

        # ── Cập nhật LAgent sau mỗi episode ───────────────────────────────
        lagent.update(buffer)

        logger.log([
            episode,
            total_reward,
            ep_R_env,
            ep_R_passing,
            # ep_R_facing,
            ep_R_in_box,
            ep_R_assist,
            ep_R_role,
            ep_R_approach,
            # ep_R_possession,
        ])
        print(f"Episode: {episode:4d}/{n_episodes} | "
              f"Reward: {total_reward:.3f} | "
              f"env={ep_R_env:.2f} pass={ep_R_passing:.2f} "
              f"box={ep_R_in_box:.2f} "
              f"asst={ep_R_assist:.2f} role={ep_R_role:.2f} "
              f"appr={ep_R_approach:.2f} ")

        if episode % args.save_freq_model == 0:
            # Lưu model định kỳ
            os.makedirs(os.path.dirname(args.lagent_model), exist_ok=True)
            lagent.save_model(args.lagent_model, episode=episode)

    print("Huấn luyện Phase 2 hoàn tất. Model LAgent đã được lưu.")


if __name__ == '__main__':
    main()