# from __future__ import annotations

# import os
# from typing import Any

# import numpy as np
# from numpy.typing import NDArray
# import gfootball.env as football_env
# from gym import Env

# from envs.wrapper_global import GFootballGlobalWrapper
# from agents.HES_COMA import HES_COMA_Agent
# from utils.buffer import RolloutBuffer
# from utils.logger import CSVLogger


# def create_env() -> Env:
#     """Tạo môi trường GRF cho Phase 1 (không render để tăng tốc training)."""
#     return football_env.create_environment(
#         env_name="academy_corner",
#         number_of_left_players_agent_controls=11,
#         representation="raw",
#         rewards="scoring",
#         render=True,
#     )


# def main() -> None:
#     base_env: Env                   = create_env()
#     env:      GFootballGlobalWrapper = GFootballGlobalWrapper(base_env, num_agents=11)
#     agent:    HES_COMA_Agent         = HES_COMA_Agent(
#         state_dim=env.state_dim,
#         obs_dim=env.obs_dim,
#         n_actions=9,
#         n_agents=11,
#     )
#     # agent.load_model('experiments/models/gagent_model_test3', episode=1000)  # Tải model đã huấn luyện từ Phase 1
    
#     buffer: RolloutBuffer = RolloutBuffer()
#     logger: CSVLogger     = CSVLogger(
#         "experiments/phase1_training_test4.csv",
#         [
#             "Episode", 
#             "Reward_total", 
#             "Reward_env", 
#             "Reward_energy", 
#             "Reward_possession",
#             "Reward_handover"
#         ],
#     )

#     n_episodes: int = 3000
#     max_steps:  int = 150

#     print("Bắt đầu huấn luyện Phase 1 (Global Agent)...")
#     for episode in range(1, n_episodes + 1):
#         state: NDArray[np.float32]
#         obses: NDArray[np.float32]
#         state, obses = env.reset()
#         buffer.clear()

#         total_reward:       float = 0.0
#         total_env:          float = 0.0
#         total_energy:       float = 0.0
#         total_possession:   float = 0.0
#         total_handover:     float = 0.0

#         for t in range(max_steps):
#             actions: NDArray[np.int64]
#             actions, _ = agent.get_actions(obses)

#             next_state:   NDArray[np.float32]
#             next_obses:   NDArray[np.float32]
#             rewards:      NDArray[np.float32]
#             rewards_view: dict[str, NDArray[np.float32]]
#             done:         bool

#             next_state, next_obses, rewards, rewards_view, done = env.step(actions)

#             buffer.store(state, obses, actions, rewards, next_state, next_obses, done)

#             total_reward     += float(np.sum(rewards))
#             total_env        += float(np.sum(rewards_view["rewards_env"]))
#             total_energy     += float(np.sum(rewards_view["rewards_energy"]))
#             total_possession += float(np.sum(rewards_view["rewards_possession"]))
#             total_handover   += float(np.sum(rewards_view["rewards_handover"]))

#             state, obses = next_state, next_obses
#             if done:
#                 break

#         agent.update(buffer)
#         logger.log([episode, total_reward, total_env, total_energy, total_possession, total_handover])
#         print(
#             f"Episode: {episode}/{n_episodes} | "
#             f"Total: {total_reward:.3f} | "
#             f"Env: {total_env:.3f} | "
#             f"Energy: {total_energy:.3f} | "
#             f"Possession: {total_possession:.3f} | "
#             f"Handover: {total_handover:.3f}"
#         )
#         # if episode % 500 == 0:
#         #     os.makedirs("experiments/models", exist_ok=True)
#         #     agent.save_model("experiments/models/gagent_model_test3", episode=episode)
#     print("Huấn luyện hoàn tất. Đã lưu mô hình GAgent cùng cấu hình môi trường.")


# if __name__ == "__main__":
#     main()


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
        number_of_left_players_agent_controls=11,
        representation='raw',
        rewards='scoring',
        render=args.render,
        write_full_episode_dumps=(args.dump_freq > 0),
        dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,
        logdir=args.video_dir
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Test Phase 2 (LAgent)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train/test")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent_model", help="Path to load GAgent model")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase2", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_env = create_env(args)
    env      = GFootballLocalWrapper(base_env, num_agents=11)

    # ── GAgent: tải model đã frozen từ Phase 1 ───────────────────────────────
    model_path = args.gagent_model
    gagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_g,   # 54 chiều
        n_actions=9,
        n_agents=11
    )
    n_episodes = args.episodes
    max_steps  = args.max_steps


    if os.path.exists(model_path + '_ep_' + str(n_episodes) +'.pth'):
        gagent.load_model(model_path, n_episodes)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {model_path}. GAgent chạy với trọng số ngẫu nhiên!")

    # Freeze GAgent hoàn toàn — Phase 2 CHỈ train LAgent
    # for param in gagent.actor.parameters():
    #     param.requires_grad = False
    # for param in gagent.critic.parameters():
    #     param.requires_grad = False

    # ── LAgent: khởi tạo mới, obs_dim = 7 (tactical context) ─────────────────
    lagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_l,          # 32 chiều (ray-info)
        n_actions=env.n_tactic_actions,  # 2 tactical actions
        n_agents=11
    )
    buffer = RolloutBuffer()
    # logger = CSVLogger(
    #     'experiments/phase2_training_test2.csv',
    #     ['Episode', 'Total_Local_Reward']
    # )

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
            a = 0
            # ── Bước 2: Xác định agent nào chọn STOP ─────────────────────────
            # active_mask[i] = True → GAgent[i] dừng → LAgent[i] được kích hoạt
            active_mask = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)

            if np.any(active_mask):
                # ── Bước 3a: LAgent quyết định tactical action ───────────────
                actions_l, _ = lagent.get_actions(obs_l)

                next_state, next_obs_g, next_obs_l, rewards, done = env.step_local(
                    actions_l, active_mask, actions_g
                )
                a+=1

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

        # logger.log([episode, total_reward])
        print(f"Episode: {episode:4d}/{n_episodes} | "
            #   f"Steps_Local: {steps_local:3d} | "
              f"Reward: {total_reward:.3f}" 
              )

        # if episode % 500 == 0:
        #     # Lưu model định kỳ
        #     os.makedirs('experiments/models', exist_ok=True)
        #     lagent.save_model('experiments/models/lagent_model_test2', episode=episode)

    print("Huấn luyện Phase 2 hoàn tất. Model LAgent đã được lưu.")


if __name__ == '__main__':
    main()