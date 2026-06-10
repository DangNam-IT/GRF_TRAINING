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
        logdir=args.video_dir,
        # [builtin_ai] 'action_set' là config key nội bộ — phải truyền qua other_config_options
        # action_set_v2 = action_set_v1 + [action_builtin_ai=19]
        other_config_options={'action_set': 'v2'},
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Test Phase 2 (LAgent)")
    parser.add_argument("--g_eps",     type=int,   default=500,                               help="Số episodes chạy global agent")
    parser.add_argument("--l_eps",     type=int,   default=500,                               help="Số episodes chạy local agent")
    parser.add_argument("--max_steps",   type=int,   default=150,                               help="Bước tối đa mỗi episode")
    parser.add_argument("--render",      action="store_true",  default=True,                    help="Bật render")
    parser.add_argument("--no_render",   action="store_false", dest="render",                   help="Tắt render")
    parser.add_argument("--g_model",type=str,   default="experiments/models/gagent/g_model",help="Đường dẫn model GAgent")
    parser.add_argument("--l_model",type=str,   default="experiments/models/lagent/l_model",help="Đường dẫn model LAgent")
    parser.add_argument("--video_dir",   type=str,   default="experiments/videos/phase2/test",        help="Thư mục video")
    parser.add_argument("--dump_freq",   type=int,   default=0,                                 help="Tần suất xuất video (0 = tắt)")
    # ── Built-in AI exploration ─────────────────────────────────────────────────
    parser.add_argument("--builtin_eps", type=float, default=0.0,
        help="Xác suất [0.0–1.0] dùng action_builtin_ai=19 thay vì GAgent. "
             "0.0 = tắt (chỉ dùng GAgent); 1.0 = chỉ dùng Built-in AI")
    return parser.parse_args()


def main():
    args = parse_args()
    base_env = create_env(args)
    env      = GFootballLocalWrapper(base_env, num_agents=11)

    # ── GAgent: tải model đã frozen từ Phase 1 ───────────────────────────────
    model_path  = args.g_model
    lmodel_path = args.l_model

    gagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_g,
        n_actions=10,
        n_agents=11
    )

    # ── LAgent ────────────────────────────────────────────────────────
    lagent = HES_COMA_Agent(
        state_dim=env.state_dim,
        obs_dim=env.obs_dim_l,
        n_actions=env.n_tactic_actions,
        n_agents=11
    )

    # Inference mode — không cần gradient
    gagent.actor.eval()
    gagent.critic.eval()
    lagent.actor.eval()
    lagent.critic.eval()

    n_episodes = args.g_eps
    m_episodes = args.l_eps
    max_steps  = args.max_steps
    builtin_eps: float = args.builtin_eps   # Xác suất dùng Built-in AI

    # ── Load models ───────────────────────────────────────────────────
    if os.path.exists(model_path + '_ep_' + str(n_episodes) + '.pth'):
        gagent.load_model(model_path, n_episodes)
        print(f"[GAgent] Đã tải: {model_path}_ep_{n_episodes}.pth")
    else:
        print(f"[GAgent] CẢNH BÁO: Không tìm thấy model global → chạy với trọng số ngẫu nhiên")

    if os.path.exists(lmodel_path + '_ep_' + str(m_episodes) + '.pth'):
        lagent.load_model(lmodel_path, m_episodes)
        print(f"[LAgent] Đã tải: {lmodel_path}_ep_{m_episodes}.pth")
    else:
        print(f"[LAgent] CẢNH BÁO: Không tìm thấy model local → chạy với trọng số ngẫu nhiên")

    STOP_ACTIONS        = set(env.STOP_ACTIONS)
    ACTION_BUILTIN_AI   = 19   # V2 action set — đã bật với action_set='v2'

    print(f"\nBắt đầu test | Episodes: {n_episodes} | builtin_eps: {builtin_eps:.0%}")
    print("-" * 60)

    for episode in range(1, n_episodes + 1):
        state, obs_g, obs_l = env.reset()
        total_reward   = 0.0
        # steps_builtin  = 0   # Số bước dùng Built-in AI
        # steps_gagent   = 0   # Số bước dùng GAgent thật sự
        # steps_local    = 0   # Số bước LAgent được kích hoạt

        for t in range(max_steps):
            # # ── Bước 1: Chọn nguồn hành động cho tất cả 11 agent ─────────
            # # Epsilon-Greedy: với xác suất builtin_eps → dùng Built-in AI
            # if builtin_eps > 0.0 and np.random.rand() < builtin_eps:
            #     # [BUILTIN] Gửi action=19 THẲNG vào GRF engine — KHÔNG qua
            #     # _map_global_actions (vì hàm đó sẽ map act>=8 → GRF 14)
            #     grf_actions = np.full(env.num_agents, ACTION_BUILTIN_AI, dtype=np.int64)
            #     raw_obs_bi, rewards, done, _ = env.env.step(grf_actions)
            #     env.last_raw_obs = raw_obs_bi
            #     total_reward += float(np.sum(rewards))
            #     next_state, next_obs_g, next_obs_l = env._get_all_obses_and_state(raw_obs_bi)
            #     steps_builtin += 1
            # else:
            # [GAGENT] Mạng nơ-ron tự quyết định
            actions_g, _ = gagent.get_actions(obs_g)
            active_mask  = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)
            # steps_gagent += 1

            if np.any(active_mask):
                # ── Bước 2a: LAgent quyết định tactical action ────────
                actions_l, _ = lagent.get_actions(obs_l)
                next_state, next_obs_g, next_obs_l, rewards, done, _ = env.step_local(
                    actions_l, active_mask, actions_g
                )
                total_reward += float(np.sum(rewards))
                # steps_local  += 1
            else:
                # ── Bước 2b: Tất cả di chuyển, LAgent không hành động ──
                next_state, next_obs_g, next_obs_l, _, done = env.step_global(actions_g)

            state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
            if done:
                break

        print(
            f"Ep {episode:4d}/{n_episodes} | "
            f"Reward: {total_reward:+.3f} | "
            # f"GAgent: {steps_gagent:3d} steps | "
            # f"LActive: {steps_local:3d} | "
            # f"Builtin: {steps_builtin:3d}"
        )

    print("-" * 60)
    print("Test hoàn tất.")


if __name__ == '__main__':
    main()