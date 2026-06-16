import os
import argparse
import torch
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
    parser.add_argument("--g_eps",     type=int,   default=250,                               help="Số episodes chạy global agent")
    parser.add_argument("--l_eps",     type=int,   default=100,                               help="Số episodes chạy local agent")
    parser.add_argument("--max_steps",   type=int,   default=300,                               help="Bước tối đa mỗi episode")
    parser.add_argument("--render",      action="store_true",  default=False,                    help="Bật render")
    parser.add_argument("--no_render",   action="store_false", dest="render",                   help="Tắt render")
    parser.add_argument("--g_model",type=str,   default="experiments/models/gagent/gru/g_model_gru",help="Đường dẫn model GAgent")
    parser.add_argument("--l_model",type=str,   default="experiments/models/lagent/model_gru",help="Đường dẫn model LAgent")
    parser.add_argument("--video_dir",   type=str,   default="experiments/videos/phase2/replay",        help="Thư mục video")
    parser.add_argument("--logdir",     type=str,   default="experiments/logs/phase2/replay",          help="Thư mục log")
    parser.add_argument("--dump_freq",   type=int,   default=2,                                 help="Tần suất xuất video (0 = tắt)")

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

    logger = CSVLogger(args.logdir, ['Episode', 'Reward', 'Actor_Loss', 'Critic_Loss'])
    # Inference mode — không cần gradient
    gagent.actor.eval()
    gagent.critic.eval()
    lagent.actor.eval()
    lagent.critic.eval()

    n_episodes = args.g_eps
    m_episodes = args.l_eps
    max_steps  = args.max_steps
    # ── Load models ───────────────────────────────────────────────────
    if os.path.exists(model_path + '_ep_' + str(n_episodes) + '.pth'):
        gagent.load_model(model_path, n_episodes)
        print(f"[GAgent] Đã tải: {model_path}_ep_{n_episodes}.pth")
    else:
        print(f"[GAgent] CẢNH BÁO: Không tìm thấy model global → chạy với trọng số ngẫu nhiên")
    if os.path.exists(lmodel_path + '_ep_' + str(m_episodes) + '.pth') and m_episodes > 0:
        lagent.load_model(lmodel_path, m_episodes)
        print(f"[LAgent] Đã tải: {lmodel_path}_ep_{m_episodes}.pth")
    else:
        print(f"[LAgent] CẢNH BÁO: Không tìm thấy model local → chạy với trọng số ngẫu nhiên")
    STOP_ACTIONS = set(env.STOP_ACTIONS)

    for episode in range(1, n_episodes + 1):
        state, obs_g, obs_l = env.reset()
        total_reward   = 0.0
        total_reward_env = 0.0
        total_assist = 0
        total_role = 0
        total_pass = 0
        total_inbox = 0
        total_approach =0
        # Khởi tạo bộ nhớ (hidden states) cho cả 2 mạng ở thời điểm bắt đầu Game
        gagent_hidden_states = torch.zeros(1 * 11, 128, device=gagent.device)
        lagent_hidden_states = torch.zeros(1 * 11, 128, device=lagent.device)

        for t in range(max_steps):

            # [GAGENT] Mạng nơ-ron tự quyết định
            obs_g_batch = np.expand_dims(obs_g, axis=0)
            actions_g_batch, _, next_gagent_hidden_states = gagent.get_actions(obs_g_batch, gagent_hidden_states, epsilon=0.0)
            actions_g = actions_g_batch[0]
            
            active_mask  = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)
            # [LAGENT] LAgent quyết định tactical action 
            obs_l_batch = np.expand_dims(obs_l, axis=0)
            actions_l_batch, _, next_lagent_hidden_states = lagent.get_actions(obs_l_batch, lagent_hidden_states, epsilon=0.0)
            actions_l = actions_l_batch[0]
            
            next_state, next_obs_g, next_obs_l, rewards, done, info = env.step_local(
                actions_l, active_mask, actions_g
            )
            total_reward += float(np.sum(rewards))
            total_reward_env += info['R_env']
            total_assist += info['R_assist']
            total_role += info['R_role']
            total_approach += info['R_approach']
            total_pass += info['R_passing']
            total_inbox += info['R_in_box']

            state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
            
            # Cập nhật bộ nhớ sang timestep tiếp theo
            gagent_hidden_states = next_gagent_hidden_states
            lagent_hidden_states = next_lagent_hidden_states
            
            if done:
                break
        logger.log([episode, total_reward, total_reward_env, total_assist, total_role, total_approach, total_pass, total_inbox])
        print(
            f"Ep {episode:4d}/{n_episodes} | "
            f"Reward: {total_reward:+.3f} | "
            f"R_env: {total_reward_env:.3f} | "
            f"R_assist: {total_assist:.3f} | "
            f"R_role: {total_role:.3f} |"
            f"R_approach: {total_approach:.3f} | "
            f"R_passing: {total_pass:.3f} | "
            f"R_in_box: {total_inbox:.3f}"
        )
    print("-" * 60)
    print("Test hoàn tất.")
if __name__ == '__main__':
    main()