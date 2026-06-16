# import os
# import argparse

# import numpy as np
# import gfootball.env as football_env

# from envs.wrapper_local import GFootballLocalWrapper
# from agents.HES_COMA import HES_COMA_Agent
# from utils.buffer import RolloutBuffer
# from utils.logger import CSVLogger


# def create_env(args):
#     return football_env.create_environment(
#         env_name="academy_corner",
#         number_of_left_players_agent_controls=args.number_agents,
#         representation='raw',
#         rewards='scoring',
#         render=args.render,
#         write_full_episode_dumps=(args.dump_freq > 0),
#         dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,
#         logdir=args.video_dir,
#         other_config_options={'action_set': 'v2'},
#     )

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train Phase 2 (LAgent)")
#     parser.add_argument("--eps", type=int, default=7000, help="Number of episodes to train")
#     parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
#     parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
#     parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent/test1/g_model", help="Path to load GAgent model")
#     parser.add_argument("--lagent_model", type=str, default="experiments/models/lagent/H_lagent/l_model", help="Path to save LAgent model")
#     parser.add_argument("--pre_lagent", type=str, default="experiments/models/lagent/test5/l_model", help="Path to load a pre-trained LAgent model")

#     parser.add_argument("--g_eps",     type=int,   default=0,                               help="Number eps to run global agent")
#     parser.add_argument("--l_eps",     type=int,   default=0,                               help="Number episodes to run local agent")
#     parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
#     parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
#     parser.add_argument("--save_freq_model", type=int, default=100, help="Model save frequency")
#     parser.add_argument("--log_file", type=str, default="experiments/train_reward/HCOMA_train.csv", help="Path to the log CSV file")
#     parser.add_argument("--video_dir", type=str, default="experiments/videos/H_train", help="Directory to save videos")
#     parser.add_argument("--dump_freq", type=int, default=1, help="Video dump frequency (0 to disable)")
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     base_env = create_env(args)
#     env      = GFootballLocalWrapper(base_env, num_agents=args.number_agents)

#     # ── GAgent: tải model đã frozen từ Phase 1 ───────────────────────────────
#     model_path = args.gagent_model
#     g_eps_model   = args.g_eps
#     gagent = HES_COMA_Agent(
#         state_dim=env.state_dim,
#         obs_dim=env.obs_dim_g,
#         n_actions=10,
#         n_agents=args.number_agents
#     )

#     lagent = HES_COMA_Agent(
#         state_dim=env.state_dim,
#         obs_dim=env.obs_dim_l,          # 48 chiều (ray-info 16×3, không energy)
#         n_actions=env.n_tactic_actions,  # 2 tactical actions
#         n_agents=args.number_agents
#     )

#     n_episodes = args.eps
#     max_steps  = args.max_steps

#     for param in gagent.actor.parameters():
#         param.requires_grad = False
#     for param in gagent.critic.parameters():
#         param.requires_grad = False

#     if os.path.exists(model_path + '_ep_' + str(g_eps_model) +'.pth'):
#         gagent.load_model(model_path, g_eps_model)
#     else:
#         print(f"CẢNH BÁO: Không tìm thấy {model_path}. GAgent chạy với trọng số ngẫu nhiên!")

#     if os.path.exists(args.pre_lagent + '_ep_' + str(args.l_eps) +'.pth') and args.l_eps > 0:
#         lagent.load_model(args.pre_lagent, episode=args.l_eps)
#         print(f"Loaded LAgent từ ep {args.l_eps} của {args.pre_lagent}")
#     else:
#         print(f"CẢNH BÁO: Không tìm thấy pre-trained {args.pre_lagent}. LAgent chạy với trọng số ngẫu nhiên!")

 
#     buffer = RolloutBuffer()
#     logger = CSVLogger(
#         args.log_file,
#         [
#             'Episode',
#             'Total_Local_Reward',
#             'R_env',
#             'R_passing',
#             'R_in_box',
#             'R_assist',
#             'R_role',
#             'R_approach',
#         ]
#     )

#     # ── GAgent "stop" actions — output 8 hoặc 9 → GRF idle ──────────────────
#     # Trong wrapper_global: actions 0-7 → GRF 1-8 (di chuyển), 8-9 → GRF 0, 14 (stop).
#     STOP_ACTIONS = set(env.STOP_ACTIONS)

#     print("Bắt đầu huấn luyện Phase 2 (Local Agent)...")
#     for episode in range(1, n_episodes + 1):
#         state, obs_g, obs_l = env.reset()
#         buffer.clear()
#         total_reward = 0.0
#         ep_R_env        = 0.0
#         ep_R_passing    = 0.0
#         ep_R_in_box     = 0.0
#         ep_R_assist     = 0.0
#         ep_R_role       = 0.0
#         ep_R_approach   = 0.0


#         for t in range(max_steps):
#             # ── Bước 1: GAgent (frozen) ra quyết định ────────────────────────
#             actions_g, _ = gagent.get_actions(obs_g)

#             # ── Bước 2: Xác định agent nào chọn STOP ─────────────────────────
#             # active_mask[i] = True → GAgent[i] dừng → LAgent[i] được kích hoạt
#             active_mask = np.array([int(a) in STOP_ACTIONS for a in actions_g], dtype=bool)

#             # ── Bước 3a: LAgent quyết định tactical action ───────────────
#             actions_l, _ = lagent.get_actions(obs_l)

#             next_state, next_obs_g, next_obs_l, rewards, done, reward_info = env.step_local(
#                 actions_l, active_mask, actions_g
#             )

#             # Lưu vào buffer kèm active_mask
#             # Buffer lưu (s_t, o_t^l, a_t^l, r_t^l, s_{t+1}, o_{t+1}^l, d, mask)
#             buffer.store(state, obs_l, actions_l, rewards, next_state, next_obs_l, done, active_mask)
#             total_reward += np.sum(rewards)

#             # Tích lũy từng phần thưởng thành phần
#             ep_R_env        += reward_info['R_env']
#             ep_R_passing    += reward_info['R_passing']
#             ep_R_in_box     += reward_info['R_in_box']
#             ep_R_assist     += reward_info['R_assist']
#             ep_R_role       += reward_info['R_role']
#             ep_R_approach   += reward_info['R_approach']

#             state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
#             if done:
#                 break

#         # ── Cập nhật LAgent sau mỗi episode ───────────────────────────────
#         lagent.update(buffer)

#         logger.log([
#             episode,
#             total_reward,
#             ep_R_env,
#             ep_R_passing,
#             ep_R_in_box,
#             ep_R_assist,
#             ep_R_role,
#             ep_R_approach,
#         ])
#         print(f"Episode: {episode:4d}/{n_episodes} | "
#               f"Reward: {total_reward:.3f} | "
#               f"env={ep_R_env:.2f} pass={ep_R_passing:.2f} "
#               f"box={ep_R_in_box:.2f} "
#               f"asst={ep_R_assist:.2f} role={ep_R_role:.2f} "
#               f"appr={ep_R_approach:.2f} ")

#         if episode % args.save_freq_model == 0:
#             # Lưu model định kỳ
#             os.makedirs(os.path.dirname(args.lagent_model), exist_ok=True)
#             lagent.save_model(args.lagent_model, episode=episode)

#     print("Huấn luyện Phase 2 hoàn tất. Model LAgent đã được lưu.")


# if __name__ == '__main__':
#     main()


import os
import argparse
import copy
import multiprocessing as mp
import glob
import re

import numpy as np
import gfootball.env as football_env

from envs.wrapper_local import GFootballLocalWrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger


# =====================================================================
# CUSTOM VECTORIZED ENVIRONMENT
# Tự động đồng bộ các môi trường con mà không cần cài đặt thêm thư viện
# =====================================================================
def worker(remote, parent_remote, env_fn):
    """Tiến trình con quản lý 1 môi trường GFootball độc lập."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step_local':
                actions_l, active_mask, actions_g = data
                next_state, next_obs_g, next_obs_l, rewards, done, reward_info = env.step_local(
                    actions_l, active_mask, actions_g
                )
                if done:
                    # Tự động reset và giữ lại trạng thái cuối (terminal state)
                    state, obs_g, obs_l = env.reset()
                    reward_info['terminal_state'] = next_state
                    reward_info['terminal_obs_g'] = next_obs_g
                    reward_info['terminal_obs_l'] = next_obs_l
                    remote.send((state, obs_g, obs_l, rewards, done, reward_info))
                else:
                    remote.send((next_state, next_obs_g, next_obs_l, rewards, done, reward_info))
            elif cmd == 'reset':
                state, obs_g, obs_l = env.reset()
                remote.send((state, obs_g, obs_l))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker Error: {e}")

class CustomVecEnv:
    """Quản lý giao tiếp và thu thập Batch Observation từ các tiến trình con."""
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = [
            mp.Process(target=worker, args=(work_remote, remote, env_fn))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_local(self, actions_l_batch, active_mask_batch, actions_g_batch):
        for i, remote in enumerate(self.remotes):
            remote.send(('step_local', (actions_l_batch[i], active_mask_batch[i], actions_g_batch[i])))
        
        results = [remote.recv() for remote in self.remotes]
        states, obs_gs, obs_ls, rewards, dones, infos = zip(*results)
        
        return (
            np.stack(states),
            np.stack(obs_gs),
            np.stack(obs_ls),
            np.stack(rewards),
            np.stack(dones),
            infos  # Trả về list of dicts
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        states, obs_gs, obs_ls = zip(*results)
        return np.stack(states), np.stack(obs_gs), np.stack(obs_ls)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

class PicklableEnvFactory:
    """Lớp toàn cục giúp đóng gói cấu hình môi trường, tương thích 100% với cơ chế spawn."""
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args

    def __call__(self):
        import copy
        args_copy = copy.copy(self.args)
        # Ép buộc tắt render đồ họa trên Cloud để bảo đảm an toàn nghẽn luồng
        args_copy.render = False 
        
        if self.rank > 0:
            args_copy.dump_freq = 0
        else:
            args_copy.dump_freq = self.args.dump_freq // self.args.num_envs 
            
        base_env = create_env(args_copy)
        return GFootballLocalWrapper(base_env, num_agents=args_copy.number_agents)

# =====================================================================
# MAIN TRAINING LOOP (VECTORIZED)
# =====================================================================
def create_env(args):
    return football_env.create_environment(
        env_name="academy_corner",
        number_of_left_players_agent_controls=args.number_agents,
        representation='raw',
        rewards='scoring',
        render=args.render,
        write_goal_dumps=(args.dump_freq > 0),
        write_full_episode_dumps=(args.dump_freq > 0),
        dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,
        logdir=args.video_dir,
        other_config_options={'action_set': 'v2'}
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 (LAgent) - Vectorized")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to run")
    parser.add_argument("--eps", type=int, default=300, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent/gru/g_model_gru", help="Path to load GAgent model")
    parser.add_argument("--lagent_model", type=str, default="experiments/models/lagent/model_gru", help="Path to save LAgent model")
    parser.add_argument("--pre_lagent", type=str, default="experiments/models/", help="Path to load a pre-trained LAgent model")

    parser.add_argument("--g_eps", type=int, default=250, help="Number eps to run global agent")
    parser.add_argument("--l_eps", type=int, default=0, help="Number episodes to run local agent")
    parser.add_argument("--render", action="store_true", default=False, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=50, help="Model save frequency")
    parser.add_argument("--log_file", type=str, default="experiments/train_reward/l_train.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase2/gru", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=3, help="Video dump frequency (0 to disable)")
    return parser.parse_args()


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Thiết lập luồng tối ưu cho PyTorch trên chip AMD EPYC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    args = parse_args()

    # Tạo 1 env ảo (dummy) để trích xuất các thông số kích thước không gian (dim)
    dummy_base = create_env(args)
    dummy_env = GFootballLocalWrapper(dummy_base, num_agents=args.number_agents)
    state_dim = dummy_env.state_dim
    obs_dim_g = dummy_env.obs_dim_g
    obs_dim_l = dummy_env.obs_dim_l
    n_tactic_actions = dummy_env.n_tactic_actions
    STOP_ACTIONS = set(dummy_env.STOP_ACTIONS)
    dummy_env.close()

    print(f"Đang khởi tạo {args.num_envs} môi trường song song...")
    vec_env = CustomVecEnv([PicklableEnvFactory(i, args) for i in range(args.num_envs)])
    
    # Khởi tạo Mô hình
    gagent = HES_COMA_Agent(state_dim=state_dim, obs_dim=obs_dim_g, n_actions=10, n_agents=args.number_agents, lr=0.00025)
    lagent = HES_COMA_Agent(state_dim=state_dim, obs_dim=obs_dim_l, n_actions=n_tactic_actions, n_agents=args.number_agents)

    for param in gagent.actor.parameters():
        param.requires_grad = False
    for param in gagent.critic.parameters():
        param.requires_grad = False

    # 1. Load GAgent từ Phase 1
    if os.path.exists(args.gagent_model + '_ep_' + str(args.g_eps) + '.pth'):
        gagent.load_model(args.gagent_model, args.g_eps)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {args.gagent_model}. GAgent chạy ngẫu nhiên!")

    # 2. Cơ chế TỰ ĐỘNG KHÔI PHỤC (AUTO-RESUME) thông minh cho LAgent
    start_episode = 0
    model_dir = os.path.dirname(args.lagent_model)
    
    # Kiểm tra xem thư mục lưu mô hình đã tồn tại và có chứa file checkpoint cũ chưa
    if os.path.exists(model_dir):
        checkpoint_files = glob.glob(os.path.join(model_dir, "*_ep_*.pth"))
        if checkpoint_files:
            # Quét tìm số episode lớn nhất đã được lưu
            episodes = [int(re.findall(r'_ep_(\d+)\.pth', f)[0]) for f in checkpoint_files if re.findall(r'_ep_(\d+)\.pth', f)]
            if episodes:
                start_episode = max(episodes)
                base_name = os.path.basename(args.lagent_model)
                args.pre_lagent = os.path.join(model_dir, base_name)

    # Tiến hành nạp trọng số tương ứng từ mốc lớn nhất tìm thấy
    if start_episode > 0:
        lagent.load_model(args.pre_lagent, episode=start_episode)
        print(f"==> [TỰ ĐỘNG KHÔI PHỤC] Đã tìm thấy và tải bộ não mới nhất tại Episode: {start_episode}")
    else:
        # Nếu người dùng cố tình truyền tham số l_eps qua terminal (như dự phòng)
        if args.l_eps > 0 and os.path.exists(args.pre_lagent + '_ep_' + str(args.l_eps) + '.pth'):
            lagent.load_model(args.pre_lagent, episode=args.l_eps)
            start_episode = args.l_eps
            print(f"Loaded LAgent theo chỉ định terminal tại ep {args.l_eps}")
        else:
            print(f"CẢNH BÁO: Không tìm thấy checkpoint cũ nào. LAgent sẽ học hoàn toàn mới!")

    buffer = RolloutBuffer()
    logger = CSVLogger(
        args.log_file,
        ['Episode', 'Total_Local_Reward', 'R_env', 'R_passing', 'R_in_box', 'R_assist', 'R_role', 'R_approach']
    )

    print("Bắt đầu huấn luyện Phase 2 (Local Agent) với Vectorized Environments...")
    
    # Gán biến đếm bằng start_episode thay vì 0 để tiếp tục đếm tiếp
    episodes_completed = start_episode  
    
    total_reward_accum = np.zeros(args.num_envs)
    ep_R_env = np.zeros(args.num_envs)
    ep_R_passing = np.zeros(args.num_envs)
    ep_R_in_box = np.zeros(args.num_envs)
    ep_R_assist = np.zeros(args.num_envs)
    ep_R_role = np.zeros(args.num_envs)
    ep_R_approach = np.zeros(args.num_envs)

    states, obs_gs, obs_ls = vec_env.reset()

    import torch
    # Khởi tạo bộ nhớ cho cả GAgent và LAgent
    gagent_hidden_states = torch.zeros(args.num_envs * args.number_agents, 128, device=gagent.device)
    lagent_hidden_states = torch.zeros(args.num_envs * args.number_agents, 128, device=lagent.device)

    # Vòng lặp liên tục cho đến khi thu thập đủ tổng số episodes
    while episodes_completed < args.eps:
        # Tính Epsilon cho LAgent (GAgent giữ bằng 0 vì đã bị freeze)
        epsilon = max(0.05, 0.5 - 0.45 * (episodes_completed / (args.eps * 0.5)))
        
        # BATCH INFERENCE GAgent
        obs_g_flat = obs_gs.reshape(args.num_envs * args.number_agents, -1)
        actions_g_flat, _, next_gagent_hidden_states = gagent.get_actions(obs_g_flat, gagent_hidden_states, epsilon=0.0)
        actions_g = actions_g_flat

        active_mask = np.isin(actions_g, list(STOP_ACTIONS))

        # BATCH INFERENCE LAgent
        obs_l_flat = obs_ls.reshape(args.num_envs * args.number_agents, -1)
        actions_l_flat, _, next_lagent_hidden_states = lagent.get_actions(obs_l_flat, lagent_hidden_states, epsilon=epsilon)
        actions_l = actions_l_flat

        # Chạy step song song cho tất cả các envs
        next_states, next_obs_gs, next_obs_ls, rewards, dones, infos = vec_env.step_local(
            actions_l, active_mask, actions_g
        )

        # Reset hidden state cho các env đã hoàn thành
        for i in range(args.num_envs):
            if dones[i]:
                start_idx = i * args.number_agents
                end_idx = (i + 1) * args.number_agents
                next_gagent_hidden_states[start_idx:end_idx] = 0.0
                next_lagent_hidden_states[start_idx:end_idx] = 0.0

        for i in range(args.num_envs):
            total_reward_accum[i] += np.sum(rewards[i])
            ep_R_env[i] += infos[i].get('R_env', 0.0)
            ep_R_passing[i] += infos[i].get('R_passing', 0.0)
            ep_R_in_box[i] += infos[i].get('R_in_box', 0.0)
            ep_R_assist[i] += infos[i].get('R_assist', 0.0)
            ep_R_role[i] += infos[i].get('R_role', 0.0)
            ep_R_approach[i] += infos[i].get('R_approach', 0.0)

            # Xử lý tự động Reset (Auto-Reset)
            if dones[i]:
                # Sử dụng terminal state/obs cho vòng lặp cuối cùng của tập phim
                real_next_state = infos[i]['terminal_state']
                real_next_obs_l = infos[i]['terminal_obs_l']
                
                try:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], real_next_state, real_next_obs_l, dones[i], active_mask[i])
                except TypeError:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], real_next_state, real_next_obs_l, dones[i])
                
                episodes_completed += 1
                logger.log([
                    episodes_completed, total_reward_accum[i], ep_R_env[i], 
                    ep_R_passing[i], ep_R_in_box[i], ep_R_assist[i], ep_R_role[i], ep_R_approach[i]
                ])
                print(f"Episode: {episodes_completed:4d}/{args.eps} | Env Rank: {i} | Eps: {epsilon:.2f} | Reward: {total_reward_accum[i]:.3f} | "
                      f"env={ep_R_env[i]:.2f} pass={ep_R_passing[i]:.2f} box={ep_R_in_box[i]:.2f} "
                      f"asst={ep_R_assist[i]:.2f} role={ep_R_role[i]:.2f} appr={ep_R_approach[i]:.2f}")

                # Lưu model định kỳ
                if episodes_completed % args.save_freq_model == 0:
                    os.makedirs(os.path.dirname(args.lagent_model), exist_ok=True)
                    lagent.save_model(args.lagent_model, episode=episodes_completed)

                # Reset bộ đếm phần thưởng
                total_reward_accum[i] = 0.0
                ep_R_env[i] = ep_R_passing[i] = ep_R_in_box[i] = ep_R_assist[i] = ep_R_role[i] = ep_R_approach[i] = 0.0

            else:
                try:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], next_states[i], next_obs_ls[i], dones[i], active_mask[i])
                except TypeError:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], next_states[i], next_obs_ls[i], dones[i])

        states, obs_gs, obs_ls = next_states, next_obs_gs, next_obs_ls
        gagent_hidden_states = next_gagent_hidden_states
        lagent_hidden_states = next_lagent_hidden_states

        # Tiến hành update model khi buffer đầy dữ liệu
        if len(buffer.states) >= args.max_steps * args.num_envs:
            lagent.update(buffer, num_envs=args.num_envs)
            buffer.clear()

    vec_env.close()
    print("Huấn luyện Phase 2 Vectorized hoàn tất. Model LAgent đã được lưu.")

if __name__ == '__main__':
    main()