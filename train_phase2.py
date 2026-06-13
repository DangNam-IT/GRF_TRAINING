import os
import argparse
import copy
import multiprocessing as mp

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
        write_full_episode_dumps=(args.dump_freq > 0),
        dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,
        logdir=args.video_dir
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 (LAgent) - Vectorized")
    parser.add_argument("--num_envs", type=int, default=14, help="Number of parallel environments to run")
    parser.add_argument("--eps", type=int, default=7000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    parser.add_argument("--gagent_model", type=str, default="experiments/models/gagent/test1/g_model", help="Path to load GAgent model")
    parser.add_argument("--lagent_model", type=str, default="experiments/models/lagent/l_model", help="Path to save LAgent model")
    parser.add_argument("--pre_lagent", type=str, default="experiments/models/lagent/test1/l_model", help="Path to load a pre-trained LAgent model")

    parser.add_argument("--g_eps", type=int, default=400, help="Number eps to run global agent")
    parser.add_argument("--l_eps", type=int, default=1400, help="Number episodes to run local agent")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=100, help="Model save frequency")
    parser.add_argument("--log_file", type=str, default="experiments/reward_data/l_train_vec.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase2/test1", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Tạo 1 env ảo (dummy) để trích xuất các thông số kích thước không gian (dim)
    dummy_base = create_env(args)
    dummy_env = GFootballLocalWrapper(dummy_base, num_agents=args.number_agents)
    state_dim = dummy_env.state_dim
    obs_dim_g = dummy_env.obs_dim_g
    obs_dim_l = dummy_env.obs_dim_l
    # g_actions = dummy_env.action_space
    n_tactic_actions = dummy_env.n_tactic_actions
    STOP_ACTIONS = set(dummy_env.STOP_ACTIONS)
    dummy_env.close()

    # Định nghĩa factory function để sinh ra các env con cho Vectorized Env
    def make_env_fn(rank):
        def _init():
            args_copy = copy.copy(args)
            # Chỉ render ở luồng 0 để tránh mở quá nhiều cửa sổ
            if rank > 0:
                args_copy.render = False
                args_copy.dump_freq = 0
            else:
                args_copy.dump_freq = args.dump_freq // args.num_envs 
            base_env = create_env(args_copy)
            return GFootballLocalWrapper(base_env, num_agents=args_copy.number_agents)
        return _init

    print(f"Đang khởi tạo {args.num_envs} môi trường song song...")
    vec_env = CustomVecEnv([make_env_fn(i) for i in range(args.num_envs)])

    # Khởi tạo Mô hình
    gagent = HES_COMA_Agent(state_dim=state_dim, obs_dim=obs_dim_g, n_actions=10, n_agents=args.number_agents)
    lagent = HES_COMA_Agent(state_dim=state_dim, obs_dim=obs_dim_l, n_actions=n_tactic_actions, n_agents=args.number_agents)

    for param in gagent.actor.parameters():
        param.requires_grad = False
    for param in gagent.critic.parameters():
        param.requires_grad = False

    if os.path.exists(args.gagent_model + '_ep_' + str(args.g_eps) + '.pth'):
        gagent.load_model(args.gagent_model, args.g_eps)
    else:
        print(f"CẢNH BÁO: Không tìm thấy {args.gagent_model}. GAgent chạy ngẫu nhiên!")

    if os.path.exists(args.pre_lagent + '_ep_' + str(args.l_eps) + '.pth') and args.l_eps > 0:
        lagent.load_model(args.pre_lagent, episode=args.l_eps)
        print(f"Loaded LAgent từ ep {args.l_eps}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy pre-trained LAgent. LAgent chạy ngẫu nhiên!")

    buffer = RolloutBuffer()
    logger = CSVLogger(
        args.log_file,
        ['Episode', 'Total_Local_Reward', 'R_env', 'R_passing', 'R_in_box', 'R_assist', 'R_role', 'R_approach']
    )

    print("Bắt đầu huấn luyện Phase 2 (Local Agent) với Vectorized Environments...")
    
    # Biến kiểm soát số lượng episodes đã hoàn tất toàn cục
    episodes_completed = 0
    total_reward_accum = np.zeros(args.num_envs)
    ep_R_env = np.zeros(args.num_envs)
    ep_R_passing = np.zeros(args.num_envs)
    ep_R_in_box = np.zeros(args.num_envs)
    ep_R_assist = np.zeros(args.num_envs)
    ep_R_role = np.zeros(args.num_envs)
    ep_R_approach = np.zeros(args.num_envs)

    states, obs_gs, obs_ls = vec_env.reset()

    # Vòng lặp liên tục cho đến khi thu thập đủ tổng số episodes
    while episodes_completed < args.eps:
        
        # BATCH INFERENCE GAgent
        obs_g_flat = obs_gs.reshape(args.num_envs * args.number_agents, -1)
        actions_g_flat, _ = gagent.get_actions(obs_g_flat)
        actions_g = actions_g_flat.reshape(args.num_envs, args.number_agents)

        active_mask = np.isin(actions_g, list(STOP_ACTIONS))

        # BATCH INFERENCE LAgent
        obs_l_flat = obs_ls.reshape(args.num_envs * args.number_agents, -1)
        actions_l_flat, _ = lagent.get_actions(obs_l_flat)
        actions_l = actions_l_flat.reshape(args.num_envs, args.number_agents)

        # Chạy step song song cho tất cả các envs
        next_states, next_obs_gs, next_obs_ls, rewards, dones, infos = vec_env.step_local(
            actions_l, active_mask, actions_g
        )

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
                
                # Lưu vào buffer (Lưu ý: Bạn có truyền `active_mask` trong `store()` tùy thuộc chữ ký của `RolloutBuffer` hiện tại)
                try:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], real_next_state, real_next_obs_l, dones[i], active_mask[i])
                except TypeError:
                    buffer.store(states[i], obs_ls[i], actions_l[i], rewards[i], real_next_state, real_next_obs_l, dones[i])
                
                episodes_completed += 1
                logger.log([
                    episodes_completed, total_reward_accum[i], ep_R_env[i], 
                    ep_R_passing[i], ep_R_in_box[i], ep_R_assist[i], ep_R_role[i], ep_R_approach[i]
                ])
                print(f"Episode: {episodes_completed:4d}/{args.eps} | Env Rank: {i} | Reward: {total_reward_accum[i]:.3f} | "
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

        # Tiến hành update model khi buffer đầy dữ liệu tương đương với 1 episode đầy đủ
        # Có thể tuỳ chỉnh cập nhật sau X step thay vì đợi cả episode.
        if len(buffer.states) >= args.max_steps * args.num_envs:
            lagent.update(buffer)
            buffer.clear()

    vec_env.close()
    print("Huấn luyện Phase 2 Vectorized hoàn tất. Model LAgent đã được lưu.")

if __name__ == '__main__':
    main()