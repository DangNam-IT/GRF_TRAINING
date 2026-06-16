import os
import argparse
import multiprocessing as mp
import copy

import numpy as np
import gfootball.env as football_env

from envs.wrapper_ecoma import ECOMA_Wrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger

# =====================================================================
# CUSTOM VECTORIZED ENVIRONMENT FOR E-COMA
# =====================================================================
def worker(remote, parent_remote, env_fn):
    """Tiến trình con quản lý 1 môi trường GFootball độc lập."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                actions = data
                next_state, next_obs, rewards, done, reward_info = env.step(actions)
                if done:
                    # Tự động reset và giữ lại trạng thái cuối (terminal state)
                    state, obs = env.reset()
                    reward_info['terminal_state'] = next_state
                    reward_info['terminal_obs'] = next_obs
                    remote.send((state, obs, rewards, done, reward_info))
                else:
                    remote.send((next_state, next_obs, rewards, done, reward_info))
            elif cmd == 'reset':
                state, obs = env.reset()
                remote.send((state, obs))
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

    def step(self, actions_batch):
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions_batch[i]))
        
        results = [remote.recv() for remote in self.remotes]
        states, obses, rewards, dones, infos = zip(*results)
        
        return (
            np.stack(states),
            np.stack(obses),
            np.stack(rewards),
            np.stack(dones),
            infos  # Trả về list of dicts
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        states, obses = zip(*results)
        return np.stack(states), np.stack(obses)

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
            args_copy.dump_freq = self.args.dump_freq // self.args.num_envs if self.args.dump_freq > 0 else 0
            
        base_env = create_env(args_copy)
        return ECOMA_Wrapper(base_env, num_agents=args_copy.number_agents)

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
        logdir=args.video_dir,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train E-COMA Vectorized")
    parser.add_argument("--num_envs", type=int, default=14, help="Number of parallel environments to run")
    parser.add_argument("--eps", type=int, default=7000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    parser.add_argument("--model_path", type=str, default="experiments/models/E-coma/e_coma", help="Path to save a Agent model")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=100, help="Model save frequency")
    parser.add_argument("--load_model", type=str, default="experiments/models/e_coma", help="Path to load a pre-trained model")
    parser.add_argument("--eps_model", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--log_file", type=str, default="experiments/train_reward/ecoma_train.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/e-coma", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
    return parser.parse_args()


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Thiết lập luồng tối ưu cho PyTorch trên chip AMD EPYC
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    args = parse_args()
    
    # Dummy env to extract dimensions
    dummy_base = create_env(args)
    dummy_env = ECOMA_Wrapper(dummy_base, num_agents=args.number_agents)
    state_dim = dummy_env.state_dim
    obs_dim = dummy_env.obs_dim
    dummy_env.close()

    print(f"Đang khởi tạo {args.num_envs} môi trường song song...")
    vec_env = CustomVecEnv([PicklableEnvFactory(i, args) for i in range(args.num_envs)])

    agent = HES_COMA_Agent(
        state_dim=state_dim,
        obs_dim=obs_dim,
        n_actions=14,
        n_agents=args.number_agents
    )

    if args.load_model != "" and args.eps_model > 0:
        if os.path.exists(args.load_model + '_ep_' + str(args.eps_model) + '.pth'):
            agent.load_model(args.load_model, episode=args.eps_model)
            print(f"Đã tải model từ: {args.load_model} | Episode: {args.eps_model}")
        else:
            print(f"CẢNH BÁO: Không tìm thấy model tại {args.load_model}_ep_{args.eps_model}.pth. Train lại từ đầu.")
    else:
        print("Không tải model")

    buffer = RolloutBuffer()
    logger = CSVLogger(
        args.log_file,
        [
            'Episode', 'R_total', 'R_env', 'R_energy', 
            'R_passing', 'R_in_box', 'R_assist', 'R_role', 'R_approach'
        ]
    )

    print("Bắt đầu huấn luyện E-COMA với Vectorized Environments...")
    
    episodes_completed = 0
    total_reward_accum = np.zeros(args.num_envs)
    ep_R_env = np.zeros(args.num_envs)
    ep_R_energy = np.zeros(args.num_envs)
    ep_R_passing = np.zeros(args.num_envs)
    ep_R_in_box = np.zeros(args.num_envs)
    ep_R_assist = np.zeros(args.num_envs)
    ep_R_role = np.zeros(args.num_envs)
    ep_R_approach = np.zeros(args.num_envs)

    states, obses = vec_env.reset()
    
    import torch
    actor_hidden_states = torch.zeros(args.num_envs * args.number_agents, 128, device=agent.device)

    while episodes_completed < args.eps:
        # Tính Epsilon
        epsilon = max(0.05, 0.5 - 0.45 * (episodes_completed / (args.eps * 0.5)))

        # BATCH INFERENCE
        actions_np, probs_np, next_actor_hidden_states = agent.get_actions(obses, actor_hidden_states, epsilon)
        actions = actions_np

        next_states, next_obses, rewards, dones, infos = vec_env.step(actions)

        # Reset hidden state cho các env đã xong
        for i in range(args.num_envs):
            if dones[i]:
                start_idx = i * args.number_agents
                end_idx = (i + 1) * args.number_agents
                next_actor_hidden_states[start_idx:end_idx] = 0.0

        for i in range(args.num_envs):
            total_reward_accum[i] += np.sum(rewards[i])
            ep_R_env[i]      += infos[i].get('R_env', 0.0)
            ep_R_energy[i]   += infos[i].get('R_energy', 0.0)
            ep_R_passing[i]  += infos[i].get('R_passing', 0.0)
            ep_R_in_box[i]   += infos[i].get('R_in_box', 0.0)
            ep_R_assist[i]   += infos[i].get('R_assist', 0.0)
            ep_R_role[i]     += infos[i].get('R_role', 0.0)
            ep_R_approach[i] += infos[i].get('R_approach', 0.0)

            if dones[i]:
                real_next_state = infos[i]['terminal_state']
                real_next_obs = infos[i]['terminal_obs']
                
                try:
                    buffer.store(states[i], obses[i], actions[i], rewards[i], real_next_state, real_next_obs, dones[i])
                except TypeError:
                    pass
                
                episodes_completed += 1
                logger.log([
                    episodes_completed,
                    total_reward_accum[i],
                    ep_R_env[i],
                    ep_R_energy[i],
                    ep_R_passing[i],
                    ep_R_in_box[i],
                    ep_R_assist[i],
                    ep_R_role[i],
                    ep_R_approach[i]
                ])
                print(f"Episode: {episodes_completed:4d}/{args.eps} | Env Rank: {i} | Eps: {epsilon:.2f} | Reward: {total_reward_accum[i]:.3f} | "
                      f"env={ep_R_env[i]:.2f} energy={ep_R_energy[i]:.2f} pass={ep_R_passing[i]:.2f} "
                      f"box={ep_R_in_box[i]:.2f} asst={ep_R_assist[i]:.2f} role={ep_R_role[i]:.2f} appr={ep_R_approach[i]:.2f}")

                if episodes_completed % args.save_freq_model == 0:
                    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
                    agent.save_model(args.model_path, episode=episodes_completed)

                total_reward_accum[i] = 0.0
                ep_R_env[i] = ep_R_energy[i] = ep_R_passing[i] = ep_R_in_box[i] = ep_R_assist[i] = ep_R_role[i] = ep_R_approach[i] = 0.0
            else:
                try:
                    buffer.store(states[i], obses[i], actions[i], rewards[i], next_states[i], next_obses[i], dones[i])
                except TypeError:
                    pass

        states, obses = next_states, next_obses
        actor_hidden_states = next_actor_hidden_states

        # Tiến hành update model khi buffer đầy dữ liệu tương đương với 1 episode đầy đủ
        if len(buffer.states) >= args.max_steps * args.num_envs:
            agent.update(buffer, num_envs=args.num_envs)
            buffer.clear()

    vec_env.close()
    print("Huấn luyện E-COMA Vectorized hoàn tất. Model đã được lưu.")

if __name__ == '__main__':
    main()