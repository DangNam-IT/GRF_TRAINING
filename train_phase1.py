# from __future__ import annotations

# import os
# import argparse
# import random
# from typing import Any

# import numpy as np
# from numpy.typing import NDArray
# import gfootball.env as football_env
# from gym import Env

# from envs.wrapper_global import GFootballGlobalWrapper
# from agents.HES_COMA import HES_COMA_Agent
# from pygame import math
# from utils.buffer import RolloutBuffer
# from utils.logger import CSVLogger


# def create_env(args: argparse.Namespace) -> Env:
#     """Tạo môi trường GRF cho Phase 1 (không render để tăng tốc training)."""
#     return football_env.create_environment(
#         env_name="academy_corner",
#         number_of_left_players_agent_controls=11,
#         representation="raw",
#         rewards="scoring",
#         render=args.render,  # Bật render để xem quá trình huấn luyện
#         write_full_episode_dumps=(args.dump_freq > 0),
#         dump_frequency=args.dump_freq if args.dump_freq > 0 else 1,  # Lưu video mỗi 100 episode
#         logdir=args.video_dir,
#         # other_config_options={'action_set': 'v2'},  # Sử dụng action set v2 để có action_builtin_ai
#     )

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Train Phase 1 (GAgent)")
#     parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes to train")
#     parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
#     parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
#     parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
#     parser.add_argument("--save_freq_model", type=int, default=100, help="Model save dump frequency")
#     parser.add_argument("--model_path", type=str, default="experiments/models/gagent/test2/g_model", help="Path to save the model")
#     parser.add_argument("--log_file", type=str, default="experiments/test2/g_train.csv", help="Path to the log CSV file")
#     parser.add_argument("--video_dir", type=str, default="experiments/videos/phase1/test2", help="Directory to save videos")
#     parser.add_argument("--dump_freq", type=int, default=0, help="Video dump frequency (0 to disable)")
#     parser.add_argument("--load_model", type=str, default="experiments/models/gagent/test2/g_model", help="Path to load a pre-trained model")
#     parser.add_argument("--load_eps", type=int, default=0, help="Episode number of the loaded model")
#     parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     base_env: Env                   = create_env(args)
#     env:      GFootballGlobalWrapper = GFootballGlobalWrapper(base_env, num_agents=args.number_agents)
#     agent:    HES_COMA_Agent         = HES_COMA_Agent(
#         state_dim=env.state_dim,
#         obs_dim=env.obs_dim,
#         n_actions=10,
#         n_agents=args.number_agents,
#     )
    
#     if args.load_model != "" and args.load_eps > 0:
#         agent.load_model(args.load_model, episode=args.load_eps)
#         print(f"Đã tải model từ: {args.load_model} | Episode: {args.load_eps}")
    
#     buffer: RolloutBuffer = RolloutBuffer()
#     logger: CSVLogger     = CSVLogger(
#         args.log_file,
#         [
#             "Episode", 
#             "R_total",
#             "R_energy",
#             "R_handover"
#         ],
#     )

#     n_episodes: int = args.episodes
#     max_steps:  int = args.max_steps

#     print("Bắt đầu huấn luyện Phase 1 (Global Agent)...")
#     for episode in range(1, n_episodes + 1):
#         state: NDArray[np.float32]
#         obses: NDArray[np.float32]
#         state, obses = env.reset()
#         buffer.clear()
       

#         total_reward:       float = 0.0
#         total_energy:       float = 0.0
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
#             total_energy     += float(np.sum(rewards_view["r_energy"]))
#             total_handover   += float(np.sum(rewards_view["r_handover"]))
            
#             state, obses = next_state, next_obses
#             if done:
#                 break

#         agent.update(buffer)
#         logger.log([episode, total_reward, total_energy, total_handover])
#         print(
#             f"Episode: {episode}/{n_episodes} | "
#             f"Total: {total_reward:.3f} | "
#             f"Energy: {total_energy:.3f} | "
#             f"Handover: {total_handover:.3f}"
#         )
    
#         if episode % args.save_freq_model == 0:
#             os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
#             agent.save_model(args.model_path, episode=episode)
#     print("Huấn luyện hoàn tất. Đã lưu mô hình GAgent cùng cấu hình môi trường.")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import os
import argparse
import glob
import re
import multiprocessing as mp
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gfootball.env as football_env
from gym import Env
import torch

from envs.wrapper_global import GFootballGlobalWrapper
from agents.HES_COMA import HES_COMA_Agent
from pygame import math
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
            if cmd == 'step':
                actions = data
                next_state, next_obses, rewards, rewards_view, done = env.step(actions)
                if done:
                    # Tự động reset và giữ lại trạng thái cuối (terminal state)
                    state, obses = env.reset()
                    remote.send((state, obses, rewards, rewards_view, done, next_state, next_obses))
                else:
                    remote.send((next_state, next_obses, rewards, rewards_view, done, None, None))
            elif cmd == 'reset':
                state, obses = env.reset()
                remote.send((state, obses))
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
        states, obses, rewards, rewards_views, dones, term_states, term_obses = zip(*results)
        
        return (
            np.stack(states),
            np.stack(obses),
            np.stack(rewards),
            rewards_views,  # list of dicts
            np.stack(dones),
            term_states,    # tuple of (None or array)
            term_obses      # tuple of (None or array)
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
        return GFootballGlobalWrapper(base_env, num_agents=args_copy.number_agents)


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
        logdir=args.video_dir,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 (GAgent) - Vectorized")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to run")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", default=False, help="Enable rendering")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--save_freq_model", type=int, default=50, help="Model save dump frequency")
    parser.add_argument("--model_path", type=str, default="experiments/models/gagent/gru/g_model", help="Path to save the model")
    parser.add_argument("--log_file", type=str, default="experiments/g_train.csv", help="Path to the log CSV file")
    parser.add_argument("--video_dir", type=str, default="experiments/videos/phase1", help="Directory to save videos")
    parser.add_argument("--dump_freq", type=int, default=10, help="Video dump frequency (0 to disable)")
    # parser.add_argument("--load_model", type=str, default="experiments/models/gagent/gru/g_model", help="Path to load a pre-trained model")
    parser.add_argument("--load_eps", type=int, default=0, help="Episode number of the loaded model")
    parser.add_argument("--number_agents", type=int, default=11, help="Number of agents in the environment")
    # parser.add_argument("--builtin_eps", type=float, default=0.0,
    #     help="Xác suất [0.0–1.0] dùng action_builtin_ai=19 thay vì GAgent. "
    #          "0.0 = tắt (chỉ dùng GAgent); 1.0 = chỉ dùng Built-in AI")
    return parser.parse_args()


def main() -> None:
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
    dummy_env = GFootballGlobalWrapper(dummy_base, num_agents=args.number_agents)
    state_dim = dummy_env.state_dim
    obs_dim = dummy_env.obs_dim
    dummy_env.close()

    print(f"Đang khởi tạo {args.num_envs} môi trường song song...")
    vec_env = CustomVecEnv([PicklableEnvFactory(i, args) for i in range(args.num_envs)])

    agent: HES_COMA_Agent = HES_COMA_Agent(
        state_dim=state_dim,
        obs_dim=obs_dim,
        n_actions=10,
        n_agents=args.number_agents,
    )
    
   # 2. Cơ chế TỰ ĐỘNG KHÔI PHỤC (AUTO-RESUME) thông minh cho LAgent
    start_episode = 0
    model_dir = os.path.dirname(args.model_path)
    
    # Kiểm tra xem thư mục lưu mô hình đã tồn tại và có chứa file checkpoint cũ chưa
    if os.path.exists(model_dir):
        checkpoint_files = glob.glob(os.path.join(model_dir, "*_ep_*.pth"))
        if checkpoint_files:
            episodes = [int(re.findall(r'_ep_(\d+)\.pth', f)[0]) for f in checkpoint_files if re.findall(r'_ep_(\d+)\.pth', f)]
            if episodes:
                start_episode = max(episodes)
                print(episodes)
                base_name = os.path.basename(args.model_path)

    # Tiến hành nạp trọng số tương ứng từ mốc lớn nhất tìm thấy
    if start_episode > 0:
        agent.load_model(args.model_path, episode=start_episode)
        print(f"==> [TỰ ĐỘNG KHÔI PHỤC] Đã tìm thấy và tải bộ não mới nhất tại Episode: {start_episode}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint cũ nào. LAgent sẽ học hoàn toàn mới!")
    
    buffer: RolloutBuffer = RolloutBuffer()
    logger: CSVLogger     = CSVLogger(
        args.log_file,
        [
            "Episode", 
            "R_total",
            "R_energy",
            "R_handover"
        ],
    )

    print("Bắt đầu huấn luyện Phase 1 (Global Agent) Vectorized...")
    
    episodes_completed = start_episode
    total_reward_accum = np.zeros(args.num_envs)
    ep_R_energy = np.zeros(args.num_envs)
    ep_R_handover = np.zeros(args.num_envs)

    states, obses = vec_env.reset()
    
    actor_hidden_states = torch.zeros(args.num_envs * args.number_agents, 128, device=agent.device)

    while episodes_completed < args.episodes:
        # Tính Epsilon-Greedy theo cấu hình (decay trong 50% thời gian đầu)
        epsilon = max(0.05, 0.5 - 0.45 * (episodes_completed / (args.episodes * 0.5)))

        # BATCH INFERENCE
        actions_np, probs_np, next_actor_hidden_states = agent.get_actions(obses, actor_hidden_states, epsilon)
        actions = actions_np

        next_states, next_obses, rewards, rewards_views, dones, term_states, term_obses = vec_env.step(actions)

        # Reset hidden state cho các env đã xong
        for i in range(args.num_envs):
            if dones[i]:
                start_idx = i * args.number_agents
                end_idx = (i + 1) * args.number_agents
                next_actor_hidden_states[start_idx:end_idx] = 0.0

        for i in range(args.num_envs):
            total_reward_accum[i] += float(np.sum(rewards[i]))
            ep_R_energy[i] += float(np.sum(rewards_views[i]["r_energy"]))
            ep_R_handover[i] += float(np.sum(rewards_views[i]["r_handover"]))

            if dones[i]:
                real_next_state = term_states[i]
                real_next_obs = term_obses[i]
                
                try:
                    buffer.store(states[i], obses[i], actions[i], rewards[i], real_next_state, real_next_obs, dones[i])
                except TypeError:
                    pass
                
                episodes_completed += 1
                logger.log([
                    episodes_completed,
                    total_reward_accum[i],
                    ep_R_energy[i],
                    ep_R_handover[i]
                ])
                print(
                    f"Episode: {episodes_completed:4d}/{args.episodes} | Env Rank: {i} | Eps: {epsilon:.2f} | "
                    f"Total: {total_reward_accum[i]:.3f} | "
                    f"Energy: {ep_R_energy[i]:.3f} | "
                    f"Handover: {ep_R_handover[i]:.3f}"
                )

                if episodes_completed % args.save_freq_model == 0:
                    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
                    agent.save_model(args.model_path, episode=episodes_completed)

                total_reward_accum[i] = 0.0
                ep_R_energy[i] = 0.0
                ep_R_handover[i] = 0.0
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
    print("Huấn luyện Phase 1 Vectorized hoàn tất. Mô hình đã được lưu.")

if __name__ == "__main__":
    main()
