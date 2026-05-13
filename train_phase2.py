import gym
import torch
from envs.env import create_football_env
from envs.wrapper_local import GFootballLocalWrapper
from utils.buffer import RolloutBuffer
from agents.HES_COMA import HES_COMA_Agent
from utils.logger import CSVLogger


def main():
    env = create_football_env("academy_corner")
    local_env = GFootballLocalWrapper(env)
    
    n_episodes = 500
    max_steps = 3000
    
    # Khởi tạo GAgent (đã train) và LAgent (cần train)
    gagent = HES_COMA_Agent(state_dim=local_env.state_dim, obs_dim=45, n_actions=10) # obs_dim giả định của GAgent
    # Load model GAgent tại đây: gagent.actor.load_state_dict(torch.load('...'))
    
    lagent = HES_COMA_Agent(state_dim=local_env.state_dim, obs_dim=local_env.obs_dim, n_actions=local_env.n_tactic_actions)
    buffer_local = RolloutBuffer()

    # Khởi tạo Logger
    csv_logger = CSVLogger(
        filename='experiments/phase2_training.csv', 
        headers=['Episode', 'Total_Reward']
    )
    STAY_ACTIONS = [8, 9] # IDs quy ước cho hành động dừng

    print("Bắt đầu Phase 2: Huấn luyện Local Agent...")
    for episode in range(1, n_episodes + 1):
        state, obs_g, obs_l = local_env.reset()
        buffer_local.clear()
        total_local_reward = 0
        
        for t in range(max_steps):
            # GAgent không cập nhật tham số trong phase này
            action_g, _ = gagent.get_action(obs_g)
            
            if action_g in STAY_ACTIONS:
                action_l, _ = lagent.get_action(obs_l)
                next_state, next_obs_g, next_obs_l, reward_l, done = local_env.step_local(action_l)
                buffer_local.store(state, obs_l, action_l, reward_l, next_state, done)
                total_local_reward += reward_l
            else:
                next_state, next_obs_g, next_obs_l, _, done = local_env.step_global(action_g)
                
            state, obs_g, obs_l = next_state, next_obs_g, next_obs_l
            if done: 
                break
                
        lagent.update(buffer_local)
        # GHI DỮ LIỆU VÀO FILE CSV SAU MỖI EPISODE
        csv_logger.log([episode, total_local_reward])
        print(f"Episode {episode}/{n_episodes} - Local Reward: {total_local_reward:.2f}")

if __name__ == "__main__":
    main()