import os
import numpy as np
import gfootball.env as football_env
from envs.wrapper_global import GFootballGlobalWrapper
from agents.HES_COMA import HES_COMA_Agent
from utils.buffer import RolloutBuffer
from utils.logger import CSVLogger

def create_env():
    return football_env.create_environment(
        env_name="academy_corner",
        number_of_left_players_agent_controls=11,
        representation='raw',
        rewards='scoring',
        render=True
    )



def main():
    base_env = create_env()
    env = GFootballGlobalWrapper(base_env, num_agents=11)
    
    agent = HES_COMA_Agent(state_dim=env.state_dim, obs_dim=env.obs_dim, n_actions=10, n_agents=11)
    buffer = RolloutBuffer()
    logger = CSVLogger('experiments/phase1_training.csv', ['Episode', 'Total_Reward'])
    
    n_episodes = 10
    max_steps = 100
    
    print("Bắt đầu huấn luyện Phase 1 (Global Agent)...")
    for episode in range(1, n_episodes + 1):
        state, obses = env.reset()
        buffer.clear()
        total_reward = 0
        
        for t in range(max_steps):
            actions, _ = agent.get_actions(obses)

            next_state, next_obses, rewards, done = env.step(actions)
            
            buffer.store(state, obses, actions, rewards, next_state, done)
            total_reward += np.sum(rewards)
            
            state, obses = next_state, next_obses
            if done: break
                
        agent.update(buffer)
        logger.log([episode, total_reward])
        print(f"Episode: {episode}/{n_episodes} | Lợi nhuận: {total_reward:.2f}")

    # Thay thế đoạn torch.save cũ bằng:
    os.makedirs('experiments/models', exist_ok=True)
    agent.save_model('experiments/models/gagent_model.pth')
    print("Huấn luyện hoàn tất. Đã lưu mô hình GAgent cùng cấu hình môi trường.")

if __name__ == '__main__':
    main()