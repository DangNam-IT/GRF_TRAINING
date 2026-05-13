import gym
from envs.env import create_football_env
from envs.wrapper_global import GFootballGlobalWrapper
from utils.buffer import RolloutBuffer
from agents.HES_COMA import HES_COMA_Agent
from utils.logger import CSVLogger

def main():
    # Khởi tạo môi trường gfootball cơ bản
    env = create_football_env("academy_corner")
    global_env = GFootballGlobalWrapper(env)
    
    n_episodes = 500
    max_steps = 3000
    
    gagent = HES_COMA_Agent(state_dim=global_env.state_dim, obs_dim=global_env.obs_dim, n_actions=global_env.action_space.n)
    buffer = RolloutBuffer()

    # Khởi tạo Logger
    csv_logger = CSVLogger(
        filename='experiments/phase1_training.csv', 
        headers=['Episode', 'Total_Reward']
    )

    print("Bắt đầu Phase 1: Huấn luyện Global Agent...")
    for episode in range(1, n_episodes + 1):
        state, obs = global_env.reset()
        buffer.clear()
        total_reward = 0
        
        for t in range(max_steps):
            # 1. Khởi tạo mảng chứa 11 hành động cho 11 cầu thủ
            actions = []
            
            # 2. Sinh hành động cho từng tác tử (Dùng chung Policy Actor)
            for i in range(11):
                # obs[i] là mảng quan sát của cầu thủ thứ i
                act, _ = gagent.get_action(obs[i])
                actions.append(act)
                
            # 3. Thực thi toàn bộ 11 hành động cùng lúc trong môi trường
            next_state, next_obs, rewards, done = global_env.step(actions)
            
            # 4. Lưu transition vào Buffer cho TỪNG TÁC TỬ
            for i in range(11):
                # rewards lúc này là 1 mảng gồm 11 phần thưởng tương ứng
                buffer.store(state, obs[i], actions[i], rewards[i], next_state, done)
                total_reward += rewards[i]
                
            state, obs = next_state, next_obs
            if done: 
                break
                
        gagent.update(buffer)
        # GHI DỮ LIỆU VÀO FILE CSV SAU MỖI EPISODE
        csv_logger.log([episode, total_reward])
        print(f"Episode {episode}/{n_episodes} - Reward: {total_reward:.2f}")

    # Cần thêm logic lưu model (torch.save) tại đây
    print("Huấn luyện Phase 1 hoàn tất.")

if __name__ == "__main__":
    main()