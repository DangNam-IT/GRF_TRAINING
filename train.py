import time
import numpy as np
import torch
import gymnasium as gym
from envs.env import create_football_env
from envs.graph_converter import convert_raw_to_graph
from models.gnn_encoder import TacticGNN

class GraphFootballWrapper(gym.Env):
    """
    Lớp bọc chuyển đổi môi trường GRF thô thành chuẩn Gymnasium hiện đại,
    đồng thời tích hợp sẵn mạng GNN để trích xuất đặc trưng.
    """
    def __init__(self, render=True):
        super().__init__()
        # Khởi tạo môi trường GRF gốc
        self.env = create_football_env("academy_corner")
        
        # ĐỊNH NGHĨA KHÔNG GIAN HÀNH ĐỘNG (GRF có 19 hành động mặc định)
        self.action_space = gym.spaces.Discrete(19)
        
        # ĐỊNH NGHĨA KHÔNG GIAN QUAN SÁT (Giải quyết triệt để lỗi AttributeError)
        # Vì GNN của chúng ta output ra vector 64 chiều, ta khai báo Box(64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        
        # Khởi tạo GNN Encoder
        self.gnn = TacticGNN(num_node_features=7, hidden_channels=32)
        self.gnn.eval()

    def _process_obs(self, raw_obs):
        """Hàm nội bộ để đưa dữ liệu thô qua mạng GNN"""
        # GRF trả về list, lấy phần tử đầu tiên
        obs_dict = raw_obs[0] if isinstance(raw_obs, list) else raw_obs
        graph_data = convert_raw_to_graph(obs_dict, distance_threshold=0.3)
        
        with torch.no_grad(): # Không tính đạo hàm khi chỉ đang lấy mẫu môi trường
            output_vector = self.gnn(graph_data).numpy()
        return output_vector

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # SỬ DỤNG .unwrapped ĐỂ BỎ QUA LỚP VỎ LỖI CỦA GYM
        raw_obs = self.env.unwrapped.reset()
        
        # Xử lý qua GNN
        obs = self._process_obs(raw_obs)
        
        # Ép về chuẩn Gymnasium: trả về (obs, info)
        info = {}
        return obs, info

    def step(self, action):
        # SỬ DỤNG .unwrapped ĐỂ LẤY TRỰC TIẾP 4 BIẾN TỪ LÕI GRF
        raw_obs, reward, done, info = self.env.unwrapped.step(action)
        
        # Xử lý qua GNN
        obs = self._process_obs(raw_obs)
        
        # Ép về chuẩn Gymnasium: tách done thành terminated và truncated
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def main():
    print("1. Khởi tạo môi trường Wrapper hoàn chỉnh (chuẩn Gymnasium)...")
    env = GraphFootballWrapper(render=True)
    
    print("2. Test hàm reset()...")
    obs, info = env.reset()
    print(f"   -> Shape của Observation: {obs.shape}")
    
    print("3. Bắt đầu vòng lặp 50 khung hình...")
    for step in range(500):
        action = env.action_space.sample() 
        
        # step() giờ đây chắc chắn trả về 5 giá trị theo chuẩn mới
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Bước {step+1}/50: Hành động {action} | Output chuẩn bị cho PPO: {obs.shape}")
        
        time.sleep(0.05) 
        
        if terminated or truncated:
            print("\nKịch bản đã kết thúc!")
            break
            
    env.close()
    print("Hoàn tất!")

if __name__ == "__main__":
    main()