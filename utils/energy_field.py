# import numpy as np

# def calculate_gaussian_energy(pos, target_pos, is_attraction=True, sigma=1.0, scale=10.0):
#     """
#     Tính toán giá trị năng lượng dựa trên phân phối Gaussian.
#     - Lực hút (is_attraction=True): mang giá trị âm, hướng tới mục tiêu có lợi.
#     - Lực đẩy (is_attraction=False): mang giá trị dương, đại diện cho chướng ngại vật/đối thủ.
#     """
#     pos = np.array(pos)
#     target_pos = np.array(target_pos)
    
#     # Tính bình phương khoảng cách
#     distance_sq = np.sum((pos - target_pos)**2)
    
#     # Tính năng lượng theo Gaussian
#     energy = scale * np.exp(-distance_sq / (2 * sigma**2))
    
#     return -energy if is_attraction else energy

# def get_total_energy_field(agent_pos, goals, obstacles):
#     """
#     Tổng hợp trường năng lượng từ các lực hút (goals) và lực đẩy (obstacles).
#     """
#     total_energy = 0.0
    
#     for goal in goals:
#         total_energy += calculate_gaussian_energy(agent_pos, goal, is_attraction=True)
        
#     for obs in obstacles:
#         total_energy += calculate_gaussian_energy(agent_pos, obs, is_attraction=False)
        
#     return total_energy

