# import time
# import math
# import random
# import sys
# from typing import Dict, Optional, Tuple

# import matplotlib.pyplot as plt
# import gymnasium as gym
# from pynput import keyboard

# from envs.env import create_football_env 
# from envs.graph_converter import convert_raw_to_graph, get_simulated_height
# from envs.visualize_graph import draw_pitch_graph_live

# # Global variables for keyboard input
# key_pressed = {}
# selected_player = 0
# current_mode = "AUTO"  # AUTO or MANUAL


# # ==========================
# # KEYBOARD CONTROL SYSTEM
# # ==========================

# def on_press(key):
#     global key_pressed, selected_player, current_mode
#     try:
#         if hasattr(key, 'char'):
#             key_pressed[key.char] = True
#         else:
#             key_name = str(key).split('.')[-1].lower()
#             key_pressed[key_name] = True
#     except:
#         pass

# def on_release(key):
#     global key_pressed
#     try:
#         if hasattr(key, 'char'):
#             if key.char in key_pressed:
#                 del key_pressed[key.char]
#         else:
#             key_name = str(key).split('.')[-1].lower()
#             if key_name in key_pressed:
#                 del key_pressed[key_name]
#     except:
#         pass

# def get_manual_action(player_idx: int) -> int:
#     """
#     Chuyển đổi phím bấm thành hành động GRF:
#     - W/↑: Lên
#     - S/↓: Xuống  
#     - A/←: Trái
#     - D/→: Phải
#     - Q: Sút/Pass cao
#     - E: Pass ngắn
#     - SPACE: Rê bóng
#     - 0: No action
#     """
#     action = 0
    
#     # Định hướng (1-8)
#     if 'w' in key_pressed or 'up' in key_pressed:
#         action = 3  # Up
#     elif 's' in key_pressed or 'down' in key_pressed:
#         action = 7  # Down
#     elif 'a' in key_pressed or 'left' in key_pressed:
#         action = 1  # Left
#     elif 'd' in key_pressed or 'right' in key_pressed:
#         action = 5  # Right
#     elif 'w' in key_pressed and 'a' in key_pressed:
#         action = 2  # Up-Left
#     elif 'w' in key_pressed and 'd' in key_pressed:
#         action = 4  # Up-Right
#     elif 's' in key_pressed and 'a' in key_pressed:
#         action = 8  # Down-Left
#     elif 's' in key_pressed and 'd' in key_pressed:
#         action = 6  # Down-Right
    
#     # Hành động đặc biệt
#     if 'q' in key_pressed:
#         action = 10  # High pass / Sút cao
#     elif 'e' in key_pressed:
#         action = 11  # Short pass / Sút thấp
#     elif ' ' in key_pressed:
#         action = 9   # Rê bóng (Sprint)
    
#     return action

# def get_player_info_string(obs_dict: dict) -> str:
#     """Hiển thị thông tin cầu thủ đã chọn"""
#     left_team = obs_dict['left_team']
#     ball = obs_dict['ball']
    
#     if selected_player >= 0 and selected_player < len(left_team):
#         player_pos = left_team[selected_player]
#         ball_x, ball_y = float(ball[0]), float(ball[1])
#         px, py = float(player_pos[0]), float(player_pos[1])
#         dist_to_ball = math.hypot(px - ball_x, py - ball_y)
#         return f"Cầu thủ: {selected_player} | Pos: ({px:.2f}, {py:.2f}) | Bóng: ({ball_x:.2f}, {ball_y:.2f}) | Dist: {dist_to_ball:.3f}"
#     return "Không có cầu thủ được chọn"


# # ==========================
# # CÁC HÀM HỖ TRỢ CHO ĐÁ PHẠT GÓC
# # ==========================

# PASS_RELEASE_MAPPING: Dict[int, int] = {
#     10: 25,  # High pass -> Release high pass
#     11: 26,  # Short pass -> Release short pass
# }


# def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
#     return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# def get_directional_action_towards_target(current_x: float, current_y: float, target_x: float, target_y: float) -> int:
#     """Uses math.atan2 to calculate the angle and return the closest GRF action (1-8)."""
#     dx = target_x - current_x
#     dy = target_y - current_y

#     if abs(dx) < 1e-6 and abs(dy) < 1e-6:
#         return 0

#     angle = math.atan2(dy, dx)
#     # GRF Y-axis is inverted (Positive is Bottom, Negative is Top)
#     directions = [
#         (5, 0.0),                       # Right
#         (6, math.pi / 4),               # Bottom-Right
#         (7, math.pi / 2),               # Bottom
#         (8, 3 * math.pi / 4),           # Bottom-Left
#         (1, math.pi),                   # Left
#         (2, -3 * math.pi / 4),          # Top-Left
#         (3, -math.pi / 2),              # Top
#         (4, -math.pi / 4),              # Top-Right
#     ]

#     best_action = 0
#     min_diff = float('inf')
#     for action, base_angle in directions:
#         diff = abs(((angle - base_angle + math.pi) % (2 * math.pi)) - math.pi)
#         if diff < min_diff:
#             min_diff = diff
#             best_action = action
#     return best_action


# def choose_corner_strategy(obs: dict, kicker_idx: int) -> dict:
#     """
#     Chọn phương án đá phạt góc: Bắt buộc tạt bóng bổng (high pass - action 10) 
#     vào vùng tính điểm (scoring zone) theo yêu cầu.
#     """
#     left_team = obs['left_team']
#     left_roles = obs.get('left_team_roles', [0]*11)
#     kicker_pos = left_team[kicker_idx]
#     kicker_x, kicker_y = float(kicker_pos[0]), float(kicker_pos[1])

#     teammates = []
#     for idx, pos in enumerate(left_team):
#         if idx == kicker_idx or idx == 0: # Bỏ qua kicker và thủ môn
#             continue
#         dist = euclidean_distance(kicker_pos, pos)
#         role = left_roles[idx] if idx < len(left_roles) else 0
#         height = get_simulated_height(role)
#         teammates.append((dist, idx, pos, height))

#     # ===== CHIẾN THUẬT: TẠT BÓNG BỔNG VÀO VÒNG CẤM =====
#     # Scoring zone: x trong [0.75, 0.85], y = 0.0 (theo rule e.g., x in [0.7, 0.9])
#     target_x = 0.8
#     target_y = 0.0
    
#     candidates = []
#     for dist, idx, pos, height in teammates:
#         # Tính khoảng cách từ cầu thủ đến vùng nhận bóng (scoring zone)
#         dist_to_target = math.hypot(float(pos[0]) - target_x, float(pos[1]) - target_y)
        
#         # Điểm đánh giá: Ưu tiên cầu thủ có chiều cao tốt và đang đứng gần vùng nhận bóng
#         score = (height * 3.0) - dist_to_target
#         candidates.append((score, idx, pos))

#     if candidates:
#         candidates.sort(reverse=True, key=lambda x: x[0])
#         _, target_idx, _ = candidates[0]
#     else:
#         target_idx = teammates[0][1] if teammates else None

#     # Hướng đá của kicker phải hướng về target
#     direction_action = get_directional_action_towards_target(kicker_x, kicker_y, target_x, target_y)
    
#     return {
#         'pass_action': 10, # Bắt buộc là 10 (High Pass) theo Rule
#         'target_idx': target_idx,
#         'target_x': target_x, # Tọa độ x của scoring zone
#         'target_y': target_y, # Tọa độ y của scoring zone
#         'direction_action': direction_action,
#         'description': f"Tạt bóng bổng (Action 10) vào scoring zone ({target_x:.2f}, {target_y:.2f}) cho cầu thủ {target_idx}",
#     }

# def main():
#     global selected_player, current_mode
    
#     print("=" * 80)
#     print("🎮 FOOTBALL MATCH INTERACTIVE CONTROLLER")
#     print("=" * 80)
#     print("\n📋 HƯỚNG DẪN ĐIỀU KHIỂN:")
#     print("  MODE CHUYỂN ĐỔI:")
#     print("    M: Chuyển MANUAL/AUTO")
#     print("    P: Chọn cầu thủ tiếp theo (0-10)")
#     print("    R: Reset trận đấu")
#     print("\n  ĐIỀU KHIỂN (MANUAL MODE):")
#     print("    W/↑: Chạy lên")
#     print("    S/↓: Chạy xuống")
#     print("    A/←: Chạy trái")
#     print("    D/→: Chạy phải")
#     print("    Q: Pass/Sút cao (Action 10)")
#     print("    E: Pass/Sút thấp (Action 11)")
#     print("    SPACE: Rê bóng/Sprint (Action 9)")
#     print("    0: Không hành động")
#     print("\n" + "=" * 80 + "\n")
    
#     # Khởi tạo keyboard listener
#     listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.start()
    
#     print("1. Khởi tạo GRF (Cửa sổ 3D)...")
#     env = create_football_env("academy_corner") 
    
#     print("2. Khởi tạo Matplotlib...")
#     plt.ion() 
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     raw_obs = env.unwrapped.reset()

#     # CÁC BIẾN QUẢN LÝ TRẠNG THÁI
#     episode_step = 0 
#     kicker_idx = -1
#     corner_phase = "INIT" # Máy trạng thái bắt đầu ở INIT (chỉ dùng khi AUTO)
#     strategy = None
#     has_kicked = False
#     selected_player = 1  # Bắt đầu chọn cầu thủ 1
#     current_mode = "AUTO"
#     mode_switch_cooldown = 0
#     player_select_cooldown = 0
    
#     print("3. Bắt đầu vòng lặp... (Nhấn M để chuyển sang MANUAL)")
#     print("="* 80)

#     while True:
#         # XỬ LÝ INPUT KEYBOARD
#         mode_switch_cooldown = max(0, mode_switch_cooldown - 1)
#         player_select_cooldown = max(0, player_select_cooldown - 1)
        
#         if 'm' in key_pressed and mode_switch_cooldown == 0:
#             current_mode = "MANUAL" if current_mode == "AUTO" else "AUTO"
#             mode_switch_cooldown = 30  # 300ms cooldown
#             print(f"\n🔄 Chuyển sang chế độ: {current_mode}\n")
        
#         if 'p' in key_pressed and player_select_cooldown == 0:
#             selected_player = (selected_player + 1) % 11
#             player_select_cooldown = 30
#             print(f"\n👤 Chọn cầu thủ: {selected_player}\n")
        
#         if 'r' in key_pressed and mode_switch_cooldown == 0:
#             print("\n🔄 Reset trận đấu...\n")
#             mode_switch_cooldown = 30
#             raw_obs = env.reset()
#             episode_step = 0
#             corner_phase = "INIT"
#             strategy = None
#             has_kicked = False
#             continue
        
#         obs_dict = raw_obs[0] if isinstance(raw_obs, list) else raw_obs
        
#         # Trích xuất tọa độ hiện tại của bóng và đội hình
#         _ball = obs_dict['ball']
#         ball_x, ball_y = float(_ball[0]), float(_ball[1])
#         _ball_dir = obs_dict['ball_direction']
#         ball_vx, ball_vy = float(_ball_dir[0]), float(_ball_dir[1])
#         ball_speed = math.hypot(ball_vx, ball_vy) * 100 # Adjust speed scale appropriately
        
#         left_coords = obs_dict['left_team']
        
#         # =========================================================
#         # CHẾ ĐỘ MANUAL: ĐIỀU KHIỂN TRỰC TIẾP
#         # =========================================================
#         if current_mode == "MANUAL":
#             actions = [0] * 11
            
#             # Điều khiển cầu thủ đã chọn
#             if 0 <= selected_player < 11:
#                 actions[selected_player] = get_manual_action(selected_player)
            
#             # In thông tin
#             if episode_step % 10 == 0:
#                 print(f"[{episode_step}] {get_player_info_string(obs_dict)} | Mode: {current_mode} | Action: {actions[selected_player]}")
        
#         # =========================================================
#         # CHẾ ĐỘ AUTO: TỰ ĐỘNG CHẠY CHIẾN THUẬT ĐÁ PHẠT GÓC
#         # =========================================================
#         else:
#             # 1. KHỞI TẠO (INIT): TÌM KICKER DỰA TRÊN TỌA ĐỘ VÀ CHỌN CHIẾN THUẬT
#             if corner_phase == "INIT":
#                 min_dist = 999.0
#                 kicker_idx = -1
#                 # Kicker: The player closest to the ball
#                 for i, coord in enumerate(left_coords):
#                     if i == 0:  # Bỏ qua thủ môn
#                         continue
#                     px, py = float(coord[0]), float(coord[1])
#                     dist = math.hypot(px - ball_x, py - ball_y)
#                     if dist < min_dist:
#                         min_dist = dist
#                         kicker_idx = i
                
#                 # Chọn chiến thuật và mục tiêu
#                 strategy = choose_corner_strategy(obs_dict, kicker_idx)
#                 print(f"🎯 [INIT] Khóa mục tiêu: Cầu thủ số {kicker_idx} là Kicker.")
#                 print(f"🧠 [Chiến thuật] {strategy['description']}")
                
#                 corner_phase = "PHASE_1"
#                 has_kicked = False

#             actions = [0] * 11
            
#             target_idx = strategy.get('target_idx') if strategy else None
#             target_x = strategy.get('target_x') if strategy else 0
#             target_y = strategy.get('target_y') if strategy else 0
            
#             # Default fallback for receiver and kicker positioning
#             receiver_dist_to_target = 999.0
#             if target_idx is not None:
#                 rx, ry = float(left_coords[target_idx][0]), float(left_coords[target_idx][1])
#                 receiver_dist_to_target = math.hypot(rx - target_x, ry - target_y)
            
#             kicker_x, kicker_y = float(left_coords[kicker_idx][0]), float(left_coords[kicker_idx][1])
            
#             # TRẠNG THÁI A: POSITIONING & AIMING
#             if corner_phase == "PHASE_1":
#                 # Kicker đứng yên (0) chờ Receiver chạy chỗ, KHÔNG ôm bóng rê
#                 actions[kicker_idx] = 0
                
#                 # Cả đội ngũ hỗ trợ dừng lại, ngoại trừ receiver
#                 for i in range(11):
#                     if i == kicker_idx:
#                         continue
#                     if i == target_idx:
#                         curr_x, curr_y = float(left_coords[i][0]), float(left_coords[i][1])
#                         # Receiver applies directional action towards the target point until destination
#                         if receiver_dist_to_target > 0.05:
#                             actions[i] = get_directional_action_towards_target(curr_x, curr_y, target_x, target_y)
#                         else:
#                             actions[i] = 0 # Đã tới điểm đến
#                     else:
#                         actions[i] = 0
                        
#                 # Wait until Receiver is within 0.05 distance of the target point
#                 if receiver_dist_to_target <= 0.05:
#                     print(f"✔️ Receiver {target_idx} đã tới ({target_x:.2f}, {target_y:.2f}). Kích hoạt PHASE_2!")
#                     corner_phase = "PHASE_2"
#                     kick_frames = 0
                    
#             # TRẠNG THÁI B: SINGLE-TRIGGER PASS
#             elif corner_phase == "PHASE_2":
#                 if 'kick_frames' not in locals():
#                     kick_frames = 0
                    
#                 if kick_frames == 0:
#                     # Kicker: Áp dụng hướng ngắm quay body 1 frame trước khi sút
#                     actions[kicker_idx] = get_directional_action_towards_target(kicker_x, kicker_y, target_x, target_y)
#                     for i in range(11):
#                         if i != kicker_idx:
#                             actions[i] = 0
#                     kick_frames += 1
#                 elif kick_frames == 1:
#                     # Kicker: Pass Action (10) strictly for 1 frame
#                     actions[kicker_idx] = strategy['pass_action']
#                     has_kicked = True
#                     print(f"🚀 [PHASE_2] Trigger pass action {strategy['pass_action']}!")
                    
#                     for i in range(11):
#                         if i != kicker_idx:
#                             actions[i] = 0
#                     kick_frames += 1
#                 else:
#                     # Transition immediately to PHASE_3 in the next frame
#                     corner_phase = "PHASE_3"
                
#             # TRẠNG THÁI C: PHYSICS FOLLOW-THROUGH
#             elif corner_phase == "PHASE_3":
#                 actions[kicker_idx] = 0 # Kicker = Idle
                
#                 # Receiver vẫn đứng đợi hoặc chạy theo mục tiêu
#                 for i in range(11):
#                     if i != kicker_idx:
#                         if i == target_idx:
#                             curr_x, curr_y = float(left_coords[i][0]), float(left_coords[i][1])
#                             actions[i] = get_directional_action_towards_target(curr_x, curr_y, target_x, target_y)
#                         else:
#                             actions[i] = 0
                            
#                 distance_to_ball = math.hypot(ball_x - kicker_x, ball_y - kicker_y)
                
#                 if (ball_speed > 0.1) or (distance_to_ball > 0.05):
#                     print(f"✅ [PHASE_3] Độ rẽ bóng đạt chuẩn (speed={ball_speed:.2f}, dist={distance_to_ball:.3f}). Chuyển OPEN_PLAY!")
#                     corner_phase = "OPEN_PLAY"

#             # TRẠNG THÁI D: BÓNG SỐNG (AI tự do tranh chấp)
#             elif corner_phase == "OPEN_PLAY":
#                 actions = [int(a) for a in env.action_space.sample().tolist()]

#         # =========================================================

#         # KIỂM TRA ĐỐI THỦ CHẠM BÓNG ĐỂ RESET NGAY LẬP TỨC (chỉ áp dụng khi AUTO)
#         force_reset = False
#         done = False
        
#         if current_mode == "AUTO" and corner_phase in ["PHASE_3", "OPEN_PLAY"] and has_kicked:
#             right_coords = obs_dict['right_team']
#             owned_team = obs_dict.get('ball_owned_team', -1)
            
#             opponent_touched = False
#             if owned_team == 1:
#                 opponent_touched = True
#             else:
#                 for coord in right_coords:
#                     if math.hypot(float(coord[0]) - ball_x, float(coord[1]) - ball_y) < 0.02:
#                         opponent_touched = True
#                         break
                        
#             if opponent_touched:
#                 print("❌ [ĐỐI THỦ CHẠM BÓNG] Reset ngay lập tức!")
#                 force_reset = True

#         # THỰC THI HÀNH ĐỘNG
#         if force_reset:
#             done = True
#         else:
#             step_result = env.step(actions)
            
#             if len(step_result) == 5:
#                 raw_obs, reward, terminated, truncated, info = step_result
#                 done = terminated or truncated
#             else:
#                 raw_obs, reward, done, info = step_result
#                 if done:
#                     env.reset()
#             # DELAY để quan sát môi trường
#             time.sleep(0.01)

#         episode_step += 1

#         # XỬ LÝ RESET (chỉ tự động reset khi auto, khi manual phải nhấn R)
#         if done and current_mode == "AUTO":
#             print("🔄 Reset lại môi trường...\n")
#             reset_result = env.reset()
#             raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
#             # Trả máy trạng thái về vạch xuất phát
#             episode_step = 0 
#             corner_phase = "INIT" 
#             strategy = None
#             has_kicked = False

# if __name__ == "__main__":
#     main()


import gfootball.env as football
import time

print("Đang khởi tạo môi trường...")
try:
    env = football.create_environment(
        env_name='11_vs_11_easy_stochastic', 
        render=True, 
        write_video=False,
        logdir='/tmp/football',
        write_full_episode_dumps=False
    )
    env.reset()
    print("Khởi tạo thành công! Đang chạy thử 100 bước...")
    for _ in range(100):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
    print("Hoàn thành!")
except Exception as e:
    print(f"Lỗi: {e}")