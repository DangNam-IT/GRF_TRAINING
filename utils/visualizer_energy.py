import pygame
import numpy as np

# --- THÔNG SỐ CẤU HÌNH ---
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 500
# Mô phỏng TOÀN BỘ KHÔNG GIAN SÂN BÓNG GFOOTBALL
# X từ -1.05 đến 1.05 (Khung thành trái X = -1.0, Khung thành phải X = 1.0)
# Y từ -0.45 đến 0.45
X_MIN, X_MAX = -1.05, 1.05
Y_MIN, Y_MAX = -0.45, 0.45

# Kích thước lưới để vẽ Heatmap (Nâng lên 5 để tối ưu cho toàn sân)
GRID_SIZE = 5 

def to_screen(x, y):
    """Chuyển đổi tọa độ gfootball sang tọa độ pixel màn hình."""
    screen_x = int((x - X_MIN) / (X_MAX - X_MIN) * WINDOW_WIDTH)
    screen_y = int((y - Y_MIN) / (Y_MAX - Y_MIN) * WINDOW_HEIGHT)
    return screen_x, screen_y

def to_game_coords(screen_x, screen_y):
    """Chuyển đổi từ pixel màn hình sang tọa độ gfootball."""
    x = screen_x / WINDOW_WIDTH * (X_MAX - X_MIN) + X_MIN
    y = screen_y / WINDOW_HEIGHT * (Y_MAX - Y_MIN) + Y_MIN
    return x, y

def gaussian_kernel(pos, target, sigma, scale):
    """Hàm tính năng lượng Gaussian chuẩn."""
    dist_sq = np.sum((pos - target)**2, axis=-1)
    return scale * np.exp(-dist_sq / (sigma**2))

def draw_pitch_lines(screen):
    """Vẽ các vạch kẻ cơ bản của sân bóng."""
    line_color = (200, 200, 200)
    # Đường biên dọc/ngang
    top_l = to_screen(-1.0, -0.42)
    bot_r = to_screen(1.0, 0.42)
    pygame.draw.rect(screen, line_color, (top_l[0], top_l[1], bot_r[0] - top_l[0], bot_r[1] - top_l[1]), 2)
    
    # Vạch giữa sân
    mid_top = to_screen(0.0, -0.42)
    mid_bot = to_screen(0.0, 0.42)
    pygame.draw.line(screen, line_color, mid_top, mid_bot, 2)
    
    # Vòng tròn trung tâm (Bán kính xấp xỉ 0.15)
    center = to_screen(0.0, 0.0)
    radius = int(0.15 * WINDOW_WIDTH / (X_MAX - X_MIN))
    pygame.draw.circle(screen, line_color, center, radius, 2)
    
    # Vòng cấm địa Phải
    box_r_top_l = to_screen(0.65, -0.2)
    box_r_bot_r = to_screen(1.0, 0.2)
    pygame.draw.rect(screen, line_color, (box_r_top_l[0], box_r_top_l[1], box_r_bot_r[0] - box_r_top_l[0], box_r_bot_r[1] - box_r_top_l[1]), 2)
    
    # Vòng cấm địa Trái
    box_l_top_l = to_screen(-1.0, -0.2)
    box_l_bot_r = to_screen(-0.65, 0.2)
    pygame.draw.rect(screen, line_color, (box_l_top_l[0], box_l_top_l[1], box_l_bot_r[0] - box_l_top_l[0], box_l_bot_r[1] - box_l_top_l[1]), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HES-COMA: Toàn Cảnh 22 Cầu Thủ")
    font = pygame.font.SysFont("Arial", 14)

    # --- KHỞI TẠO TỌA ĐỘ 22 CẦU THỦ & BÓNG ---
    ball = np.array([0.98, 0.42]) # Phạt góc
    
    # 11 Cầu thủ đối phương (Right Team) - Tập trung thủ vòng cấm
    right_team_pos = [
        [0.95, 0.0],   # Thủ môn
        [0.85, 0.08],  # Hậu vệ 1 (Kèm cột gần)
        [0.82, -0.1],  # Hậu vệ 2 (Kèm cột xa)
        [0.78, 0.02],  # Hậu vệ 3 (Trung vệ)
        [0.85, -0.2],  # Hậu vệ 4 (Cánh)
        [0.7, 0.15],   # Tiền vệ trụ
        [0.6, 0.0],    # Tiền vệ trung tâm
        [0.5, -0.15],  # Tiền vệ cánh
        [0.4, 0.2],    # Tiền vệ công
        [0.1, 0.0],    # Tiền đạo 1 (Cắm giữa sân)
        [-0.2, 0.1]    # Tiền đạo 2 (Sẵn sàng phản công)
    ]
    
    # 10 Đồng đội (Left Team) - Không tính Agent đang được điều khiển
    left_team_pos = [
        [0.98, 0.40],  # Kicker (Đứng sát bóng)
        [0.8, -0.05],  # Tiền đạo cắm 1
        [0.7, 0.1],    # Tiền đạo cắm 2
        [0.85, 0.2],   # Cầu thủ băng cắt
        [0.4, 0.2],    # Tiền vệ hỗ trợ 1
        [0.4, -0.2],   # Tiền vệ hỗ trợ 2
        [0.1, 0.0],    # Tiền vệ phòng ngự
        [-0.5, 0.1],   # Hậu vệ thòng 1
        [-0.5, -0.1],  # Hậu vệ thòng 2
        [-0.9, 0.0]    # Thủ môn đội nhà
    ]

    # --- THIẾT LẬP TRƯỜNG NĂNG LƯỢNG (Goals & Obstacles) ---
    goals = [
        {"name": "Near Post", "position": np.array([0.9, 0.1]), "sigma": 0.15, "scale": -2.5, "color": (0, 255, 0)},
        {"name": "Far Post", "position": np.array([0.9, -0.1]), "sigma": 0.20, "scale": -1.5, "color": (0, 255, 0)},
        {"name": "Penalty Spot", "position": np.array([0.75, 0.0]), "sigma": 0.25, "scale": -2.0, "color": (0, 255, 0)},
    ]
    
    obstacles = []
    # Thêm toàn bộ 11 cầu thủ đối phương làm đỉnh năng lượng đẩy
    for i, pos in enumerate(right_team_pos):
        obstacles.append({
            "name": f"Defender {i}", 
            "position": np.array(pos), 
            "sigma": 0.06, 
            "scale": 1.5, 
            "color": (255, 50, 50)
        })
    # Khu vực bóng phạt góc cũng là vùng đẩy mạnh
    obstacles.append({"name": "Corner Area", "position": ball, "sigma": 0.25, "scale": 2.5, "color": (255, 150, 0)})

    # --- KHỞI TẠO BẢN ĐỒ NHIỆT (Nền) ---
    print("Đang tính toán Bản đồ nhiệt (Heatmap) toàn sân...")
    heatmap_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    xs = np.linspace(X_MIN, X_MAX, WINDOW_WIDTH // GRID_SIZE)
    ys = np.linspace(Y_MIN, Y_MAX, WINDOW_HEIGHT // GRID_SIZE)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    total_energy = np.zeros(len(grid_points))
    for g in goals:
        total_energy += gaussian_kernel(grid_points, g["position"], g["sigma"], g["scale"])
    for o in obstacles:
        total_energy += gaussian_kernel(grid_points, o["position"], o["sigma"], o["scale"])
        
    for i, point in enumerate(grid_points):
        val = total_energy[i]
        if val < -0.05:
            intensity = min(int(abs(val) * 80), 255)
            color = (0, intensity, 255 - intensity//2)
        elif val > 0.05:
            intensity = min(int(val * 80), 255)
            color = (255, 255 - intensity, 255 - intensity)
        else:
            color = (30, 40, 30) # Sân cỏ tối
            
        sx, sy = to_screen(point[0], point[1])
        pygame.draw.rect(heatmap_surface, color, (sx, sy, GRID_SIZE, GRID_SIZE))

    # --- VÒNG LẶP CHÍNH ---
    agent_pos = np.array([0.65, 0.0]) # Vị trí Agent chủ chốt
    is_dragging = False
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                is_dragging = False
            elif event.type == pygame.MOUSEMOTION and is_dragging:
                mx, my = event.pos
                gx, gy = to_game_coords(mx, my)
                agent_pos = np.array([gx, gy])

        # 1. Vẽ Heatmap và Vạch kẻ sân
        screen.blit(heatmap_surface, (0, 0))
        draw_pitch_lines(screen)
        
        # 2. Vẽ 10 Đồng Đội (Trừ Agent điều khiển)
        for t_pos in left_team_pos:
            sx, sy = to_screen(t_pos[0], t_pos[1])
            pygame.draw.circle(screen, (0, 100, 255), (sx, sy), 6) # Xanh đậm
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 1)

        # 3. Vẽ Goals (Xanh lá) và Đối phương (Đỏ)
        for g in goals:
            sx, sy = to_screen(g["position"][0], g["position"][1])
            radius = int(g["sigma"] * WINDOW_WIDTH / (X_MAX - X_MIN) * 0.3) 
            pygame.draw.circle(screen, g["color"], (sx, sy), radius, 2)
            
        for o in obstacles:
            sx, sy = to_screen(o["position"][0], o["position"][1])
            # Vẽ vùng đẩy
            radius = int(o["sigma"] * WINDOW_WIDTH / (X_MAX - X_MIN) * 0.3)
            pygame.draw.circle(screen, o["color"], (sx, sy), radius, 1)
            # Vẽ Cầu thủ đối phương (Vuông đỏ)
            pygame.draw.rect(screen, (255, 0, 0), (sx - 4, sy - 4, 8, 8))

        # 4. Tia Ray-cast của Agent
        agent_sx, agent_sy = to_screen(agent_pos[0], agent_pos[1])
        num_rays = 8
        max_ray_dist = 0.5 
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            end_pos = agent_pos + direction * max_ray_dist
            end_sx, end_sy = to_screen(end_pos[0], end_pos[1])
            
            ray_energy = 0
            for g in goals:
                ray_energy += gaussian_kernel(end_pos, g["position"], g["sigma"], g["scale"])
            for o in obstacles:
                ray_energy += gaussian_kernel(end_pos, o["position"], o["sigma"], o["scale"])
            
            if ray_energy < -0.5:
                ray_color = (0, 255, 255) # Tốt
            elif ray_energy > 0.5:
                ray_color = (255, 100, 100) # Nguy hiểm
            else:
                ray_color = (200, 200, 200) # Bình thường
                
            pygame.draw.line(screen, ray_color, (agent_sx, agent_sy), (end_sx, end_sy), 1)
            
            text = font.render(f"{ray_energy:.1f}", True, (255, 255, 255))
            screen.blit(text, (end_sx, end_sy))

        # 5. Vẽ Agent
        pygame.draw.circle(screen, (0, 0, 0), (agent_sx, agent_sy), 10)
        pygame.draw.circle(screen, (255, 255, 0), (agent_sx, agent_sy), 8) # Vàng nổi bật
        
        info = font.render(f"Agent X={agent_pos[0]:.2f}, Y={agent_pos[1]:.2f} | Kéo chuột để di chuyển Agent", True, (255, 255, 255))
        screen.blit(info, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()