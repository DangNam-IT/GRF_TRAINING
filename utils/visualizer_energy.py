import pygame
import numpy as np
import math

# --- THÔNG SỐ CẤU HÌNH ---
# Mở rộng màn hình để có chỗ vẽ biểu đồ Raycast bên phải
WINDOW_WIDTH = 1400 
WINDOW_HEIGHT = 600
PITCH_WIDTH = 1000 # Kích thước riêng của phần sân bóng

# Mô phỏng TOÀN BỘ KHÔNG GIAN SÂN BÓNG GFOOTBALL
X_MIN, X_MAX = -1.05, 1.05
Y_MIN, Y_MAX = -0.45, 0.45

GRID_SIZE = 5 
N_RAYS = 16
MAX_DISTANCE = 0.3
DETECTION_RADIUS = 0.02

def to_screen(x, y):
    """Chuyển đổi tọa độ gfootball sang pixel màn hình (cho phần sân)."""
    screen_x = int((x - X_MIN) / (X_MAX - X_MIN) * PITCH_WIDTH)
    screen_y = int((y - Y_MIN) / (Y_MAX - Y_MIN) * WINDOW_HEIGHT)
    return screen_x, screen_y

def to_game_coords(screen_x, screen_y):
    """Chuyển đổi từ pixel màn hình sang tọa độ gfootball."""
    if screen_x > PITCH_WIDTH: # Giới hạn không cho click vào vùng biểu đồ
        screen_x = PITCH_WIDTH
    x = screen_x / PITCH_WIDTH * (X_MAX - X_MIN) + X_MIN
    y = screen_y / WINDOW_HEIGHT * (Y_MAX - Y_MIN) + Y_MIN
    return x, y

def gaussian_kernel(pos, target, sigma, scale):
    """Hàm tính năng lượng Gaussian chuẩn."""
    dist_sq = np.sum((pos - target)**2, axis=-1)
    return scale * np.exp(-dist_sq / (sigma**2))

def draw_pitch_lines(screen):
    line_color = (200, 200, 200)
    top_l = to_screen(-1.0, -0.42)
    bot_r = to_screen(1.0, 0.42)
    pygame.draw.rect(screen, line_color, (top_l[0], top_l[1], bot_r[0] - top_l[0], bot_r[1] - top_l[1]), 2)
    
    mid_top = to_screen(0.0, -0.42)
    mid_bot = to_screen(0.0, 0.42)
    pygame.draw.line(screen, line_color, mid_top, mid_bot, 2)
    
    center = to_screen(0.0, 0.0)
    radius = int(0.15 * PITCH_WIDTH / (X_MAX - X_MIN))
    pygame.draw.circle(screen, line_color, center, radius, 2)
    
    box_r_top_l = to_screen(0.65, -0.2)
    box_r_bot_r = to_screen(1.0, 0.2)
    pygame.draw.rect(screen, line_color, (box_r_top_l[0], box_r_top_l[1], box_r_bot_r[0] - box_r_top_l[0], box_r_bot_r[1] - box_r_top_l[1]), 2)
    
    box_l_top_l = to_screen(-1.0, -0.2)
    box_l_bot_r = to_screen(-0.65, 0.2)
    pygame.draw.rect(screen, line_color, (box_l_top_l[0], box_l_top_l[1], box_l_bot_r[0] - box_l_top_l[0], box_l_bot_r[1] - box_l_top_l[1]), 2)

def draw_raycast_chart(screen, font, ray_distances):
    """Vẽ biểu đồ Bar Chart ngang hiển thị 16 tia x 3 kênh."""
    chart_x = PITCH_WIDTH + 20
    chart_y = 20
    bar_height = 8
    spacing = 2
    
    # Nền cho biểu đồ
    pygame.draw.rect(screen, (30, 30, 30), (PITCH_WIDTH, 0, WINDOW_WIDTH - PITCH_WIDTH, WINDOW_HEIGHT))
    
    title = font.render("Ray-cast Distance (0.0 -> 1.0)", True, (255, 255, 255))
    screen.blit(title, (chart_x, chart_y))
    chart_y += 30

    # Vẽ vạch 0 và 1.0
    pygame.draw.line(screen, (100, 100, 100), (chart_x + 30, chart_y), (chart_x + 30, chart_y + (bar_height*3 + spacing*4) * 16), 1)
    pygame.draw.line(screen, (100, 100, 100), (chart_x + 30 + 300, chart_y), (chart_x + 30 + 300, chart_y + (bar_height*3 + spacing*4) * 16), 1)

    for i in range(16):
        # Thông tin tia (3 kênh)
        opp_dist = ray_distances[i * 3]
        tm_dist = ray_distances[i * 3 + 1]
        tgt_dist = ray_distances[i * 3 + 2]

        label = font.render(f"R{i}", True, (200, 200, 200))
        screen.blit(label, (chart_x, chart_y + bar_height))

        base_x = chart_x + 30
        max_bar_width = 300

        # Opponent (Đỏ)
        pygame.draw.rect(screen, (255, 50, 50), (base_x, chart_y, int(opp_dist * max_bar_width), bar_height))
        # Teammate (Xanh dương)
        pygame.draw.rect(screen, (50, 150, 255), (base_x, chart_y + bar_height + spacing, int(tm_dist * max_bar_width), bar_height))
        # Target (Xanh lá)
        pygame.draw.rect(screen, (50, 255, 50), (base_x, chart_y + (bar_height + spacing)*2, int(tgt_dist * max_bar_width), bar_height))
        
        # Chỉ in giá trị nhỏ hơn 1.0 (tia bị chặn) để dễ nhìn
        if opp_dist < 1.0:
            val_txt = font.render(f"{opp_dist:.2f}", True, (255, 100, 100))
            screen.blit(val_txt, (base_x + int(opp_dist * max_bar_width) + 5, chart_y - 2))
        if tm_dist < 1.0:
            val_txt = font.render(f"{tm_dist:.2f}", True, (100, 150, 255))
            screen.blit(val_txt, (base_x + int(tm_dist * max_bar_width) + 5, chart_y + bar_height + spacing - 2))
        if tgt_dist < 1.0:
            val_txt = font.render(f"{tgt_dist:.2f}", True, (100, 255, 100))
            screen.blit(val_txt, (base_x + int(tgt_dist * max_bar_width) + 5, chart_y + (bar_height + spacing)*2 - 2))

        chart_y += bar_height * 3 + spacing * 4

    # Legend
    leg_y = WINDOW_HEIGHT - 60
    pygame.draw.rect(screen, (255, 50, 50), (chart_x, leg_y, 10, 10))
    screen.blit(font.render("Opponent (Chướng ngại vật)", True, (200, 200, 200)), (chart_x + 15, leg_y - 2))
    
    pygame.draw.rect(screen, (50, 150, 255), (chart_x, leg_y + 15, 10, 10))
    screen.blit(font.render("Teammate (Đồng đội)", True, (200, 200, 200)), (chart_x + 15, leg_y + 13))
    
    pygame.draw.rect(screen, (50, 255, 50), (chart_x, leg_y + 30, 10, 10))
    screen.blit(font.render("Target (Mục tiêu hút)", True, (200, 200, 200)), (chart_x + 15, leg_y + 28))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HES-COMA: Toàn Cảnh & Phân Tích Ray-cast")
    font = pygame.font.SysFont("Arial", 12)
    font_large = pygame.font.SysFont("Arial", 16)

    # --- KHỞI TẠO TỌA ĐỘ ---
    right_team_pos = np.array([
        [0.95, 0.0], [0.85, 0.08], [0.82, -0.1], [0.78, 0.02], 
        [0.85, -0.2], [0.7, 0.15], [0.6, 0.0], [0.5, -0.15], 
        [0.4, 0.2], [0.1, 0.0], [-0.2, 0.1]
    ])
    
    left_team_pos = np.array([
        [0.98, 0.40], [0.8, -0.05], [0.7, 0.1], [0.85, 0.2], 
        [0.4, 0.2], [0.4, -0.2], [0.1, 0.0], [-0.5, 0.1], 
        [-0.5, -0.1], [-0.9, 0.0]
    ])

    targets = [
        np.array([0.9, 0.1]),   # Near Post
        np.array([0.9, -0.1]),  # Far Post
        np.array([0.75, 0.0])   # Penalty Spot
    ]

    # Energy field goals/obstacles (cho heatmap)
    goals_energy = [
        {"position": targets[0], "sigma": 0.15, "scale": -2.5, "color": (0, 255, 0)},
        {"position": targets[1], "sigma": 0.20, "scale": -1.5, "color": (0, 255, 0)},
        {"position": targets[2], "sigma": 0.25, "scale": -2.0, "color": (0, 255, 0)},
    ]
    obstacles_energy = [{"position": pos, "sigma": 0.06, "scale": 1.5, "color": (255, 50, 50)} for pos in right_team_pos]

    print("Đang tính toán Bản đồ nhiệt (Heatmap) toàn sân...")
    heatmap_surface = pygame.Surface((PITCH_WIDTH, WINDOW_HEIGHT))
    xs = np.linspace(X_MIN, X_MAX, PITCH_WIDTH // GRID_SIZE)
    ys = np.linspace(Y_MIN, Y_MAX, WINDOW_HEIGHT // GRID_SIZE)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    total_energy = np.zeros(len(grid_points))
    for g in goals_energy:
        total_energy += gaussian_kernel(grid_points, g["position"], g["sigma"], g["scale"])
    for o in obstacles_energy:
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
            color = (30, 40, 30)
            
        sx, sy = to_screen(point[0], point[1])
        pygame.draw.rect(heatmap_surface, color, (sx, sy, GRID_SIZE, GRID_SIZE))

    agent_pos = np.array([0.65, 0.0])
    is_dragging = False
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx <= PITCH_WIDTH:
                    is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                is_dragging = False
            elif event.type == pygame.MOUSEMOTION and is_dragging:
                mx, my = event.pos
                if mx <= PITCH_WIDTH:
                    gx, gy = to_game_coords(mx, my)
                    agent_pos = np.array([gx, gy])

        # Vẽ sân
        screen.blit(heatmap_surface, (0, 0))
        draw_pitch_lines(screen)
        
        for t_pos in left_team_pos:
            sx, sy = to_screen(t_pos[0], t_pos[1])
            pygame.draw.circle(screen, (0, 100, 255), (sx, sy), 6)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 1)

        for g in targets:
            sx, sy = to_screen(g[0], g[1])
            pygame.draw.circle(screen, (0, 255, 0), (sx, sy), 8, 2)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 2)
            
        for o in right_team_pos:
            sx, sy = to_screen(o[0], o[1])
            pygame.draw.rect(screen, (255, 0, 0), (sx - 4, sy - 4, 8, 8))

        # --- LOGIC TÍNH TOÁN TIA (Sao chép y hệt hàm _raycast_from_agent) ---
        angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
        ray_distances = []
        
        agent_sx, agent_sy = to_screen(agent_pos[0], agent_pos[1])
        
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # --- Kênh 0: OPPONENTS ---
            min_opp = MAX_DISTANCE
            for opp in right_team_pos:
                ov = opp - agent_pos
                dist = float(np.linalg.norm(ov))
                if dist > 0:
                    proj = float(np.dot(ov, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < DETECTION_RADIUS:
                        min_opp = min(min_opp, dist)

            # --- Kênh 1: TEAMMATES ---
            min_tm = MAX_DISTANCE
            for tm in left_team_pos:
                if not np.allclose(tm, agent_pos):
                    tv = tm - agent_pos
                    dist = float(np.linalg.norm(tv))
                    if dist > 0:
                        proj = float(np.dot(tv, direction))
                        if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < DETECTION_RADIUS:
                            min_tm = min(min_tm, dist)

            # --- Kênh 2: TARGETS ---
            min_target = MAX_DISTANCE
            for tgt in targets:
                tv2 = tgt - agent_pos
                dist = float(np.linalg.norm(tv2))
                if dist > 0:
                    proj = float(np.dot(tv2, direction))
                    if proj > 0 and float(np.sqrt(max(0.0, dist**2 - proj**2))) < DETECTION_RADIUS:
                        min_target = min(min_target, dist)

            ray_distances.extend([min_opp, min_tm, min_target])
            
            # --- VẼ TIA TRÊN SÂN ---
            # Chỉ vẽ tia dài đến vật thể gần nhất bị đụng (hoặc max_distance)
            actual_dist = min([min_opp, min_tm, min_target])
            end_pos = agent_pos + direction * actual_dist
            end_sx, end_sy = to_screen(end_pos[0], end_pos[1])
            
            # Đổi màu tia dựa trên việc đụng ai
            if actual_dist == min_opp and min_opp < MAX_DISTANCE:
                ray_color = (255, 50, 50) # Đụng đối phương (Đỏ)
            elif actual_dist == min_tm and min_tm < MAX_DISTANCE:
                ray_color = (50, 150, 255) # Đụng đồng đội (Xanh dương)
            elif actual_dist == min_target and min_target < MAX_DISTANCE:
                ray_color = (50, 255, 50) # Đụng mục tiêu (Xanh lá)
            else:
                ray_color = (150, 150, 150) # Trống
                
            pygame.draw.line(screen, ray_color, (agent_sx, agent_sy), (end_sx, end_sy), 1)
            
            # Ghi số R0, R1... ở đầu tia
            lbl = font.render(f"R{i}", True, (255, 255, 255))
            screen.blit(lbl, (end_sx, end_sy))

        arr = np.array(ray_distances, dtype=np.float32)
        norm_distances = np.minimum(arr / MAX_DISTANCE, 1.0)
        
        # Vẽ biểu đồ bên phải
        draw_raycast_chart(screen, font, norm_distances)

        # Vẽ Agent
        pygame.draw.circle(screen, (0, 0, 0), (agent_sx, agent_sy), 10)
        pygame.draw.circle(screen, (255, 255, 0), (agent_sx, agent_sy), 8)
        
        info = font_large.render(f"Agent X={agent_pos[0]:.2f}, Y={agent_pos[1]:.2f} | Mũi tên R0 -> R15 chỉ hướng", True, (255, 255, 255))
        screen.blit(info, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()