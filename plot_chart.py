import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curve(csv_path, save_path, title):
    """
    Đọc dữ liệu từ CSV và vẽ biểu đồ Learning Curve.
    """
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy file dữ liệu: {csv_path}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(csv_path)
    
    # Sử dụng moving average để làm mượt biểu đồ (giúp dễ nhìn xu hướng hơn)
    window_size = max(1, len(df) // 20) 
    df['Reward_Moving_Avg'] = df['Total_Reward'].rolling(window=window_size).mean()

    # Cấu hình biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Vẽ reward gốc (màu nhạt)
    plt.plot(df['Episode'], df['Total_Reward'], alpha=0.3, color='blue', label='Raw Reward')
    
    # Vẽ đường trung bình (màu đậm)
    plt.plot(df['Episode'], df['Reward_Moving_Avg'], color='red', linewidth=2, label=f'Moving Avg (Window={window_size})')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Lưu biểu đồ thành file ảnh
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Đã lưu biểu đồ tại: {save_path}")
    
    # Hiển thị biểu đồ lên màn hình
    plt.show()

if __name__ == "__main__":
    # Đảm bảo thư mục lưu ảnh tồn tại
    os.makedirs('results', exist_ok=True)
    
    # Vẽ biểu đồ cho Phase 1
    plot_learning_curve(
        csv_path='results/phase1_training.csv',
        save_path='results/phase1_learning_curve.png',
        title='HES-COMA Phase 1: Global Agent Learning Curve'
    )
    
    # Nếu có file Phase 2, bạn có thể gọi thêm:
    # plot_learning_curve('results/phase2_training.csv', 'results/phase2_learning_curve.png', 'HES-COMA Phase 2: Local Agent')