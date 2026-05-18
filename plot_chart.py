import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curve(csv_path, save_path, title):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} không tồn tại!")
        return

    df = pd.read_csv(csv_path, header=None, names=['Episode', 'Reward', 'Loss'])
    if len(df) == 0: return
     # In ra tất cả các tên cột để kiểm tra
    print("Các cột có trong file:", df.columns.tolist()) 
    
    # Xóa khoảng trắng thừa ở tên cột (nếu có)
    df.columns = df.columns.str.strip()
    # Cột Y tự động lấy cột thứ 2
    y_col = df.columns[1]
    window_size = max(1, len(df) // 20)
    df['Moving_Avg'] = df[y_col].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df[y_col], alpha=0.3, color='blue', label='Raw Reward')
    plt.plot(df['Episode'], df['Moving_Avg'], color='red', linewidth=2, label=f'Moving Avg ({window_size} eps)')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Lưu biểu đồ thành công: {save_path}")

if __name__ == "__main__":
    plot_learning_curve('experiments/phase1_training.csv', 'experiments/phase1_chart.png', 'HES-COMA Phase 1: Global Strategic Movement')
    # plot_learning_curve('experiments/phase2_training.csv', 'experiments/phase2_chart.png', 'HES-COMA Phase 2: Local Tactical Actions')