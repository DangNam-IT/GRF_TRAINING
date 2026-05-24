import pandas as pd
import matplotlib.pyplot as plt
import os
# # Cột CSV được ghi bởi train_phase1.py (6 cột):
# [Episode, Total_reward_buffer, Reward_env, Reward_energy, Reward_ball_owned]
CSV_COLUMNS = ['Episode', 'Total_reward_buffer', 'Reward_env', 'Reward_energy', 'Reward_ball_owned']

REWARD_CONFIGS = [
    {'col': 'Total_reward_buffer', 'color': '#2196F3', 'label': 'Total Reward (Buffer / Train)'},
    {'col': 'Reward_env',          'color': '#4CAF50', 'label': 'Env Reward (Score)'},
    {'col': 'Reward_energy',       'color': '#FF9800', 'label': 'Energy Reward'},
    {'col': 'Reward_ball_owned',   'color': '#9C27B0', 'label': 'Ball Owned Reward'},
]

def plot_learning_curve(csv_path, save_path, title):
    """Vẽ 4 subplots — mỗi subplot cho 1 loại reward."""
    if not os.path.exists(csv_path):
        print(f"File {csv_path} không tồn tại!")
        return

      # CSV do CSVLogger ghi có header nhưng chỉ 2 tên → bỏ qua dòng đầu, gán tên đúng
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=CSV_COLUMNS)
    if len(df) == 0:
        print("File CSV rỗng, bỏ qua.")
        return

    # Ép kiểu numeric (phòng trường hợp đọc ra string)
    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Episode'], inplace=True)
    print("Các cột có trong file:", df.columns.tolist())

    window_size = max(1, len(df) // 20)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for ax, cfg in zip(axes.flatten(), REWARD_CONFIGS):
        col   = cfg['col']
        color = cfg['color']
        label = cfg['label']

        ma = df[col].rolling(window=window_size).mean()

        ax.plot(df['Episode'], df[col], alpha=0.25, color=color, linewidth=1)
        ax.plot(df['Episode'], ma, color=color, linewidth=2,
                label=f'Moving Avg ({window_size} eps)')

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Lưu biểu đồ thành công: {save_path}")
    plt.show()


def plot_combined(csv_path, save_path, title):
    """Vẽ tất cả reward (Moving Avg) trên 1 đồ thị tổng hợp."""
    if not os.path.exists(csv_path):
        print(f"File {csv_path} không tồn tại!")
        return

    df = pd.read_csv(csv_path, skiprows=1, header=None, names=CSV_COLUMNS)
    if len(df) == 0:
        return

    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Episode'], inplace=True)
    window_size = max(1, len(df) // 20)

    plt.figure(figsize=(12, 6))
    for cfg in REWARD_CONFIGS:
        col   = cfg['col']
        color = cfg['color']
        label = cfg['label']
        ma = df[col].rolling(window=window_size).mean()
        plt.plot(df['Episode'], ma, color=color, linewidth=2, label=label)
    
    plt.title(f"{title} — Overview (Moving Avg)", fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    combined_path = save_path.replace('.png', '_combined.png')
    os.makedirs(os.path.dirname(combined_path) if os.path.dirname(combined_path) else '.', exist_ok=True)
    plt.savefig(combined_path, dpi=300)
    print(f"Lưu biểu đồ tổng hợp: {combined_path}")
    plt.show()


if __name__ == "__main__":
    CSV   = 'experiments/phase1_training_test.csv'
    TITLE = 'HES-COMA Phase 1: Global Strategic Movement'

    plot_learning_curve(CSV, 'experiments/phase1_chart.png', TITLE)
    plot_combined(CSV, 'experiments/phase1_chart.png', TITLE)

    # Phase 2 (bỏ comment khi cần):
    # plot_learning_curve('experiments/phase2_training.csv', 'experiments/phase2_chart.png', 'HES-COMA Phase 2: Local Tactical Actions')
    # plot_combined('experiments/phase2_training.csv', 'experiments/phase2_chart.png', 'HES-COMA Phase 2: Local Tactical Actions')MA Phase 2: Local Tactical Actions') Phase 2: Local Tactical Actions')