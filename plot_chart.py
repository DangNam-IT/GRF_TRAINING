"""
plot_chart.py  —  Vẽ learning-curve từ file CSV huấn luyện.

Cú pháp:
  python plot_chart.py [options]

Ví dụ:
  # Vẽ tất cả cột, tự động MA
  python plot_chart.py --csv experiments/phase1_train_test4.csv

  # Chuẩn hoá Min-Max [0-100%] cho 2 cột
  python plot_chart.py \\
      --csv experiments/phase1_train_test4.csv \\
      --cols Total_reward_buffer Reward_energy \\
      --normalize Reward_energy \\
      --normalize-mode minmax

  # Chuẩn hoá Z-score cho Reward_energy
  python plot_chart.py \\
      --csv experiments/phase1_train_test4.csv \\
      --normalize Reward_energy \\
      --normalize-mode zscore

  # Chuẩn hoá % so với baseline (điểm đầu tiên = 100%)
  python plot_chart.py \\
      --csv experiments/phase1_train_test4.csv \\
      --normalize Reward_energy \\
      --normalize-mode baseline

  # Kết hợp nhiều tuỳ chọn
  python plot_chart.py \\
      --csv experiments/phase1_train_test4.csv \\
      --cols Total_reward_buffer Reward_env Reward_energy \\
      --frames 1000 --ma 50 \\
      --normalize Reward_energy Reward_env \\
      --normalize-mode zscore \\
      --title "Phase 1 Training" \\
      --save experiments/figures/phase1_chart.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Màu mặc định tuần hoàn khi người dùng không cấu hình thủ công
# ---------------------------------------------------------------------------
DEFAULT_COLORS = [
    '#FF9800', '#9C27B0', '#2196F3', '#4CAF50',
    '#FF5722', '#00BCD4', '#E91E63', '#8BC34A',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: str, columns: list[str] | None, frames: int | None) -> pd.DataFrame:
    """Đọc CSV, lọc cột, cắt số frame (dòng) nếu cần."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File không tồn tại: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("File CSV rỗng.")

    print(f"[INFO] Đọc được {len(df)} dòng, các cột: {df.columns.tolist()}")

    # Giới hạn số khung hình
    if frames is not None:
        df = df.iloc[:frames]
        print(f"[INFO] Cắt còn {len(df)} dòng (--frames {frames})")

    # Lọc cột nếu người dùng chỉ định
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Cột không tìm thấy trong CSV: {missing}")
    else:
        # Loại cột Episode/Step khỏi danh sách vẽ
        columns = [c for c in df.columns if c not in ('Episode', 'Step', 'step', 'episode')]

    # Ép kiểu numeric
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cột trục x
    x_col = next((c for c in ('Episode', 'episode', 'Step', 'step') if c in df.columns), None)
    if x_col is None:
        df['Episode'] = range(len(df))
        x_col = 'Episode'

    return df, x_col, columns


# ---------------------------------------------------------------------------
# Normalization modes
# ---------------------------------------------------------------------------

def _norm_minmax(series: pd.Series, col: str) -> pd.Series:
    """Min-Max → [0, 100%]. Toàn bộ dải giá trị được ánh xạ tuyến tính."""
    col_min, col_max = series.min(), series.max()
    if col_max == col_min:
        print(f"[WARN] Cột '{col}' có min == max, không thể chuẩn hoá minmax.")
        return series
    result = (series - col_min) / (col_max - col_min) * 100
    print(f"[INFO] [{col}] minmax → [{result.min():.1f}%, {result.max():.1f}%]")
    return result


def _norm_zscore(series: pd.Series, col: str) -> pd.Series:
    """Z-score → (x - mean) / std. Giá trị = 0 là trung bình, ±1 là 1 độ lệch chuẩn."""
    mean, std = series.mean(), series.std()
    if std == 0:
        print(f"[WARN] Cột '{col}' có std == 0, không thể chuẩn hoá zscore.")
        return series
    result = (series - mean) / std
    print(f"[INFO] [{col}] zscore → mean={mean:.4f}, std={std:.4f}")
    return result


def _norm_baseline(series: pd.Series, col: str) -> pd.Series:
    """% so với baseline — điểm đầu tiên hợp lệ (khác 0) = 100%."""
    # Tìm giá trị baseline đầu tiên khác 0
    baseline = series.dropna()
    baseline = baseline[baseline != 0]
    if baseline.empty:
        print(f"[WARN] Cột '{col}' không có giá trị baseline hợp lệ (khác 0).")
        return series
    base_val = baseline.iloc[0]
    result = series / base_val * 100
    print(f"[INFO] [{col}] baseline → base={base_val:.4f}, "
          f"range=[{result.min():.1f}%, {result.max():.1f}%]")
    return result


def _norm_rate(series: pd.Series, col: str, **kwargs) -> pd.Series:
    """
    Tỉ lệ sự kiện tích lũy (Cumulative Event Rate).
    rate(t) = count(values != 0  trong [0..t]) / (t + 1)  × 100%
    Câu hỏi: "Tính đến lượt này, bao nhiêu % lượt có sự kiện xảy ra?"
    """
    event = (series > 0).astype(float)
    result = event.expanding().mean() * 100
    total = int(event.sum())
    print(f"[INFO] [{col}] rate (cumulative) — {total}/{len(series)} sự kiện, "
          f"tỉ lệ cuối={result.iloc[-1]:.1f}%")
    return result


def _norm_rate_rolling(series: pd.Series, col: str, window: int = 50, **kwargs) -> pd.Series:
    """
    Tỉ lệ sự kiện cửa sổ trượt (Rolling Event Rate).
    rate(t) = count(values != 0  trong cửa sổ [t-w+1..t]) / w  × 100%
    Sử dụng --ma để đặt kích thước cửa sổ w (default = ma_window).
    Câu hỏi: "Trong w lượt gần nhất, bao nhiêu % có sự kiện?"
    """
    event = (series != 0).astype(float)
    result = event.rolling(window=window, min_periods=1).mean() * 100
    print(f"[INFO] [{col}] rate-rolling (w={window}) — "
          f"range=[{result.min():.1f}%, {result.max():.1f}%]")
    return result


_NORM_FN = {
    'minmax':        _norm_minmax,
    'zscore':        _norm_zscore,
    'baseline':      _norm_baseline,
    'rate':          _norm_rate,
    'rate-rolling':  _norm_rate_rolling,
}


def normalize_columns(df: pd.DataFrame, normalize_cols: list[str],
                      mode: str = 'minmax', ma_window: int = 50) -> pd.DataFrame:
    """
    Biến đổi các cột chỉ định theo mode:
      - 'minmax'       : [0, 100%]  — (x-min)/(max-min)*100
      - 'zscore'       : z-score    — (x-mean)/std
      - 'baseline'     : % baseline — x/x_first*100
      - 'rate'         : tỉ lệ sự kiện tích lũy  (values!=0 / total)
      - 'rate-rolling' : tỉ lệ sự kiện cửa sổ trượt (dùng ma_window làm w)
    """
    if mode not in _NORM_FN:
        raise ValueError(f"normalize-mode không hợp lệ: '{mode}'. Chọn: {list(_NORM_FN)}.")

    fn = _NORM_FN[mode]
    for col in normalize_cols:
        if col not in df.columns:
            print(f"[WARN] Cột '{col}' không có trong dataframe, bỏ qua.")
            continue
        df[col] = fn(df[col].copy(), col, window=ma_window)
    return df


def _y_label(col: str, normalize_cols: list[str], mode: str) -> str:
    """Tạo nhãn trục Y phù hợp với mode chuẩn hoá."""
    if col not in normalize_cols:
        return col
    suffix = {
        'minmax':       ' [0–100 %]',
        'zscore':       ' [z-score]',
        'baseline':     ' [% vs baseline]',
        'rate':         ' [event rate % cumul.]',
        'rate-rolling': ' [event rate % rolling]',
    }
    return col + suffix.get(mode, '')


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_subplots(df: pd.DataFrame, x_col: str, columns: list[str],
                  ma_window: int, normalize_cols: list[str], norm_mode: str,
                  title: str, save_path: str) -> None:
    """Vẽ subplot riêng cho từng cột."""
    n = len(columns)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    ax_list = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = ax_list[i]
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        y = df[col]
        ma = y.rolling(window=ma_window, min_periods=1).mean()

        ylabel = _y_label(col, normalize_cols, norm_mode)

        ax.plot(df[x_col], y, alpha=0.25, color=color, linewidth=1)
        ax.plot(df[x_col], ma, color=color, linewidth=2,
                label=f'Moving Avg (w={ma_window})')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    # Ẩn các ô thừa
    for j in range(i + 1, len(ax_list)):
        ax_list[j].set_visible(False)

    plt.tight_layout()
    _save(save_path)
    plt.show()


def plot_combined(df: pd.DataFrame, x_col: str, columns: list[str],
                  ma_window: int, normalize_cols: list[str], norm_mode: str,
                  title: str, save_path: str) -> None:
    """Vẽ tất cả Moving Avg trên 1 đồ thị tổng hợp."""
    plt.figure(figsize=(12, 6))

    for i, col in enumerate(columns):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        ma = df[col].rolling(window=ma_window, min_periods=1).mean()
        label = _y_label(col, normalize_cols, norm_mode)
        plt.plot(df[x_col], ma, color=color, linewidth=2, label=label)

    plt.title(f"{title} — Overview (Moving Avg, w={ma_window})", fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    combined_path = save_path.replace('.png', '_combined.png')
    _save(combined_path)
    plt.show()


def _save(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Đã lưu: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Vẽ learning-curve từ file CSV huấn luyện RL.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--csv', '-c',
        default='experiments/phase1_train_test4.csv',
        help='Đường dẫn tới file CSV (default: experiments/phase1_train_test4.csv)',
    )
    parser.add_argument(
        '--cols', '-C',
        nargs='+',
        default=None,
        metavar='COL',
        help='Danh sách tên cột muốn vẽ (default: tất cả cột trừ Episode/Step). '
             'Ví dụ: --cols Total_reward_buffer Reward_env',
    )
    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=None,
        metavar='N',
        help='Số dòng (khung hình/episode) tối đa muốn dùng (default: toàn bộ).',
    )
    parser.add_argument(
        '--ma', '-m',
        type=int,
        default=None,
        metavar='WINDOW',
        help='Kích thước cửa sổ moving-average (default: tự động = len/10).',
    )
    parser.add_argument(
        '--normalize', '-n',
        nargs='+',
        default=[],
        metavar='COL',
        help='Tên các cột muốn chuẩn hoá trước khi vẽ. '
             'Ví dụ: --normalize Reward_energy Reward_possession',
    )
    parser.add_argument(
        '--normalize-mode', '-N',
        default='minmax',
        choices=['minmax', 'zscore', 'baseline', 'rate', 'rate-rolling'],
        metavar='MODE',
        help=(
            'Phương pháp biến đổi áp dụng cho các cột trong --normalize:\n'
            '  minmax       — [0, 100%%]           : (x-min)/(max-min)*100  (default)\n'
            '  zscore       — z-score               : (x-mean)/std\n'
            '  baseline     — %% vs điểm đầu        : x/x_first*100\n'
            '  rate         — tỉ lệ sự kiện tích lũy: count(!=0 đến t)/t  × 100%%\n'
            '  rate-rolling — tỉ lệ sự kiện cửa sổ  : count(!=0 trong w)/w × 100%% (w=--ma)'
        ),
    )
    parser.add_argument(
        '--title', '-t',
        default='Learning Curve',
        help='Tiêu đề biểu đồ (default: "Learning Curve").',
    )
    parser.add_argument(
        '--save', '-s',
        default='experiments/figures/chart.png',
        help='Đường dẫn lưu biểu đồ subplot .png (default: experiments/figures/chart.png).',
    )
    parser.add_argument(
        '--no-combined',
        action='store_true',
        help='Bỏ qua vẽ biểu đồ tổng hợp (combined).',
    )
    parser.add_argument(
        '--no-subplots',
        action='store_true',
        help='Bỏ qua vẽ subplots riêng lẻ.',
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    # 1. Tải dữ liệu
    df, x_col, columns = load_csv(args.csv, args.cols, args.frames)

    # 2. Biến đổi / chuẩn hoá
    # argparse đổi dấu '-' → '_' trong tên attribute nên ta tự map lại
    norm_mode = args.normalize_mode.replace('_', '-')   # 'rate_rolling' → 'rate-rolling'
    # Tính ma_window sớm để rate-rolling dùng được đúng cửa sổ
    ma_window = args.ma if args.ma is not None else max(1, len(df) // 10)
    if args.normalize:
        print(f"[INFO] Normalize mode: {norm_mode}  |  Cột: {args.normalize}")
        df = normalize_columns(df, args.normalize, mode=norm_mode, ma_window=ma_window)

    # 3. Window moving-average (đã tính ở bước 2, chỉ log lại)
    print(f"[INFO] Moving-average window = {ma_window}")

    # 4. Vẽ
    if not args.no_subplots:
        plot_subplots(df, x_col, columns, ma_window,
                      args.normalize, norm_mode, args.title, args.save)

    if not args.no_combined:
        plot_combined(df, x_col, columns, ma_window,
                      args.normalize, norm_mode, args.title, args.save)