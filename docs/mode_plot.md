## Các mode chuẩn hóa trong `plot_chart.py`

---

### 1. `minmax` *(mặc định)*

**Công thức:** `(x - min) / (max - min) × 100`

**Ý nghĩa:** Kéo giãn toàn bộ cột về dải **[0%, 100%]** — giá trị nhỏ nhất trong toàn bộ episode = 0%, lớn nhất = 100%.

**Dùng khi:** Muốn so sánh **biên độ dao động** giữa các cột có đơn vị khác nhau trên cùng 1 biểu đồ.

**Hạn chế:** Bị kéo lệch nếu có outlier cực lớn/nhỏ.

```bash
python plot_chart.py --normalize Reward_energy --normalize-mode minmax
```

---

### 2. `zscore`

**Công thức:** `(x - mean) / std`

**Ý nghĩa:** Mỗi điểm thể hiện **lệch bao nhiêu độ lệch chuẩn** so với trung bình. Giá trị = 0 là trung bình, +1 là tốt hơn 1σ, -1 là kém hơn 1σ.

**Dùng khi:** Muốn xem **xu hướng học** (đang tiến bộ hay thụt lùi) mà không bị ảnh hưởng bởi đơn vị hay scale tuyệt đối.

**Hạn chế:** Không có giới hạn trên/dưới cố định — khó hiểu trực quan với người không quen thống kê.

```bash
python plot_chart.py --normalize Reward_env --normalize-mode zscore
```

---

### 3. `baseline`

**Công thức:** `x / x_first × 100`  *(x_first = giá trị ≠ 0 đầu tiên)*

**Ý nghĩa:** Điểm xuất phát ban đầu = **100%**. Mỗi điểm sau cho thấy agent **tiến bộ hay tụt lùi** bao nhiêu % so với lúc mới bắt đầu.

**Dùng khi:** Muốn trả lời câu hỏi *"Agent đã cải thiện được bao nhiêu % so với ban đầu?"*.

**Hạn chế:** Nếu giá trị đầu tiên âm hoặc quá nhỏ, % sẽ bị khuếch đại mất kiểm soát.

```bash
python plot_chart.py --normalize Reward_total --normalize-mode baseline
```

---

### 4. `rate`

**Công thức:** `count(x ≠ 0 trong [0..t]) / (t+1) × 100`

**Ý nghĩa:** Tại mỗi episode `t`, cho biết **bao nhiêu % tổng số episode đã qua có sự kiện xảy ra** (giá trị ≠ 0). Đây là tỉ lệ **tích lũy** — đường hội tụ dần về tỉ lệ thực sự.

**Dùng khi:** Đo tỉ lệ chiến thắng / tỉ lệ bàn thắng / tỉ lệ kiến tạo tích lũy từ đầu đến nay.

**Hạn chế:** Phản ứng chậm với thay đổi — thành tích tốt ở episode cuối bị "pha loãng" bởi lịch sử dài.

```bash
python plot_chart.py --normalize R_assist --normalize-mode rate
```

> *Ví dụ đọc:* Nếu tại episode 500 giá trị = 12%, nghĩa là trong 500 episode đã qua có **60 lần ghi bàn** (60/500 = 12%).

---

### 5. `rate-rolling`

**Công thức:** `count(x ≠ 0 trong cửa sổ [t-w+1..t]) / w × 100`  *(w lấy từ `--ma`)*

**Ý nghĩa:** Tỉ lệ sự kiện xảy ra **trong `w` episode gần nhất**. Phản ánh xu hướng **cục bộ, hiện tại** — nhanh nhạy hơn `rate`.

**Dùng khi:** Muốn xem agent đang **học tốt lên hay xấu đi gần đây**, ví dụ tỉ lệ thắng 100 trận gần nhất.

**Hạn chế:** Nhạy với nhiễu nếu `w` quá nhỏ.

```bash
python plot_chart.py --normalize R_assist --normalize-mode rate-rolling --ma 100
```

> *Ví dụ đọc:* Nếu tại episode 500, `--ma 100`, giá trị = 20%, nghĩa là trong **100 episode gần nhất** có **20 lần ghi bàn**.

---

## Bảng so sánh nhanh

| Mode | Câu hỏi trả lời | Trục Y | Nhạy outlier? |
|---|---|---|---|
| `minmax` | Giá trị này đứng đâu trong dải [min, max]? | 0 – 100 % | ✅ Có |
| `zscore` | Lệch bao nhiêu so với trung bình? | σ (vô hạn) | ❌ Không |
| `baseline` | Tăng/giảm bao nhiêu % so với đầu? | % (vô hạn) | ✅ Có |
| `rate` | Bao nhiêu % lượt *từ đầu* có sự kiện? | 0 – 100 % | ❌ Không |
| `rate-rolling` | Bao nhiêu % lượt *gần đây* có sự kiện? | 0 – 100 % | ❌ Không |