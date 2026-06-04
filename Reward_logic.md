Dựa trên mô tả `raw observations` và `actions` của môi trường GRF (Google Research Football) trong tệp `observation.md`, chúng ta sẽ phân tích kỹ lưỡng mối quan hệ giữa các trọng số $\sigma$, `scale` và tham số phần thưởng, đồng thời giải đáp cơ chế truyền bóng của `kicker_id`.

---

## 1. Phân Tích Ảnh Hưởng Của $\sigma$ Và `Scale` Lên Phần Thưởng

Trong hàm `step` hiện tại, phần thưởng chênh lệch năng lượng ($R_{energy}$) được tính bằng công thức:


$$R_{energy} = \left( E(p_t) - E(p_{t+1}) \right) \times 0.1$$


Với $0.1$ là hằng số `ENERGY_SCALE`. Điều này có nghĩa là mọi thay đổi trong thiết lập `scale` và $\sigma$ sẽ trực tiếp tỷ lệ thuận với lượng phần thưởng mà Agent nhận được mỗi bước.

### Nhận định về các cấu hình hiện tại:

* **Có ảnh hưởng tiêu cực không?** Hiện tại, cấu hình **chưa tối ưu và có thể gây ra hiện tượng "Local Optima" (Tối ưu cục bộ)**.
* **Lý do:** Bạn đang để `scale` của các mục tiêu (Goals) từ `-2.0` đến `-3.0`, trong khi `scale` của chướng ngại vật (Opponents) là `1.5`, quả bóng là `2.5`. Sự chênh lệch này dẫn đến một vấn đề lớn:
1. **Sợ hãi quả bóng (Ball Phobia):** Ở trạng thái bóng sống (không phải phạt góc), bạn đang để `scale` của bóng là `-2.0`, nhưng ở trạng thái phạt góc, bóng bị biến thành chướng ngại vật với `scale` lên tới `2.5`. Sự thay đổi đột ngột này có thể khiến Agent "sợ hãi" quả bóng ngay cả khi phạt góc vừa kết thúc.
2. **Chênh lệch Gradient phần thưởng:** Khi Agent chạy từ biên vào vòng cấm, do độ sâu của mục tiêu (`-3.0`) quá lớn so với lực cản của hậu vệ (`1.5`), Agent có thể sẵn sàng **chạy xuyên qua mặt hậu vệ** thay vì tìm kẽ hở luồn lách. Lý do là phần thưởng cộng thêm khi đến đích đủ lớn để bù đắp cho phần phạt âm khi va chạm với vùng $\sigma$ của hậu vệ.
3. **Tác động của `ENERGY_SCALE`:** Với `scale = -3.0` và `ENERGY_SCALE = 0.1`, phần thưởng lớn nhất một Agent có thể nhận khi nhảy thẳng vào hố năng lượng là $+0.3$. Tuy nhiên, `ENV_SCALE` (phần thưởng ghi bàn) chỉ là `1.0 / 11 = 0.091`. Việc **phần thưởng di chuyển lớn gấp 3 lần phần thưởng ghi bàn** sẽ làm Agent "lười" sút, chỉ thích chạy quanh vòng cấm để "farm" điểm năng lượng!



### Hướng điều chỉnh cân bằng (Reward Alignment)

Mục tiêu là phải đảm bảo **tổng phần thưởng năng lượng tích lũy không bao giờ vượt qua phần thưởng ghi bàn thực tế** ($0.091$).

1. **Hạ `scale` của toàn bộ hệ thống xuống để khớp với `ENV_SCALE`:**
* `near_post`: `scale = -0.5`
* `far_post`: `scale = -0.3`
* `penalty_spot`: `scale = -0.4`
* `obstacles` (Right Team): `scale = 0.3`
* `obstacles` (Ball ở góc): `scale = 0.5`


2. **Điều chỉnh $\sigma$ (Bán kính):** Giữ nguyên như bạn đã tối ưu, vì nó dựa trên hình học thực tế của sân.
* Hậu vệ: $\sigma = 0.06$ (Rất nhỏ để tạo kẽ hở).
* Mục tiêu: $\sigma = 0.15 \rightarrow 0.25$.



Bằng cách này, giá trị $E(p_t) - E(p_{t+1})$ cực đại chỉ rơi vào khoảng $0.05$ nhân với `ENERGY_SCALE = 0.1` sẽ là $0.005$ mỗi bước. Agent sẽ nhận ra rằng chạy chỗ chỉ là phương tiện (nhận phần thưởng nhỏ), ghi bàn mới là mục đích (nhận phần thưởng lớn $0.091$).

---

## 2. Lực Và Vị Trí Quả Bóng Phụ Thuộc Vào Đâu Khi `Kicker_id` Chuyền?

Đây là câu hỏi then chốt giải thích cơ chế vật lý của GRF, được trình bày rõ trong tệp `observation.md` bạn cung cấp:

### Cơ chế tự động (Auto-determined Direction & Power)

Trong GRF, khi bạn ra lệnh cho tác nhân chuyền bóng (`action_long_pass = 9`, `action_high_pass = 10`, `action_short_pass = 11`), **lực sút và đồng đội nhận bóng không được cung cấp như một tham số hàm** (như trong FIFA hay PES).

Đọc kỹ tài liệu `observation.md`:

> `action_long_pass` = 9, perform a long pass to the player on your team. Player to pass the ball to is auto-determined based on the movement direction.

**Sự thật về hành động Chuyền trong GRF:**

1. **Phụ thuộc vào "Movement Direction" (Hướng di chuyển hiện tại):**
Trước khi bấm nút chuyền, Agent phải được hướng dẫn xoay người về một phía (thông qua 8 hành động di chuyển `action_top`, `action_bottom_right`...). Máy gia tốc vật lý của game sẽ nội suy hướng mặt của cầu thủ (`left_team_direction`).
2. **Auto-targeting (Tự động khóa mục tiêu):**
GRF có một bộ hỗ trợ chuyền bóng ẩn (pass assistance). Khi Agent phát lệnh `9`, game sẽ quét một hình nón theo hướng mặt cầu thủ. Nó tìm xem có đồng đội nào đang đứng ở khu vực đó không. Nếu có, game sẽ **tự động tính toán lực sút (Power)** và **quỹ đạo (Trajectory)** để quả bóng rơi đúng vào vị trí của đồng đội đó.
3. **Hệ quả đối với Phạt góc:**
Trong mã của bạn, Kicker bị ép thực hiện lệnh `9` thông qua bộ lọc `_map_global_actions`. Kicker sẽ tạt quả bóng về phía vòng cấm dựa trên việc hệ thống GRF tự động tìm kiếm các đồng đội đang di chuyển.

### Làm thế nào để điều khiển quả tạt "ảo diệu" hơn?

Nếu Kicker chỉ đơn thuần gán cứng `mapped_actions[i] = 9` ở mọi bước, quả tạt sẽ có tính chất ngẫu nhiên cao vì hướng mặt của Kicker lúc đó chưa chắc đã hướng về điểm cắt mặt nguy hiểm.

Để cải thiện, ở Phase 2 (LAgent), khi kịch bản yêu cầu tạt bóng, bạn có thể thiết kế để LAgent không chỉ chọn hành động `9` mà còn phải chọn các chuỗi hành động kết hợp, ví dụ:

1. *Frame 1:* Xoay mặt hướng vào vùng Cột gần (`action_bottom_right` = 6).
2. *Frame 2:* Giữ nút tạt (`action_high_pass` = 10).

Nhờ vào việc các đồng đội của Kicker đã được huấn luyện khôn ngoan bằng Trường Năng lượng để hội tụ về 3 điểm chiến lược (Near post, Far post, Penalty spot), bộ Auto-targeting của GRF sẽ tự động nhận diện ra các điểm đến lý tưởng này và thực hiện các quả tạt có độ xoáy, điểm rơi hoàn hảo. Đó chính là vẻ đẹp của việc kết hợp Học chiến thuật (GAgent) và Cơ chế vật lý game (Engine)!