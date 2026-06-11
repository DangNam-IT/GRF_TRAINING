# Hướng dẫn cài đặt lại môi trường Football Environment

File này ghi chú lại tất cả các lệnh cần thiết để thiết lập lại môi trường `football_env` từ đầu, cùng với lý do đằng sau các lệnh đặc biệt dùng để vượt qua lỗi tương thích.

## 1. Dọn dẹp môi trường cũ (Nếu có)
```bash
conda env remove -n football_env -y
```
- **Mục đích:** Xóa sạch môi trường conda cũ để cài đặt lại từ đầu, tránh xung đột các thư viện cũ.

## 2. Tạo môi trường mới với Python 3.10
```bash
conda create -n football_env python=3.10 -y
```
- **Mục đích:** Khởi tạo lại một môi trường rỗng. Khuyên dùng Python 3.10 vì nó tương thích tốt với các thư viện cũ của `gfootball`.

## 3. Cài đặt các thư viện cơ bản từ requirements
```bash
conda run -n football_env pip install -r requirements.txt
```
*Lưu ý: Bạn nhớ CD vào thư mục gốc của project trước khi chạy lệnh này.*
- **Mục đích:** Cài đặt các thư viện thiết lập sẵn của dự án: `numpy`, `opencv-python`, `psutil` v.v.

## 4. Xử lý lỗi build Gym
```bash
conda run -n football_env python -m pip install setuptools==65.5.0 wheel==0.38.4 pip==23.1.2
```
- **Vấn đề đã giải quyết:** Thư viện `gym<=0.21.0` (yêu cầu bởi `gfootball`) dùng một hệ thống build cũ. Nếu dùng `setuptools` bản mới (ví dụ > 66), quá trình build wheel cho `gym` sẽ báo lỗi `extras_require must be a dictionary`.
- **Mục đích:** Hạ cấp `setuptools` và pip về phiên bản cũ để tương thích với kịch bản build của `gym`.

## 5. Cài đặt gfootball
```bash
conda run -n football_env pip install gfootball
```
- **Vấn đề đã giải quyết:** Ban đầu, mình cố cài `gfootball` bằng mã nguồn từ github (`pip install .`) kết hợp công cụ `vcpkg`. Tuy nhiên quá trình compile C++ bị lỗi hệ thống. 
- **Mục đích:** Giải pháp là tải trực tiếp bản `wheel` pre-compiled từ PyPI qua lệnh `pip install gfootball`. Bản release này đã được tác giả build sẵn C++ cho nền tảng Windows, qua đó khắc phục hoàn toàn lỗi build từ source.

## 6. Xử lý lỗi crash phiên bản SDL của Pygame
```bash
conda run -n football_env pip install pygame==2.1.0
```
- **Vấn đề đã giải quyết:** Bản build trên PyPI của `gfootball` cho C++ được liên kết động với thư viện hệ thống đồ họa **SDL bản 2.0.16**. Nếu dùng `pygame` bản quá mới (ví dụ bản `2.6.x` yêu cầu SDL `2.26`), khi thư viện được load vào RAM, nó phát hiện SDL bị tụt phiên bản và quăng lỗi `RuntimeError: Dynamic linking causes SDL downgrade! (compiled with version 2.26.5, linked to 2.0.16)`.
- **Mục đích:** Hạ cấp `pygame` xuống đích danh bản `2.1.0` (phiên bản dùng chuẩn SDL `2.0.16`) để đồng bộ tuyệt đối với core C++ của `gfootball`.

## 7. Cài đặt các thư viện mở rộng bổ sung
```bash
conda run -n football_env pip install gymnasium matplotlib scipy absl-py
```
- **Mục đích:** Cài đặt các thư viện mà mã nguồn của bạn (`test.py` và các kịch bản RL khác) sử dụng thêm nhưng không nằm sẵn ở tệp `requirements.txt`.

---
**Tóm tắt - Chuỗi lệnh chạy nhanh (Trong Terminal Conda):**
```bash
conda env remove -n football_env -y
conda create -n football_env python=3.10 -y
conda activate football_env
pip install setuptools==65.5.0 wheel==0.38.4 pip==23.1.2
pip install -r requirements.txt
pip install gfootball
pip install pygame==2.1.0
pip install gymnasium matplotlib scipy absl-py
```
