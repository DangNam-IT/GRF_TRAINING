Dưới đây là cấu trúc thư mục và mã nguồn Python (sử dụng PyTorch) để triển khai thuật toán học tăng cường đa tác tử HES-COMA (Hierarchical Energy Strategy Counterfactual Multi-Agent Policy Gradients) áp dụng vào môi trường Google Research Football (`gfootball`) dựa trên tài liệu bạn đã cung cấp.

### 1. Gợi Ý Cấu Trúc Thư Mục Cho `gfootball`

Để xây dựng một dự án có tổ chức và dễ dàng mở rộng, bạn nên chia các thành phần của HES-COMA thành các module riêng biệt:

```text
AI_FOR_DECISIONSUBTITUES/
│
├── envs/                       # Chứa các wrapper cho môi trường gfootball
│   ├── __init__.py
|   ├── wrapper_global.py       # Môi trường ε_g cho Phase 1 (8 hướng + 2 dừng, tính Energy Field) 
│   └── wrapper_local.py        # Môi trường ε_l cho Phase 2 (Hành động chiến thuật: sút, chuyền)
│
├── models/                     # Chứa định nghĩa kiến trúc mạng nơ-ron (PyTorch)
│   ├── __init__.py
│   ├── networks.py             # Định nghĩa lớp Actor và Centralized Critic
│
├── experiments/                     # Chứa các file experiment
│   ├── phase1_training.csv     # Chứa kết quả của việc huấn luyện GAgent
|   ├── phase2_training.csv     # Chứa kết quả của việc huấn luyện LAgent
|
├── agents/                     # Chứa logic thuật toán COMA
│   ├── __init__.py
│   ├── coma_agent.py           # Logic cập nhật Actor, Critic, tính TD-error và Advantage 
│
├── utils/                      # Chức năng phụ trợ
│   ├── __init__.py
|   ├── logger.py               # Chứa việc xử lý log để lưu trữ vào experiments
│   ├── buffer.py               # Replay Buffer lưu trữ transition (s, o, a, r, s', d)
│   └── energy_field.py         # Hàm tính toán lực hút/đẩy Gaussian trên không gian sân đấu
|
├── README.md                   # hướng dẫn sử dụng
├── requirements.txt            # thư viện phụ thuộc
├── plot_chart.py               # vẽ biểu đồ trực quan
├── train_phase1.py             # Script chạy vòng lặp huấn luyện GAgent (Không gian toàn cục) 
└── train_phase2.py             # Script chạy vòng lặp huấn luyện LAgent (Không gian cục bộ)

```
