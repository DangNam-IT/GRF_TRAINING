# Giải Thích Chi Tiết Từng Dòng Code — `networks.py`

---

## Phần Import

```python
import torch
```
Nhập thư viện PyTorch — framework học sâu chính. Cung cấp tensor, autograd, và các công cụ huấn luyện mạng nơ-ron.

```python
import torch.nn as nn
```
Nhập module `nn` (Neural Network) — chứa các lớp mạng có sẵn như `Linear`, `ReLU`, `Sequential`. Tất cả mạng tùy chỉnh đều kế thừa từ `nn.Module` của module này.

```python
import torch.nn.functional as F
```
Nhập module `F` — chứa các hàm kích hoạt và hàm mất mát dạng **stateless** (không có tham số học được), ví dụ `F.softmax`, `F.relu`. Khác với `nn.ReLU` (là một lớp object), `F.relu` chỉ là một hàm thuần túy.

---

## Mạng 1 — `ActorNetwork`

### Khai báo lớp

```python
class ActorNetwork(nn.Module):
```
Định nghĩa lớp `ActorNetwork` kế thừa từ `nn.Module`. Kế thừa này bắt buộc để PyTorch có thể:
- Tự động theo dõi các tham số học được (`θ`)
- Hỗ trợ `backward()` tính gradient
- Cho phép `.parameters()`, `.state_dict()`, `.to(device)`, v.v.

---

### Hàm khởi tạo `__init__`

```python
def __init__(self, obs_dim, n_actions):
```
Constructor nhận 2 tham số:
- `obs_dim`: số chiều của vector quan sát đầu vào (ví dụ: kích thước của $o^g$ hoặc $o^l$)
- `n_actions`: số hành động có thể chọn (đầu ra của mạng)

```python
    super(ActorNetwork, self).__init__()
```
Gọi constructor của lớp cha `nn.Module`. **Bắt buộc phải có** — nếu bỏ qua, PyTorch sẽ không đăng ký các tham số và mạng sẽ không hoạt động đúng.

```python
    self.net = nn.Sequential(
```
Tạo một pipeline tuần tự: dữ liệu đi qua từng lớp theo thứ tự từ trên xuống dưới, đầu ra của lớp trước là đầu vào của lớp sau.

```python
        nn.Linear(obs_dim, 128),
```
**Lớp tuyến tính thứ 1** — Fully Connected layer:
- Đầu vào: vector quan sát kích thước `obs_dim`
- Đầu ra: vector 128 chiều (hidden layer)
- Phép tính: $\mathbf{y} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$
- Số tham số học: `obs_dim × 128 + 128` (trọng số + bias)

```python
        nn.ReLU(),
```
**Hàm kích hoạt phi tuyến** sau lớp Linear thứ 1:
- Công thức: $\text{ReLU}(x) = \max(0, x)$
- Mục đích: nếu không có hàm này, dù xếp chồng bao nhiêu lớp Linear cũng chỉ tương đương 1 phép biến đổi tuyến tính → mất khả năng học các pattern phức tạp

```python
        nn.Linear(128, 128),
```
**Lớp tuyến tính thứ 2** — Hidden layer thứ 2:
- Cả đầu vào lẫn đầu ra đều 128 chiều
- Lớp này cho phép mạng học các biểu diễn trừu tượng hơn từ features đã được trích xuất ở lớp trước
- Số tham số: `128 × 128 + 128 = 16.512`

```python
        nn.ReLU(),
```
**Hàm kích hoạt phi tuyến** sau lớp Linear thứ 2 — tương tự ở trên.

```python
        nn.Linear(128, n_actions)
```
**Lớp đầu ra (Output layer)**:
- Đầu vào: 128 chiều
- Đầu ra: `n_actions` chiều — mỗi chiều tương ứng với **logit** (điểm số thô) của một hành động
- Chưa qua softmax nên các giá trị này có thể âm hoặc dương tùy ý
- Số tham số: `128 × n_actions + n_actions`

```python
    )
```
Kết thúc định nghĩa `nn.Sequential`.

---

### Hàm `forward`

```python
def forward(self, obs):
```
Định nghĩa chiều thuận (forward pass) — được gọi tự động khi thực hiện `actor(obs)`. PyTorch dùng hàm này để:
- Tính đầu ra
- Xây dựng computation graph cho `backward()`

Tham số `obs`: tensor quan sát, shape thường là `(batch_size, obs_dim)`.

```python
    return F.softmax(self.net(obs), dim=-1)
```
Hai thao tác lồng nhau:

**Bước 1 —** `self.net(obs)`: đưa `obs` qua toàn bộ pipeline Sequential ở trên, kết quả là tensor logit shape `(batch_size, n_actions)`.

**Bước 2 —** `F.softmax(..., dim=-1)`: chuyển logit thành **phân phối xác suất**:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

- `dim=-1` nghĩa là tính softmax dọc theo chiều cuối cùng (chiều `n_actions`)
- Đảm bảo: tất cả xác suất $\geq 0$ và tổng bằng 1
- Kết quả: $\pi_\theta(a \mid o)$ — xác suất chọn mỗi hành động
- Ví dụ output: `[0.05, 0.70, 0.10, 0.15]` nghĩa là hành động 1 có 70% khả năng được chọn

> ⚠️ **Lưu ý quan trọng:** Khi tính `actor_loss` dùng `log π(a|o)`, nên dùng `F.log_softmax` + `NLLLoss` thay vì `log(softmax(...))` để tránh vấn đề số học (underflow khi xác suất rất nhỏ).

---

## Mạng 2 — `CentralCriticNetwork`

### Khai báo lớp

```python
class CentralCriticNetwork(nn.Module):
```
Tương tự, kế thừa `nn.Module`. Đây là **Centralized Critic** — đặc trưng của kiến trúc CTDE (Centralized Training, Decentralized Execution): trong lúc huấn luyện, Critic được phép nhìn thấy **trạng thái toàn cục + hành động của tất cả agents**.

---

### Hàm khởi tạo `__init__`

```python
def __init__(self, state_dim, n_actions, n_agents=11):
```
Constructor nhận 3 tham số:
- `state_dim`: số chiều của trạng thái toàn cục $s$
- `n_actions`: số hành động của **mỗi** agent
- `n_agents=11`: số agents (mặc định 11 — đội bóng đá)

```python
    super(CentralCriticNetwork, self).__init__()
```
Gọi constructor lớp cha — bắt buộc như đã giải thích.

```python
    # Input bao gồm State toàn cục + Hành động one-hot của tất cả 11 agents
    self.input_dim = state_dim + (n_actions * n_agents)
```
Tính kích thước đầu vào thực sự của Critic:

$$\text{input\_dim} = \underbrace{\text{state\_dim}}_{\text{trạng thái toàn cục}} + \underbrace{n\_actions \times n\_agents}_{\text{joint action one-hot}}$$

**Tại sao dùng one-hot?** Vì hành động là rời rạc (categorical), cần mã hóa thành vector liên tục để mạng xử lý được. Ví dụ với `n_actions=9, n_agents=11`: joint action chiếm `9 × 11 = 99` chiều.

```python
    self.net = nn.Sequential(
        nn.Linear(self.input_dim, 256),
```
**Lớp tuyến tính thứ 1** — nhận toàn bộ `[state ‖ joint_action]` đã được ghép lại:
- Kích thước lớn hơn (256) vì đầu vào phức tạp hơn nhiều so với ActorNetwork
- Phải học mối quan hệ chéo giữa trạng thái và hành động của **nhiều agents cùng lúc**

```python
        nn.ReLU(),
```
Kích hoạt phi tuyến — lý do tương tự như trên.

```python
        nn.Linear(256, 128),
```
**Lớp thu hẹp** — giảm từ 256 xuống 128:
- Ép mạng học các đặc trưng **cô đọng, trừu tượng** hơn
- Pattern phổ biến trong thiết kế mạng: mở rộng rồi thu hẹp dần

```python
        nn.ReLU(),
```
Kích hoạt phi tuyến lần 2.

```python
        nn.Linear(128, n_agents) # Xuất Q-value cho mỗi agent
```
**Lớp đầu ra** — khác biệt quan trọng so với ActorNetwork:
- Đầu ra có `n_agents = 11` chiều
- Mỗi chiều $i$ là $Q_{\phi}^{\text{central}}(s, a^{\text{joint}})$ **riêng cho agent $i$**
- Một lần forward duy nhất tính được Q-value cho **cả 11 agents** → hiệu quả tính toán

```python
    )
```
Kết thúc `nn.Sequential`.

---

### Hàm `forward`

```python
def forward(self, state, joint_action_one_hot):
```
Forward pass nhận 2 đầu vào:
- `state`: tensor trạng thái toàn cục, shape `(batch_size, state_dim)`
- `joint_action_one_hot`: tensor hành động one-hot của tất cả agents, shape `(batch_size, 11, n_actions)`

```python
    # Flatten joint_action_one_hot từ (Batch, 11, n_actions) -> (Batch, 11 * n_actions)
    action_flat = joint_action_one_hot.view(joint_action_one_hot.size(0), -1)
```
**Làm phẳng (flatten)** tensor 3D thành 2D:
- `joint_action_one_hot.size(0)`: lấy `batch_size` (chiều đầu tiên)
- `-1`: PyTorch tự tính chiều còn lại = `11 × n_actions`
- Lý do: `nn.Linear` chỉ nhận tensor 2D `(batch, features)`, không nhận 3D

Trực quan hóa phép biến đổi:
```
(batch, 11, n_actions)           (batch, 11 * n_actions)
┌──────────────────┐             ┌─────────────────────────────┐
│ agent0: [1,0,...] │             │ [1,0,..., 0,1,..., ..., 1,0] │
│ agent1: [0,1,...] │  ─view()─► │  ^^^agent0^^^  ^^^agent1^^^  │
│    ...            │             └─────────────────────────────┘
└──────────────────┘
```

```python
    x = torch.cat([state, action_flat], dim=-1)
```
**Ghép nối (concatenate)** trạng thái và joint action dọc theo chiều cuối:
- `state` shape: `(batch, state_dim)`
- `action_flat` shape: `(batch, 11 * n_actions)`
- Kết quả `x` shape: `(batch, state_dim + 11 * n_actions)` = `(batch, input_dim)`
- Đây là bước then chốt: Critic nhìn thấy **đồng thời** trạng thái môi trường **và** quyết định của mọi agent

```python
    return self.net(x)
```
Đưa tensor đã ghép qua pipeline Sequential. Kết quả shape `(batch, 11)` — Q-value của từng agent trong batch.

> Để lấy Q-value của agent $i$ cụ thể: `q_values[:, i]`

---

## Sơ Đồ Luồng Dữ Liệu Tổng Hợp

```
ActorNetwork
─────────────────────────────────────────────────────
obs (B, obs_dim)
    │
    ▼
Linear(obs_dim → 128) → ReLU
    │
    ▼
Linear(128 → 128) → ReLU
    │
    ▼
Linear(128 → n_actions)     ← logit thô
    │
    ▼
F.softmax(dim=-1)            ← xác suất hành động
    │
    ▼
π(a|o) : (B, n_actions)     ← phân phối chính sách


CentralCriticNetwork
─────────────────────────────────────────────────────
state (B, state_dim)     joint_onehot (B, 11, n_act)
    │                           │
    │                      .view() → (B, 11*n_act)
    │                           │
    └──────── torch.cat ────────┘
                  │
                  ▼
         x : (B, input_dim)
                  │
                  ▼
    Linear(input_dim → 256) → ReLU
                  │
                  ▼
    Linear(256 → 128) → ReLU
                  │
                  ▼
    Linear(128 → 11)          ← Q-value thô (không activation)
                  │
                  ▼
    Q^central(s,a) : (B, 11)  ← Q cho từng agent
```

---

## Điểm Thiết Kế Quan Trọng Cần Nhớ

| Đặc điểm | `ActorNetwork` | `CentralCriticNetwork` |
|---|---|---|
| Đầu vào | Quan sát **cục bộ** $o_i$ | Trạng thái **toàn cục** $s$ + **joint action** |
| Đầu ra | Phân phối xác suất $\pi(\cdot\|o)$ | Q-value cho **11 agents** cùng lúc |
| Activation cuối | `softmax` (xác suất hợp lệ) | **Không có** (Q-value có thể âm/dương tùy ý) |
| Dùng lúc nào | Training **và** Execution | **Chỉ** Training (CTDE) |
| Số lượng instance | 2 cái: $\theta^g$ và $\theta^l$ | 2 cái: $\phi^g$ và $\phi^l$ |


Dựa trên phân tích kiến trúc tổng thể của thuật toán **HES-COMA** áp dụng cho môi trường Google Research Football (GRF), cấu trúc mạng nơ-ron được mô tả trong tệp `network.md` của bạn là **Chính xác về mặt lý thuyết và Phù hợp hoàn hảo với biến thể phần thưởng định hình (Reward Shaping) của Trường năng lượng**.

Dưới đây là phần kiểm tra tính đúng đắn, làm rõ định nghĩa, cách sử dụng và lý do tại sao thuật toán lại thiết kế mạng nơ-ron theo cách này để phục vụ cho Đồ án tốt nghiệp của bạn.

---

## PHẦN 1: KIỂM TRA TÍNH ĐÚNG ĐẮN (VALIDATION)

### 1. Mạng `ActorNetwork`: ĐÚNG ĐẮN

* **Tính phù hợp:** Mạng nhận đầu vào là `obs_dim` (Quan sát cục bộ) và xuất ra phân phối xác suất cho `n_actions`. Thiết kế này cho phép áp dụng kỹ thuật **Parameter Sharing (Chia sẻ trọng số)**: Bạn chỉ cần khởi tạo *một thực thể duy nhất* của lớp này, nhưng có thể truyền batch dữ liệu dạng `(11, obs_dim)` để lấy hành động cho cả 11 cầu thủ cùng lúc.
* **Lưu ý thực tế:** Hàm kích hoạt cuối cùng là `F.softmax(dim=-1)` giúp đảm bảo tổng xác suất của các hành động luôn bằng 1, hoàn toàn đúng đắn cho việc lấy mẫu hành động rời rạc (Categorical Sampling).

### 2. Mạng `CentralCriticNetwork`: ĐÚNG ĐẮN VỚI THUẬT TOÁN HES-COMA

Trong thuật toán COMA truyền thống của Meta (Foerster et al.), mạng Critic thường nhận vào $(s, a^{-i})$ (Trạng thái và hành động của các tác tử khác) và xuất ra $|A|$ đầu ra (Q-value của tất cả hành động khả dĩ của tác tử $i$).

Tuy nhiên, kiến trúc trong file `network.md` của bạn thiết kế đầu ra là `n_agents` (11 đầu ra). **Thiết kế này là ĐÚNG và RẤT THÔNG MINH vì lý do sau:**

* Trong HES-COMA, mỗi cầu thủ nhận một phần thưởng riêng biệt sau khi đã trừ đi giá trị Trường Năng Lượng tại vị trí của họ ($r_i = r_{team} - E_i$).


* Vì mỗi tác tử có một hàm phần thưởng (Reward Function) khác nhau, nên hàm giá trị $Q$ của mỗi người là khác nhau.
* Việc mạng Critic nhận vào toàn bộ Trạng thái toàn cục + Hành động liên kết (`joint_action`) và xuất ra 11 giá trị $Q$ riêng biệt cho 11 cầu thủ giúp mạng học được hàm giá trị cá nhân hóa này chỉ trong **một lần lan truyền xuôi (Single Forward Pass)**, tiết kiệm tài nguyên tính toán.

---

## PHẦN 2: LÀM RÕ VÀ GIẢI THÍCH CHI TIẾT CÁC THÀNH PHẦN CORE

Để làm nổi bật hàm lượng khoa học trong báo cáo đồ án, bạn cần làm rõ các khái niệm và cơ chế vận hành sau:

### 1. Triết lý CTDE (Centralized Training, Decentralized Execution)

* **Định nghĩa:** Huấn luyện trung tâm, Thực thi phân tán.
* **Cách thức hoạt động trong code:**
* **Lúc chạy trận đấu (Execution):** Mạng `CentralCriticNetwork` hoàn toàn bị cất đi. Chỉ có `ActorNetwork` hoạt động. Mỗi cầu thủ chỉ được biết thông tin trong tầm mắt của mình (`obs`) để tự quyết định hành động.
* **Lúc cập nhật mạng (Training):** Mạng `CentralCriticNetwork` được bật lên. Nó đứng ở góc nhìn "Thượng đế" để nhìn thấy toàn bộ sơ đồ chiến thuật (`state`) và biết rõ tất cả 11 người đã làm gì (`joint_action`).


* **Tại sao phải dùng?** Nếu không có Critic trung tâm, các Actor khi tự học sẽ nhìn nhận các đồng đội xung quanh như một phần của môi trường đang biến động liên tục (vấn đề *Non-stationarity*). Critic trung tâm giúp ổn định hóa quá trình học tập bằng cách bao quát toàn bộ hành vi phối hợp của cả đội.

### 2. Cơ chế Mã Hóa One-Hot cho Joint Action

* **Tại sao dùng?** Hành động của cầu thủ trong `gfootball` được đại diện bằng các số nguyên (0: Đứng yên, 1: Chạy trái, 2: Chạy lên...). Các con số này không mang tính chất bắc cầu hay lớn bé (Hành động số 2 không hề "lớn hơn" hay "tốt hơn" hành động số 1). Nếu bạn nạp thẳng các số nguyên này vào lớp `nn.Linear`, mạng nơ-ron sẽ thực hiện phép nhân toán học $W \times 2$ và $W \times 1$, dẫn đến hiểu sai hoàn toàn bản chất của hành động.
* **Cách xử lý (`F.one_hot`):** Biến số nguyên thành một vector nhị phân độc lập. Ví dụ với `n_actions=10`, hành động `0` biến thành `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`, hành động `1` biến thành `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`. Lúc này, mỗi hành động là một hướng hoàn toàn vuông góc và độc lập trong không gian vector toán học.

### 3. Phép biến đổi `.view(joint_action_one_hot.size(0), -1)`

* **Định nghĩa:** Đây là thao tác **Làm phẳng (Flatten)** dữ liệu tensor hình khối (3D) thành dạng bảng phẳng (2D).
* **Cách thức hoạt động:** Dữ liệu hành động ban đầu có shape là `(Batch_size, 11, n_actions)` — tức là một khối 3 chiều đại diện cho: *Với mỗi mẫu trong batch, có 11 cầu thủ, mỗi cầu thủ có 1 vector một-hot*. Lớp tuyến tính `nn.Linear` của Critic không thể đọc được cấu trúc 3D này. Hàm `.view(..., -1)` sẽ trải phẳng nó ra thành `(Batch_size, 11 * n_actions)`.
* **Tại sao dùng?** Để chuẩn bị dữ liệu cho phép toán ghép nối `torch.cat` với vector trạng thái toàn cục `state`, tạo ra một đầu vào duy nhất, liền mạch cho mạng Critic.

### 4. Tại sao đầu ra của Critic không sử dụng Hàm kích hoạt (Activation Function)?

* **Lý do:** Lớp cuối cùng của Actor sử dụng `Softmax` vì đầu ra bắt buộc phải nằm trong khoảng `[0, 1]` để đại diện cho xác suất. Ngược lại, lớp cuối cùng của Critic xuất ra giá trị **Q-value** — đại diện cho tổng phần thưởng tích lũy kỳ vọng trong tương lai.
* 
**Tại sao không dùng ReLU/Sigmoid ở lớp cuối Critic?** Phần thưởng trong bóng đá hoặc giá trị năng lượng từ Trường năng lượng có thể mang giá trị âm (ví dụ: bị đẩy ra xa, bị thủng lưới) hoặc dương (ghi bàn, chạy vào khoảng trống chiến thuật). Nếu bạn vô tình đặt hàm `nn.ReLU()` ở cuối Critic, tất cả các giá trị Q-value âm sẽ bị biến thành `0`, khiến Critic hoàn toàn mất khả năng nhận biết rủi ro hoặc các vị trí có năng lượng xấu. Do đó, lớp cuối của Critic phải là một lớp tuyến tính tự do (Linear mà không có activation).



Cấu trúc mạng này của bạn đã chuẩn hóa và tối ưu tốt cho việc xử lý batch lớn. Bạn có thể tự tin đưa toàn bộ phần giải trình chuyên sâu này vào chương "Thiết kế kiến trúc mạng nơ-ron" trong quyển báo cáo đồ án của mình!