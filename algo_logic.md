Dưới đây là nội dung từ tài liệu đính kèm đã được định dạng lại bằng Markdown và LaTeX để đảm bảo tính rõ ràng, chuẩn xác và dễ dàng đọc hiểu:

4.2 Các bước thực hiện 

Khung học tăng cường đa tác tử với cấu trúc phân cấp chia quá trình học tập thành phân cấp không gian toàn cục và phân cấp không gian cục bộ. Ở giai đoạn đầu tiên, hệ thống phân cấp không gian toàn cầu cho phép tác nhân (GAgent) học cách chuyển động tới vị trí chiến lược thông qua trường năng lượng. Ở giai đoạn thứ hai, hệ thống phân cấp không gian cục bộ sử dụng vị trí chiến lược được xác định ở giai đoạn trước đó để tác nhân (LAgent) học các hành động chiến thuật như sút bóng, chuyền bóng hay cướp bóng. Quá trình xác định hành động chỉ được diễn ra khi hệ thống phân cấp không toàn cầu xảy ra hành động dừng lại thì hành động ở giai đoạn thứ hai này mới được áp dụng vào môi trường. 

---

4.2.1 Học phân cấp không gian toàn cục để di chuyển vị trí chiến lược 

**4.2.1.1 Bước khởi tạo** 

* **Bước 1**: Khởi tạo ngẫu nhiên tham số Actor $\theta^{g}$ và Critic $\phi^{g}$. 


* **Bước 2**: Cấu hình môi trường huấn luyện cấp toàn cục $\epsilon_{g}$ chỉ cho phép 8 hướng di chuyển và 1 hành động dừng. 



**4.2.1.2 Vòng lặp chính (episode = $1$ đến $N_{ep}$)** Mỗi episode thực hiện thông qua 2 vòng lặp con: 

* **Bước 3**: Reset môi trường $\epsilon_{g}$ và quan sát trạng thái toàn cục ban đầu $s_{0}$ bao gồm Ray-based và Energy Field. 


* **Bước 4**: Khởi tạo Buffer rỗng. 


$$Buffer\leftarrow \emptyset$$





**Vòng lặp thời gian ($t=0$ đến EpisodeMaxStep)** 

* **Bước 5**: Lấy quan sát riêng của GAgent từ trạng thái hiện tại $s_{t}$ (gồm thông tin Ray và Energy Field):

$o_{t}^{g} \leftarrow PartialObservation(s_{t})$. 


* **Bước 6**: Lấy mẫu hành động từ chính sách Actor hiện tại: $a_{t}^{g} \leftarrow \pi_{\theta^{g}}^{g}(o_{t}^{g})$. 


* **Bước 7**: Thực thi hành động $a_{t}^{g}$ trong môi trường $\epsilon_{g}$.


* **Bước 8**: Quan sát kết quả sau hành động $s_{t+1}, r_{t}^{g}, d \leftarrow \mathcal{E}_{g}$, trong đó $r_{t}^{g}$ là phần thưởng dựa trên Energy Field và $d$ là cờ kết thúc episode. 


* **Bước 9**: Lưu transition vào Buffer: $Buffer \leftarrow Buffer \cup \{(s_{t},o_{t}^{g},a_{t}^{g},r_{t}^{g},s_{t+1},d)\}$. 


* **Bước 10**: Cập nhật trạng thái: $s_{t} \leftarrow s_{t+1}$. 


* **Bước 11**: Nếu $d = \text{true}$ thì kết thúc episode (break). 



**Cập Nhật COMA Trên Toàn Bộ Dữ Liệu Buffer** Với mỗi transition $(s, o^{g}, a^{g}, r^{g}, S^{\prime}, d)$ trong Buffer: 

* **Bước 12**: Tính giá trị Q từ Critic trung tâm:


$$Q_{\Phi^{g}}^{central}(s,a^{g})$$





* **Bước 13**: Tính baseline (kỳ vọng có trọng số theo chính sách):


$$b(s)=\sum_{a^{\prime g}}\pi_{\theta^{g}}^{g}({a^{\prime}}^{g}|o^{g})Q_{\phi^{g}}^{central}(s,{a^{\prime}}^{g})$$





* **Bước 14**: Tính TD-error (sai số Temporal Difference):


$$\delta \leftarrow r^{g}+{\gamma Q_{\Phi^{g}}}^{central}(s^{\prime},{a^{\prime}}^{g})-{Q_{\phi^{g}}}^{central}(s,a^{g})$$





* **Bước 15**: Cập nhật Critic $\phi^{g}$ bằng cách tối thiểu hóa $(\delta)^{2}$. 


* **Bước 16**: Tính lợi thế phản thực tế (counterfactual advantage):


$$Adv^{g} \leftarrow Q_{\Phi^{g}}^{central}(s,a^{g})-b(s)$$





* **Bước 17**: Cập nhật Actor $\theta^{g}$ theo gradient:


$$\theta^{g} \leftarrow \theta^{g}+\nabla_{\theta^{g}}\log \pi^{g}(a^{g}|o^{g})\cdot Adv^{g}$$





* **Bước 18**: Kết thúc vòng lặp cập nhật COMA và xóa sạch Buffer. 

$$Buffer\leftarrow \emptyset$$

* **Bước 19**: Kết thúc vòng lặp chính, sau $N_{ep}$ trả về bộ tham số đã huấn luyện bao gồm $\theta^{g}$ và $\phi^{g}$ đã được huấn luyện bởi GAgent. 



---

4.2.2 Học phân cấp không gian cục bộ cho các hành động chiến thuật 

**4.2.2.1 Bước khởi tạo** 

* **Bước 1**: Khởi tạo ngẫu nhiên tham số Actor $\theta^{l}$ và Critic $\phi^{l}$. 


* **Bước 2**: Cố định $\theta^{l}$ từ bước 1 (GAgent không được cập nhật trong giai đoạn này). 


* **Bước 3**: Cấu hình môi trường huấn luyện cấp cục bộ $\epsilon_{l}$, trong đó hành động di chuyển của GAgent do $\pi^{g}_{\theta^{g}}$ đảm nhận và LAgent chỉ hành động khi GAgent ở trạng thái $\text{STAY}$. 



**4.2.2.2 Vòng lặp chính (episode = $1$ đến $N_{ep}$)** Mỗi episode thực hiện thông qua 2 vòng lặp con: 

* **Bước 4**: Reset môi trường $\epsilon_{l}$ và quan sát trạng thái toàn cục ban đầu $s_{0}$. 


* **Bước 5**: Khởi tạo Buffer cục bộ rỗng: $Buffer_{l} \leftarrow \emptyset$. 



**Vòng lặp thời gian ($t=0$ đến EpisodeMaxStep)** 

* **Bước 6**: Lấy quan sát riêng của GAgent từ trạng thái hiện tại: $o_{t}^{g} \leftarrow PartialObservation_{g}(s_{t})$. 


* **Bước 7**: Lấy quan sát riêng của LAgent từ trạng thái hiện tại: $o_{t}^{l} \leftarrow PartialObservation_{l}(s_{t})$. 


* **Bước 8**: GAgent lấy mẫu hành động toàn cục (di chuyển hoặc đứng yên): ${a^{g}}_{t} \leftarrow \pi^{g}_{\theta^{g}}(o_{t}^{g})$. 


* **Bước 9**: LAgent lấy mẫu hành động cục bộ (sút, chuyền): $a_{t}^{l} \leftarrow \pi_{\theta^{l}}^{l}(o_{t}^{l})$. 


* **Bước 10**: Kiểm tra hành động của GAgent: Nếu ${a^{g}}_{t} = \text{STAY}$ thì thực thi $a_{t}^{l}$ trong $\epsilon_{l}$ (LAgent được phép hành động); Nếu ${a^{g}}_{t} \neq \text{STAY}$ thì thực thi ${a^{g}}_{t}$ trong $\epsilon_{l}$ (tác nhân di chuyển theo GAgent, LAgent không hành động). 


* **Bước 11**: Quan sát kết quả sau hành động $s_{t+1}, r_{t}^{l}, d \leftarrow \epsilon_{l}$, trong đó $r_{t}^{l}$ là phần thưởng cục bộ và $d$ là cờ kết thúc episode. 


* **Bước 12**: Lưu transition vào Buffer: $Buffer_{l} \leftarrow Buffer_{l} \cup \{(s_{t}, o_{t}^{l}, a_{t}^{l}, r_{t}^{l}, s_{t+1}, d)\}$. 


* **Bước 13**: Cập nhật trạng thái: $s_{t} \leftarrow s_{t+1}$. 


* **Bước 14**: Nếu $d = \text{true}$ thì kết thúc episode (break). 



**Cập Nhật COMA Trên Toàn Bộ Dữ Liệu Buffer** Với mỗi transition $(s, o^{l}, a^{l}, r^{l}, S^{\prime}, d)$ trong $Buffer_{l}$: 

* **Bước 15**: Tính giá trị Q từ Critic trung tâm của LAgent: $Q_{\Phi^{l}}^{central}(s,a^{l})$. 


* **Bước 16**: Tính baseline (kỳ vọng có trọng số theo chính sách LAgent):


$$b(s)=\sum_{a^{l_{}}}\pi_{\theta^{l}}^{l}(a^{l_{\prime}}|o^{l})Q_{\phi^{l}}^{central}(s,a^{l_{l}})$$





* **Bước 17**: Tính TD-error (sai số Temporal Difference) của LAgent:


$$\delta_{l} \leftarrow r^{l}+\gamma Q_{\phi^{l}}^{central}(s^{\prime},a^{l})-Q_{\phi^{l}}^{central}(s,a^{l})$$





* **Bước 18**: Cập nhật Critic $\phi^{l}$ bằng cách tối thiểu hóa $(\delta_{l})^{2}$. 


* **Bước 19**: Tính lợi thế phản thực tế (counterfactual advantage):


$$Adv^{l} \leftarrow Q_{\phi^{l}}^{central}(s,a^{l})-b(s)$$

* **Bước 20**: Cập nhật Actor $\theta^{l}$ theo gradient:


$$\theta^{l} \leftarrow \theta^{l}+\nabla_{\theta^{l}}\log \pi^{l}(a^{l}|o^{l})\cdot Adv^{l}$$





* **Bước 21**: Kết thúc vòng lặp cập nhật COMA và xóa sạch Buffer. 


$$Buffer_{l} \leftarrow \emptyset$$

* **Bước 22**: Kết thúc vòng lặp chính, sau $N_{ep}$ trả về bộ tham số đã huấn luyện bao gồm $\theta^{l}$ và $\phi^{l}$ đã được huấn luyện bởi LAgent.