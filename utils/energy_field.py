import numpy as np

class EnergyFieldDefiner:
    def __init__(self, sigma_attract=0.9, sigma_repel=0.3, scale_attract=-0.9, scale_repel=-0.5):
        self.sigma_attract = sigma_attract
        self.sigma_repel = sigma_repel
        self.scale_attract = scale_attract
        self.scale_repel = scale_repel

    def _gaussian_kernel(self, pos, target, sigma, scale):
        """Tính giá trị phân phối Gaussian cho một điểm ảnh hưởng."""
        distance_sq = np.sum((pos - target)**2, axis=-1)
        return scale * np.exp(-distance_sq / (sigma**2))

    def _extract_position_and_params(self, item, default_sigma, default_scale):
        """
        Trích xuất vị trí và tham số từ item.
        - Nếu item là dict: {'position': [...], 'sigma': ..., 'scale': ...}
        - Nếu item là numpy array/list: chỉ vị trí, dùng tham số mặc định
        """
        if isinstance(item, dict):
            position = np.array(item['position'])
            sigma = item.get('sigma', default_sigma)
            scale = item.get('scale', default_scale)
        else:
            position = np.array(item)
            sigma = default_sigma
            scale = default_scale
        return position, sigma, scale

    def calculate_field_for_agents(self, agent_positions, goals, obstacles):
        """
        Tính toán Energy Field cho toàn bộ 11 tác tử cùng lúc (Vectorized).

        :param agent_positions: Mảng numpy shape (11, 2)
        :param goals: Danh sách các tọa độ mục tiêu hoặc dict có tham số riêng
                     Format: [position] hoặc [{'position': [...], 'sigma': ..., 'scale': ...}]
        :param obstacles: Danh sách tọa độ đối thủ hoặc dict có tham số riêng
        :return: Mảng energy values shape (11,)
        """
        energy_values = np.zeros(len(agent_positions))

        # Lực hút (Mang giá trị âm để tác tử có xu hướng giảm thiểu năng lượng / tối đa hóa phần thưởng âm)
        for goal in goals:
            position, sigma, scale = self._extract_position_and_params(
                goal, self.sigma_attract, self.scale_attract
            )
            energy_values -= self._gaussian_kernel(agent_positions, position, sigma, scale)

        # Lực đẩy (Mang giá trị dương để tránh né)
        for obs in obstacles:
            position, sigma, scale = self._extract_position_and_params(
                obs, self.sigma_repel, self.scale_repel
            )
            energy_values += self._gaussian_kernel(agent_positions, position, sigma, scale)

        return energy_values