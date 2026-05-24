from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

# Một item goal/obstacle có thể là:
#   - dict: {'position': [...], 'sigma': float, 'scale': float}
#   - ndarray/list: chỉ vị trí, dùng tham số mặc định
FieldItem = Union[dict[str, Any], NDArray[np.float32]]


class EnergyFieldDefiner:
    """Tính toán trường năng lượng (Energy Field) cho các tác tử trên sân."""

    def __init__(
        self,
        sigma_attract:  float = 0.9,
        sigma_repel:    float = 0.3,
        scale_attract:  float = -0.9,
        scale_repel:    float = -0.5,
    ) -> None:
        self.sigma_attract:  float = sigma_attract
        self.sigma_repel:    float = sigma_repel
        self.scale_attract:  float = scale_attract
        self.scale_repel:    float = scale_repel

    def _gaussian_kernel(
        self,
        pos:    NDArray[np.float32],
        target: NDArray[np.float32],
        sigma:  float,
        scale:  float,
    ) -> NDArray[np.float32]:
        """Tính giá trị phân phối Gaussian cho một điểm ảnh hưởng."""
        distance_sq: NDArray[np.float32] = np.sum((pos - target) ** 2, axis=-1)
        return scale * np.exp(-distance_sq / (sigma ** 2))  # type: ignore[return-value]

    def _extract_position_and_params(
        self,
        item:          FieldItem,
        default_sigma: float,
        default_scale: float,
    ) -> tuple[NDArray[np.float32], float, float]:
        """
        Trích xuất vị trí và tham số từ item.

        - Nếu item là dict: {'position': [...], 'sigma': ..., 'scale': ...}
        - Nếu item là ndarray/list: chỉ vị trí, dùng tham số mặc định.
        """
        if isinstance(item, dict):
            position: NDArray[np.float32] = np.array(item["position"])
            sigma: float = item.get("sigma", default_sigma)
            scale: float = item.get("scale", default_scale)
        else:
            position = np.array(item)
            sigma = default_sigma
            scale = default_scale
        return position, sigma, scale

    def calculate_field_for_agents(
        self,
        agent_positions: NDArray[np.float32],   # shape (n_agents, 2)
        goals:           list[FieldItem],
        obstacles:       list[FieldItem],
    ) -> NDArray[np.float32]:
        """
        Tính toán Energy Field cho toàn bộ n tác tử cùng lúc (Vectorized).

        Args:
            agent_positions: Mảng vị trí shape (n_agents, 2).
            goals:      Danh sách mục tiêu hút (dict hoặc vị trí thuần).
            obstacles:  Danh sách chướng ngại vật đẩy (dict hoặc vị trí thuần).

        Returns:
            energy_values: Mảng giá trị năng lượng shape (n_agents,).
        """
        energy_values: NDArray[np.float32] = np.zeros(len(agent_positions))

        # Lực hút (âm → tác tử xu hướng giảm thiểu năng lượng)
        for goal in goals:
            position, sigma, scale = self._extract_position_and_params(
                goal, self.sigma_attract, self.scale_attract
            )
            energy_values += self._gaussian_kernel(agent_positions, position, sigma, scale)

        # Lực đẩy (dương → tránh né)
        for obs in obstacles:
            position, sigma, scale = self._extract_position_and_params(
                obs, self.sigma_repel, self.scale_repel
            )
            energy_values += self._gaussian_kernel(agent_positions, position, sigma, scale)

        return energy_values