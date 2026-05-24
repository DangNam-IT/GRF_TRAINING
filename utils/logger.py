from __future__ import annotations

import csv
import os
from typing import Any, Sequence


class CSVLogger:
    """Ghi kết quả training vào file CSV theo từng episode."""

    def __init__(self, filename: str, headers: Sequence[str]) -> None:
        """
        Khởi tạo Logger.

        Args:
            filename: Đường dẫn tới file CSV (VD: 'results/phase1_log.csv').
            headers:  Danh sách các cột (VD: ['Episode', 'Reward', 'Actor_Loss']).
        """
        self.filename: str           = filename
        self.headers:  Sequence[str] = headers

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Khởi tạo file và ghi header nếu file chưa tồn tại
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, data: Sequence[Any]) -> None:
        """
        Ghi một dòng dữ liệu mới vào file CSV.

        Args:
            data: Danh sách các giá trị tương ứng với headers.
        """
        with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data)