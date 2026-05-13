import csv
import os

class CSVLogger:
    def __init__(self, filename, headers):
        """
        Khởi tạo Logger.
        :param filename: Đường dẫn tới file CSV (VD: 'results/phase1_log.csv')
        :param headers: Danh sách các cột (VD: ['Episode', 'Reward', 'Actor_Loss'])
        """
        self.filename = filename
        self.headers = headers
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Khởi tạo file và ghi header nếu file chưa tồn tại
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, data):
        """
        Ghi một dòng dữ liệu mới vào file CSV.
        :param data: Danh sách các giá trị tương ứng với headers
        """
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data)