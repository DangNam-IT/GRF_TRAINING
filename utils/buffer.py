import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.buffer = []
        
    def store(self, state, obs, action, reward, next_state, done):
        """Lưu trữ dữ liệu của 1 step (s, o, a, r, s', d)"""
        self.buffer.append((state, obs, action, reward, next_state, done))
        
    def clear(self):
        """Xóa buffer sau khi đã cập nhật COMA"""
        self.buffer = []
        
    def get_data(self):
        """Trả về toàn bộ dữ liệu trong buffer"""
        return self.buffer