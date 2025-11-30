import hashlib
import logging
import time
from typing import Dict

from src.medvoice.utils.logger_utils import setup_logger

logger = setup_logger(name='CodeGenerator', level=logging.DEBUG)



class CodeGenerator:
    def __init__(self, prefix: str = "CN", code_length: int = 8):
        self.prefix = prefix
        self.code_length = code_length
        self.generated_codes: Dict[str, str] = {}  # 存储已生成的编码映射
        self.counter = 0

    def generate_code(self, chinese_name: str, use_timestamp: bool = True) -> str:
        """
        根据中文名称生成唯一编码

        Args:
            chinese_name: 中文名称
            use_timestamp: 是否使用时间戳确保唯一性，默认为True

        Returns:
            str: 生成的唯一编码
        """
        if not chinese_name or not chinese_name.strip():
            raise ValueError("中文名称不能为空")

        # 检查是否已经为该名称生成过编码
        if chinese_name in self.generated_codes:
            return self.generated_codes[chinese_name]

        # 基础编码生成
        base_code = self._generate_base_code(chinese_name)

        # 确保编码唯一性
        unique_code = self._ensure_uniqueness(base_code, chinese_name, use_timestamp)

        # 存储生成的编码
        self.generated_codes[chinese_name] = unique_code

        return unique_code

    def _generate_base_code(self, chinese_name: str) -> str:
        """生成基础编码"""
        # 方法1: 使用MD5哈希并截取
        md5_hash = hashlib.md5(chinese_name.encode('utf-8')).hexdigest()
        base_code = md5_hash[:self.code_length].upper()

        return base_code

    def _ensure_uniqueness(self, base_code: str, chinese_name: str, use_timestamp: bool) -> str:
        """确保编码的唯一性"""
        # 检查基础编码是否已存在
        existing_names = [name for name, code in self.generated_codes.items() if code.startswith(base_code)]

        if not existing_names:
            return f"{self.prefix}{base_code}"

        # 如果存在冲突，添加后缀确保唯一性
        if use_timestamp:
            # 使用时间戳作为后缀
            timestamp_suffix = str(int(time.time() * 1000))[-4:]  # 取时间戳后4位
            unique_code = f"{self.prefix}{base_code[:self.code_length - 4]}{timestamp_suffix}"
        else:
            # 使用计数器作为后缀
            self.counter += 1
            counter_suffix = str(self.counter).zfill(4)
            unique_code = f"{self.prefix}{base_code[:self.code_length - 4]}{counter_suffix}"

        # 递归检查确保新生成的编码也是唯一的
        if any(code == unique_code for code in self.generated_codes.values()):
            return self._ensure_uniqueness(base_code, chinese_name, use_timestamp)

        return unique_code
