import logging
import sys


def setup_logger(name, level=logging.INFO):
    """设置日志记录器，避免重复日志"""
    logger = logging.getLogger(name)

    # 如果logger已经有处理器，直接返回（避免重复添加）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 避免日志传播到根logger
    logger.propagate = False

    # 创建控制台处理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(handler)

    return logger
