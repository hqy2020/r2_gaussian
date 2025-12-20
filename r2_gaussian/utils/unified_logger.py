"""
统一日志工具 - SPAGS 项目

日志格式：[HH:MM:SS][方法名][ITER xxxxx][级别] 消息

使用示例：
    from r2_gaussian.utils.unified_logger import get_logger, init_logger

    # 初始化（在训练开始时调用一次）
    init_logger(method="r2_gaussian", output_dir="output/exp1")

    # 获取 logger 并使用
    logger = get_logger()
    logger.info("训练开始")
    logger.config("SPS: 启用")
    logger.info("GAR 密化完成", iteration=1000)
    logger.eval("PSNR: 28.56", iteration=5000)
    logger.warn("内存使用较高")
    logger.error("CUDA OOM")
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional
from tqdm import tqdm


class LogLevel:
    """日志级别常量"""
    CONFIG = "CONFIG"
    INFO = "INFO"
    EVAL = "EVAL"
    WARN = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class UnifiedLogger:
    """统一日志管理器"""

    # 方法名固定宽度
    METHOD_WIDTH = 11
    # 迭代数固定宽度
    ITER_WIDTH = 5
    # 级别固定宽度
    LEVEL_WIDTH = 6

    def __init__(
        self,
        method: str = "unknown",
        output_dir: Optional[str] = None,
        log_file: str = "training.log",
        use_tqdm: bool = True,
        write_to_file: bool = False,  # 默认不写文件（由 shell tee 处理）
    ):
        """
        初始化日志器

        Args:
            method: 方法名称（如 r2_gaussian, xgaussian, naf 等）
            output_dir: 输出目录，用于保存日志文件
            log_file: 日志文件名
            use_tqdm: 是否使用 tqdm.write（避免与进度条冲突）
            write_to_file: 是否同时写入文件（默认 False，由 shell tee 处理）
        """
        self.method = method
        self.output_dir = output_dir
        self.log_file = log_file
        self.use_tqdm = use_tqdm
        self.write_to_file = write_to_file
        self._file_handle = None
        self._start_time = time.time()

        # 如果需要写文件，打开文件句柄
        if self.write_to_file and self.output_dir:
            log_path = os.path.join(self.output_dir, self.log_file)
            os.makedirs(self.output_dir, exist_ok=True)
            self._file_handle = open(log_path, "a", encoding="utf-8")

    def _format_time(self) -> str:
        """格式化时间戳（相对于训练开始的时间）"""
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_method(self) -> str:
        """格式化方法名（固定宽度）"""
        return self.method.ljust(self.METHOD_WIDTH)[:self.METHOD_WIDTH]

    def _format_iteration(self, iteration: Optional[int]) -> str:
        """格式化迭代数"""
        if iteration is None:
            return " " * (self.ITER_WIDTH + 5)  # "ITER " + 数字宽度
        return f"ITER {iteration:0{self.ITER_WIDTH}d}"

    def _format_level(self, level: str) -> str:
        """格式化日志级别"""
        return level.ljust(self.LEVEL_WIDTH)[:self.LEVEL_WIDTH]

    def _format_message(
        self,
        msg: str,
        level: str,
        iteration: Optional[int] = None,
    ) -> str:
        """组装完整的日志消息"""
        time_str = self._format_time()
        method_str = self._format_method()
        iter_str = self._format_iteration(iteration)
        level_str = self._format_level(level)
        return f"[{time_str}][{method_str}][{iter_str}][{level_str}] {msg}"

    def _output(self, formatted_msg: str):
        """输出日志消息"""
        # 控制台输出
        if self.use_tqdm:
            tqdm.write(formatted_msg)
        else:
            print(formatted_msg)

        # 文件输出（如果启用）
        if self._file_handle:
            self._file_handle.write(formatted_msg + "\n")
            self._file_handle.flush()

    def log(
        self,
        msg: str,
        level: str = LogLevel.INFO,
        iteration: Optional[int] = None,
    ):
        """通用日志方法"""
        formatted = self._format_message(msg, level, iteration)
        self._output(formatted)

    def info(self, msg: str, iteration: Optional[int] = None):
        """INFO 级别日志"""
        self.log(msg, LogLevel.INFO, iteration)

    def config(self, msg: str):
        """配置信息日志（无迭代数）"""
        self.log(msg, LogLevel.CONFIG, None)

    def eval(self, msg: str, iteration: Optional[int] = None):
        """评估结果日志"""
        self.log(msg, LogLevel.EVAL, iteration)

    def warn(self, msg: str, iteration: Optional[int] = None):
        """警告日志"""
        self.log(msg, LogLevel.WARN, iteration)

    def error(self, msg: str, iteration: Optional[int] = None):
        """错误日志"""
        self.log(msg, LogLevel.ERROR, iteration)

    def debug(self, msg: str, iteration: Optional[int] = None):
        """调试日志"""
        self.log(msg, LogLevel.DEBUG, iteration)

    def separator(self, char: str = "=", width: int = 70):
        """输出分隔线"""
        self.config(char * width)

    def section(self, title: str, char: str = "=", width: int = 70):
        """输出带标题的分隔区块"""
        self.separator(char, width)
        self.config(title)
        self.separator(char, width)

    def config_block(self, title: str, items: dict, char: str = "=", width: int = 70):
        """输出配置块

        Args:
            title: 块标题
            items: 配置项字典 {key: value}
            char: 分隔符字符
            width: 分隔线宽度
        """
        self.separator(char, width)
        self.config(title)
        for key, value in items.items():
            self.config(f"  {key}: {value}")
        self.separator(char, width)

    def set_method(self, method: str):
        """动态更改方法名"""
        self.method = method

    def reset_timer(self):
        """重置计时器"""
        self._start_time = time.time()

    def close(self):
        """关闭日志器"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        """析构时关闭文件句柄"""
        self.close()


# 全局 logger 实例
_global_logger: Optional[UnifiedLogger] = None


def init_logger(
    method: str = "unknown",
    output_dir: Optional[str] = None,
    log_file: str = "training.log",
    use_tqdm: bool = True,
    write_to_file: bool = False,
) -> UnifiedLogger:
    """
    初始化全局 logger

    Args:
        method: 方法名称
        output_dir: 输出目录
        log_file: 日志文件名
        use_tqdm: 是否使用 tqdm.write
        write_to_file: 是否写入文件

    Returns:
        UnifiedLogger 实例
    """
    global _global_logger
    _global_logger = UnifiedLogger(
        method=method,
        output_dir=output_dir,
        log_file=log_file,
        use_tqdm=use_tqdm,
        write_to_file=write_to_file,
    )
    return _global_logger


def get_logger() -> UnifiedLogger:
    """
    获取全局 logger 实例

    Returns:
        UnifiedLogger 实例，如果未初始化则返回默认实例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger()
    return _global_logger


def set_method(method: str):
    """设置全局 logger 的方法名"""
    get_logger().set_method(method)
