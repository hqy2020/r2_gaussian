import os
import sys
import uuid
import logging
import os.path as osp
from argparse import Namespace
from datetime import datetime
import yaml

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.append("./")
from r2_gaussian.utils.cfg_utils import args2string


def setup_file_logger(log_path: str, name: str = "train"):
    """
    设置文件日志，输出到实验目录

    Args:
        log_path: 日志文件路径
        name: logger 名称

    Returns:
        logger: 配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件 handler
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台 handler（带时间戳）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%d/%m %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def prepare_output_and_logger(args):
    # Update model path if not specified
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = osp.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(osp.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save to yaml
    args_dict = vars(args)
    with open(osp.join(args.model_path, "cfg_args.yml"), "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        tb_writer.add_text("args", args2string(args_dict), global_step=0)
    else:
        print("Tensorboard not available: not logging progress")

    # 设置文件日志
    log_file = osp.join(args.model_path, "train.log")
    logger = setup_file_logger(log_file)
    logger.info(f"Output folder: {args.model_path}")
    logger.info(f"Log file: {log_file}")

    return tb_writer, logger
