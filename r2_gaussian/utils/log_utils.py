import os
import sys
import uuid
import os.path as osp
from argparse import Namespace
import yaml

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.append("./")
from r2_gaussian.utils.cfg_utils import args2string
from r2_gaussian.utils.unified_logger import init_logger, get_logger


def prepare_output_and_logger(args, method: str = "unknown"):
    """
    准备输出目录并初始化日志器

    Args:
        args: 命令行参数
        method: 方法名称（用于统一日志格式）

    Returns:
        tb_writer: TensorBoard writer（如果可用）
    """
    # Update model path if not specified
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = osp.join("./output/", unique_str[0:10])

    # Set up output folder
    os.makedirs(args.model_path, exist_ok=True)

    # 初始化统一日志器
    logger = init_logger(
        method=method,
        output_dir=args.model_path,
        use_tqdm=True,
        write_to_file=False,  # 由 shell tee 处理文件写入
    )
    logger.config(f"Output folder: {args.model_path}")

    # 保存配置文件
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
        logger.warn("Tensorboard not available: not logging progress")
    return tb_writer
