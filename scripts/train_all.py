
# 批量训练所有案例的脚本。

import os
import os.path as osp
import glob
import subprocess
import argparse



# 主函数，批量遍历 source 目录下的所有案例文件夹，依次调用 train.py 进行训练
def main(args):
    source_path = args.source  # 数据集根目录
    output_path = args.output  # 输出结果根目录
    device = args.device       # GPU 设备编号
    config_path = args.config  # 配置文件路径

    # 获取所有案例文件夹路径，按名称排序
    case_paths = sorted(glob.glob(osp.join(source_path, "*")))

    # 如果没有找到任何案例，抛出异常
    if len(case_paths) == 0:
        raise ValueError(f"{case_paths} find no folder!")

    # 遍历每个案例文件夹，逐个训练
    for case_path in case_paths:
        case_name = osp.basename(case_path)  # 案例名称
        case_output_path = f"{output_path}/{case_name}"  # 当前案例的输出目录
        # 如果输出目录不存在，说明还未训练过该案例
        if not osp.exists(case_output_path):
            # 构造训练命令，指定 GPU、输入、输出、可选配置
            cmd = f"CUDA_VISIBLE_DEVICES={device} python train.py -s {case_path} -m {case_output_path}"
            if config_path:
                cmd += f" --config {config_path}"
            # 执行训练命令
            os.system(cmd)



if __name__ == "__main__":
    # 命令行参数解析
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="Path to ct dataset.")  # 数据集路径
    parser.add_argument("--output", default="output/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="Path to output.")    # 输出路径
    parser.add_argument("--config", default=None, type=str, help="Path to config.")  # 配置文件路径
    parser.add_argument("--device", default=0, type=int, help="GPU device.")         # GPU 设备编号
    # fmt: on

    args = parser.parse_args()  # 解析参数
    main(args)  # 执行主流程
