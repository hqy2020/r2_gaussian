import os
import os.path as osp
import glob
import argparse

def main(args):
    source_path = args.source
    output_path = args.output
    device = args.device
    config_path = args.config

    # 只查找 .pickle 文件，按名称排序
    pickle_paths = sorted(glob.glob(osp.join(source_path, "*.pickle")))

    if len(pickle_paths) == 0:
        raise ValueError(f"{source_path} 下未找到任何 .pickle 文件！")

    for pickle_file in pickle_paths:
        case_name = osp.basename(pickle_file)  
        case_output_path = osp.join(output_path, case_name)

        # 判断 iter_030000 是否存在
        iter_030000_path = osp.join(case_output_path, "eval", "iter_030000")
        if not osp.exists(iter_030000_path):
            # 构造训练命令
            cmd = f"CUDA_VISIBLE_DEVICES={device} python train.py -s {pickle_file} -m {case_output_path}"
            if config_path:
                cmd += f" --config {config_path}"
            print(f"[训练中] {case_name}")
            os.system(cmd)
        else:
            print(f"[跳过] {case_name} 已完成训练，存在 iter_030000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/369", type=str, help="Path to ct dataset.")
    parser.add_argument("--output", default="output/369", type=str, help="Path to output.")
    parser.add_argument("--config", default=None, type=str, help="Path to config.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")

    args = parser.parse_args()
    main(args)
