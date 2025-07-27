import os
import yaml
import argparse
from tabulate import tabulate

def extract_metrics(base_dir, use_few):
    organs = ['abdomen', 'chest', 'foot', 'head', 'pancreas']
    views = ['3views', '6views', '9views'] if use_few else [None]
    results = []

    for organ in organs:
        for view in views:
            if use_few:
                subfolder = f"{organ}_50_{view}.pickle"
                view_name = view
            else:
                subfolder = f"{organ}_50.pickle"
                view_name = "full"  # 可用作标识完整视角

            eval_yaml_path = os.path.join(base_dir, subfolder, "eval", "iter_030000", "eval2d_render_test.yml")

            if not os.path.exists(eval_yaml_path):
                print(f"[跳过] 文件不存在: {eval_yaml_path}")
                continue

            with open(eval_yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            psnr_2d = round(data.get('psnr_2d', -1), 4)
            ssim_2d = round(data.get('ssim_2d', -1), 4)
            results.append([organ, view_name, psnr_2d, ssim_2d])

    headers = ["Organ", "Views", "PSNR_2D", "SSIM_2D"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取PSNR和SSIM指标")
    parser.add_argument("base_dir", type=str, help="基础目录路径，例如：/home/qyhu/Documents/r2_gaussian/output/369")
    parser.add_argument("--few", action='store_true', help="是否处理少量视角（如 3views、6views、9views）")
    args = parser.parse_args()

    extract_metrics(args.base_dir, args.few)
