#!/usr/bin/env python3
"""
扩展版指标提取脚本：支持 5 方法 × 5 器官 × 3 视角的多方法对比实验
用法:
    python extract_metrics_all.py /root/experiments/results
    python extract_metrics_all.py /root/experiments/results --tsv output.tsv
    python extract_metrics_all.py /root/experiments/results --markdown
"""

import os
import yaml
import argparse
from collections import defaultdict

METHODS = ['R2GS', 'XGS', 'DNGS', 'CORGS', 'FSGS']
ORGANS = ['chest', 'foot', 'head', 'abdomen', 'pancreas']
VIEWS = ['3', '6', '9']


def extract_all_metrics(base_dir):
    """遍历所有方法/器官/视角，提取 psnr_2d 和 ssim_2d"""
    results = []

    for method in METHODS:
        for organ in ORGANS:
            for views in VIEWS:
                exp_dir = os.path.join(base_dir, method, f"{organ}_{views}views")
                eval_yaml = os.path.join(exp_dir, "eval", "iter_030000", "eval2d_render_test.yml")

                if not os.path.exists(eval_yaml):
                    results.append({
                        'method': method, 'organ': organ, 'views': f"{views}v",
                        'psnr_2d': None, 'ssim_2d': None, 'status': 'missing'
                    })
                    continue

                try:
                    with open(eval_yaml, 'r') as f:
                        data = yaml.safe_load(f)
                    results.append({
                        'method': method, 'organ': organ, 'views': f"{views}v",
                        'psnr_2d': round(data.get('psnr_2d', -1), 4),
                        'ssim_2d': round(data.get('ssim_2d', -1), 6),
                        'status': 'done'
                    })
                except Exception as e:
                    results.append({
                        'method': method, 'organ': organ, 'views': f"{views}v",
                        'psnr_2d': None, 'ssim_2d': None, 'status': f'error: {e}'
                    })

    return results


def print_table(results):
    """打印格式化表格"""
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            psnr = f"{r['psnr_2d']:.4f}" if r['psnr_2d'] is not None else '-'
            ssim = f"{r['ssim_2d']:.6f}" if r['ssim_2d'] is not None else '-'
            rows.append([r['method'], r['organ'], r['views'], psnr, ssim, r['status']])
        print(tabulate(rows, headers=["Method", "Organ", "Views", "PSNR_2D", "SSIM_2D", "Status"], tablefmt="grid"))
    except ImportError:
        print(f"{'Method':<8} {'Organ':<10} {'Views':<6} {'PSNR_2D':>10} {'SSIM_2D':>10} {'Status':<10}")
        print("-" * 60)
        for r in results:
            psnr = f"{r['psnr_2d']:.4f}" if r['psnr_2d'] is not None else '-'
            ssim = f"{r['ssim_2d']:.6f}" if r['ssim_2d'] is not None else '-'
            print(f"{r['method']:<8} {r['organ']:<10} {r['views']:<6} {psnr:>10} {ssim:>10} {r['status']:<10}")


def save_tsv(results, output_path):
    """保存 TSV 汇总表"""
    with open(output_path, 'w') as f:
        f.write("method\torgan\tviews\tpsnr_2d\tssim_2d\tstatus\n")
        for r in results:
            psnr = f"{r['psnr_2d']:.4f}" if r['psnr_2d'] is not None else ''
            ssim = f"{r['ssim_2d']:.6f}" if r['ssim_2d'] is not None else ''
            f.write(f"{r['method']}\t{r['organ']}\t{r['views']}\t{psnr}\t{ssim}\t{r['status']}\n")
    print(f"TSV saved to: {output_path}")


def save_markdown(results, output_path):
    """生成 Markdown 对比表（按器官和视角分组，方法为列）"""
    # 按 (organ, views) 分组
    grouped = defaultdict(dict)
    for r in results:
        key = (r['organ'], r['views'])
        grouped[key][r['method']] = r

    with open(output_path, 'w') as f:
        f.write("# Comparison Results: 5 Methods x 5 Organs x 3 Views\n\n")

        # PSNR 表
        f.write("## PSNR_2D\n\n")
        f.write(f"| Organ | Views | {' | '.join(METHODS)} |\n")
        f.write(f"|-------|-------| {' | '.join(['---'] * len(METHODS))} |\n")
        for organ in ORGANS:
            for views in VIEWS:
                key = (organ, f"{views}v")
                row = [organ, f"{views}v"]
                for method in METHODS:
                    r = grouped[key].get(method)
                    if r and r['psnr_2d'] is not None:
                        row.append(f"{r['psnr_2d']:.4f}")
                    else:
                        row.append("-")
                f.write(f"| {' | '.join(row)} |\n")

        f.write("\n## SSIM_2D\n\n")
        f.write(f"| Organ | Views | {' | '.join(METHODS)} |\n")
        f.write(f"|-------|-------| {' | '.join(['---'] * len(METHODS))} |\n")
        for organ in ORGANS:
            for views in VIEWS:
                key = (organ, f"{views}v")
                row = [organ, f"{views}v"]
                for method in METHODS:
                    r = grouped[key].get(method)
                    if r and r['ssim_2d'] is not None:
                        row.append(f"{r['ssim_2d']:.6f}")
                    else:
                        row.append("-")
                f.write(f"| {' | '.join(row)} |\n")

        # 统计
        f.write("\n## Summary\n\n")
        done_count = sum(1 for r in results if r['status'] == 'done')
        f.write(f"- Completed: {done_count}/{len(results)}\n")
        for method in METHODS:
            method_results = [r for r in results if r['method'] == method and r['psnr_2d'] is not None]
            if method_results:
                avg_psnr = sum(r['psnr_2d'] for r in method_results) / len(method_results)
                avg_ssim = sum(r['ssim_2d'] for r in method_results) / len(method_results)
                f.write(f"- {method}: avg PSNR={avg_psnr:.4f}, avg SSIM={avg_ssim:.6f} ({len(method_results)} experiments)\n")

    print(f"Markdown saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 5 方法 x 5 器官 x 3 视角的 PSNR/SSIM 指标")
    parser.add_argument("base_dir", type=str, help="实验结果根目录 (如 /root/experiments/results)")
    parser.add_argument("--tsv", type=str, default=None, help="输出 TSV 文件路径")
    parser.add_argument("--markdown", action='store_true', help="生成 Markdown 对比表")
    args = parser.parse_args()

    results = extract_all_metrics(args.base_dir)
    print_table(results)

    if args.tsv:
        save_tsv(results, args.tsv)
    else:
        # 默认保存到 base_dir/summary/
        summary_dir = os.path.join(args.base_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        save_tsv(results, os.path.join(summary_dir, "all_metrics.tsv"))

    if args.markdown:
        summary_dir = os.path.join(args.base_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        save_markdown(results, os.path.join(summary_dir, "comparison_table.md"))
