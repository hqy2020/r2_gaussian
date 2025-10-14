import argparse, os, glob, yaml
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, help='实验目录（包含 eval/iter_xxxxx/）')
    ap.add_argument('--out', default=None, help='保存图片路径；为空则显示')
    args = ap.parse_args()

    ymls = sorted(glob.glob(os.path.join(args.exp_dir, 'eval', 'iter_*', 'eval2d_render_test.yml')))
    iters, psnr2d, ssim2d = [], [], []
    for y in ymls:
        it = int(os.path.basename(os.path.dirname(y)).split('_')[-1])
        with open(y, 'r') as f:
            d = yaml.safe_load(f) or {}
        if 'psnr_2d' in d and 'ssim_2d' in d:
            iters.append(it)
            psnr2d.append(float(d['psnr_2d']))
            ssim2d.append(float(d['ssim_2d']))

    if not iters:
        print('未找到指标或字段名不匹配（期望: psnr_2d, ssim_2d）'); return

    # 各自最优
    bi_psnr = max(range(len(iters)), key=lambda i: psnr2d[i])
    bi_ssim = max(range(len(iters)), key=lambda i: ssim2d[i])
    print(f'最佳(按 psnr_2d): iter={iters[bi_psnr]}, psnr_2d={psnr2d[bi_psnr]:.4f}')
    print(f'最佳(按 ssim_2d): iter={iters[bi_ssim]}, ssim_2d={ssim2d[bi_ssim]:.4f}')

    plt.figure(figsize=(8,4.5))
    plt.plot(iters, psnr2d, marker='o', linewidth=1.5, label='psnr_2d')
    plt.plot(iters, ssim2d, marker='s', linewidth=1.5, label='ssim_2d')

    # 用五角星标注各自最优点（不画竖线）
    plt.plot(iters[bi_psnr], psnr2d[bi_psnr], marker='*', markersize=14, color='C0', 
             label=f'best psnr_2d@{iters[bi_psnr]}')
    plt.plot(iters[bi_ssim], ssim2d[bi_ssim], marker='*', markersize=14, color='C1', 
             label=f'best ssim_2d@{iters[bi_ssim]}')

    plt.xlabel('Iteration'); plt.ylabel('Value'); plt.title('eval2d_render_test')
    plt.grid(True, ls='--', alpha=0.3); plt.legend()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print('Saved to', args.out)
    else:
        plt.show()

if __name__ == '__main__':
    main()