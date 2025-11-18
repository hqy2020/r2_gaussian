#!/usr/bin/env python3
"""
FSGS v4+ 实验结果汇总脚本

功能：
1. 从各实验的 eval2d_render_test.yml 和 eval2d_render_train.yml 提取指标
2. 生成对比表格（Markdown 格式）
3. 识别最佳配置
4. 生成阶段 2 实验建议

作者：@deep-learning-tuning-expert
创建时间：2025-11-18
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional
import glob

# 实验列表（与 generate_v4_configs.py 一致）
EXPERIMENT_IDS = [
    "v4_tv_0.10",
    "v4_tv_0.12",
    "v4_k_5",
    "v4_tau_7.0",
    "v4_densify_10k",
    "v4_grad_3e-4",
    "v4_dssim_0.30",
    "v4_cap_180k"
]

# 基准配置（v2）
V2_BASELINE = {
    "name": "v2 (baseline)",
    "test_psnr": 28.50,
    "test_ssim": 0.9015,
    "train_psnr": 51.10,
    "generalization_gap": 22.60
}

# 输出目录
OUTPUT_BASE = "/home/qyhu/Documents/r2_ours/r2_gaussian/output"
REPORT_OUTPUT = "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/v4_results_summary.md"


def find_eval_file(exp_name: str, eval_type: str = "test") -> Optional[str]:
    """查找评估结果文件（支持多个迭代）"""
    pattern = f"{OUTPUT_BASE}/{exp_name}/eval/iter_*/eval2d_render_{eval_type}.yml"
    files = glob.glob(pattern)

    if not files:
        return None

    # 返回最新迭代的结果（假设 iter_030000 是最终结果）
    files.sort(reverse=True)
    return files[0]


def load_eval_metrics(eval_file: str) -> Dict:
    """从评估文件加载指标"""
    if not os.path.exists(eval_file):
        return None

    with open(eval_file, 'r') as f:
        data = yaml.safe_load(f)

    return {
        "psnr": data.get("psnr_2d", None),
        "ssim": data.get("ssim_2d", None)
    }


def collect_experiment_results(exp_id: str) -> Dict:
    """收集单个实验的所有指标"""
    exp_name = f"2025_11_19_foot_3views_fsgs_{exp_id}"

    # 查找测试集和训练集评估文件
    test_file = find_eval_file(exp_name, "test")
    train_file = find_eval_file(exp_name, "train")

    result = {
        "id": exp_id,
        "name": exp_name,
        "test_psnr": None,
        "test_ssim": None,
        "train_psnr": None,
        "generalization_gap": None,
        "status": "未运行"
    }

    if test_file:
        test_metrics = load_eval_metrics(test_file)
        if test_metrics:
            result["test_psnr"] = test_metrics["psnr"]
            result["test_ssim"] = test_metrics["ssim"]
            result["status"] = "已完成"

    if train_file:
        train_metrics = load_eval_metrics(train_file)
        if train_metrics:
            result["train_psnr"] = train_metrics["psnr"]

    # 计算泛化差距
    if result["train_psnr"] is not None and result["test_psnr"] is not None:
        result["generalization_gap"] = result["train_psnr"] - result["test_psnr"]

    return result


def classify_result(result: Dict) -> str:
    """根据成功标准分级"""
    if result["status"] != "已完成":
        return "-"

    psnr = result["test_psnr"]
    ssim = result["test_ssim"]
    gap = result["generalization_gap"]

    # S 级：重大突破
    if psnr >= 28.60 and ssim >= 0.905 and gap < 18:
        return "S 级 🏆"

    # A 级：显著改进
    if (psnr >= 28.55 or ssim >= 0.903) and gap < 20:
        return "A 级 ⭐⭐⭐"

    # B 级：边际改进
    if psnr >= 28.52 or gap < 21:
        return "B 级 ⭐⭐"

    # C 级：持平
    if 28.45 <= psnr < 28.52 and 21 <= gap <= 23:
        return "C 级 ⭐"

    # 失败
    return "失败 ❌"


def calculate_improvement(value: float, baseline: float) -> str:
    """计算改进幅度（带颜色标记）"""
    if value is None or baseline is None:
        return "-"

    diff = value - baseline
    if diff > 0:
        return f"+{diff:.3f} 🔼"
    elif diff < 0:
        return f"{diff:.3f} 🔽"
    else:
        return "0.000 ➡️"


def generate_markdown_report(all_results: List[Dict]) -> str:
    """生成 Markdown 格式的报告"""
    lines = [
        "# FSGS v4+ 阶段 1 实验结果汇总",
        "",
        f"**生成时间：** 2025-11-19",
        f"**实验数量：** {len(all_results)}",
        "",
        "---",
        "",
        "## 【核心结论】",
        "",
    ]

    # 统计完成情况
    completed = [r for r in all_results if r["status"] == "已完成"]
    s_grade = [r for r in completed if "S 级" in classify_result(r)]
    a_grade = [r for r in completed if "A 级" in classify_result(r)]

    lines.append(f"**完成进度：** {len(completed)}/{len(all_results)} 个实验")
    lines.append("")

    if s_grade:
        lines.append(f"**🏆 S 级成功（重大突破）：** {len(s_grade)} 个")
        for r in s_grade:
            lines.append(f"- **{r['id']}**：PSNR {r['test_psnr']:.2f} dB, SSIM {r['test_ssim']:.4f}, 泛化差距 {r['generalization_gap']:.2f} dB")
    elif a_grade:
        lines.append(f"**⭐⭐⭐ A 级成功（显著改进）：** {len(a_grade)} 个")
        for r in a_grade:
            lines.append(f"- **{r['id']}**：PSNR {r['test_psnr']:.2f} dB, SSIM {r['test_ssim']:.4f}, 泛化差距 {r['generalization_gap']:.2f} dB")
    else:
        lines.append("**⚠️ 尚无显著改进实验（S 级或 A 级）**")

    lines.extend([
        "",
        "---",
        "",
        "## 1. 详细结果对比表",
        "",
        "| 实验 ID | 测试 PSNR | vs v2 | 测试 SSIM | vs v2 | 训练 PSNR | 泛化差距 | vs v2 | 成功等级 | 状态 |",
        "|---------|----------|-------|----------|-------|----------|---------|-------|---------|------|",
    ])

    # v2 baseline 行
    lines.append(
        f"| **v2 (baseline)** | **{V2_BASELINE['test_psnr']:.2f}** | - | "
        f"**{V2_BASELINE['test_ssim']:.4f}** | - | **{V2_BASELINE['train_psnr']:.2f}** | "
        f"**{V2_BASELINE['generalization_gap']:.2f}** | - | A 级 ⭐⭐⭐ | 参考 |"
    )

    # 各实验结果行
    for r in all_results:
        if r["status"] != "已完成":
            lines.append(
                f"| {r['id']} | - | - | - | - | - | - | - | - | {r['status']} |"
            )
            continue

        psnr_diff = calculate_improvement(r["test_psnr"], V2_BASELINE["test_psnr"])
        ssim_diff = calculate_improvement(r["test_ssim"], V2_BASELINE["test_ssim"])
        gap_diff = calculate_improvement(r["generalization_gap"], V2_BASELINE["generalization_gap"])
        # 注意：泛化差距是负向指标，降低是好的
        if r["generalization_gap"] < V2_BASELINE["generalization_gap"]:
            gap_diff = f"-{V2_BASELINE['generalization_gap'] - r['generalization_gap']:.2f} 🔽（改善）"

        lines.append(
            f"| {r['id']} | {r['test_psnr']:.2f} | {psnr_diff} | "
            f"{r['test_ssim']:.4f} | {ssim_diff} | {r['train_psnr']:.2f} | "
            f"{r['generalization_gap']:.2f} | {gap_diff} | {classify_result(r)} | {r['status']} |"
        )

    lines.extend([
        "",
        "**图例说明：**",
        "- 🔼：指标提升（对 PSNR、SSIM 是好的）",
        "- 🔽：指标下降（对泛化差距是好的，表示过拟合减轻）",
        "- ➡️：持平",
        "",
        "---",
        "",
        "## 2. 最佳配置识别",
        "",
    ])

    # 找出最佳 PSNR
    if completed:
        best_psnr = max(completed, key=lambda x: x["test_psnr"])
        best_ssim = max(completed, key=lambda x: x["test_ssim"])
        best_gap = min(completed, key=lambda x: x["generalization_gap"])

        lines.append(f"**最佳测试 PSNR：** {best_psnr['id']} ({best_psnr['test_psnr']:.2f} dB)")
        lines.append(f"**最佳测试 SSIM：** {best_ssim['id']} ({best_ssim['test_ssim']:.4f})")
        lines.append(f"**最小泛化差距：** {best_gap['id']} ({best_gap['generalization_gap']:.2f} dB)")
    else:
        lines.append("**⚠️ 尚无完成的实验**")

    lines.extend([
        "",
        "---",
        "",
        "## 3. 阶段 2 实验建议（基于阶段 1 结果）",
        "",
    ])

    # 根据结果生成阶段 2 建议
    if s_grade or a_grade:
        top_exps = (s_grade + a_grade)[:3]  # 取前 3 个最佳实验
        lines.append("**推荐策略：** 组合最佳参数")
        lines.append("")
        lines.append("建议阶段 2 实验：")

        if len(top_exps) >= 2:
            lines.append(f"1. **v5_combo_1**：组合 {top_exps[0]['id']} + {top_exps[1]['id']} 的参数")
        if len(top_exps) >= 3:
            lines.append(f"2. **v5_combo_2**：组合 {top_exps[0]['id']} + {top_exps[2]['id']} 的参数")
        if len(top_exps) >= 3:
            lines.append(f"3. **v5_combo_all**：组合所有 A 级以上实验的参数（谨慎，可能参数冲突）")
    else:
        lines.append("**⚠️ 警告：** 阶段 1 无显著改进")
        lines.append("")
        lines.append("**建议应对方案：**")
        lines.append("- **选项 A（保守）：** 接受 v2 为最优，转向其他器官（Chest, Head, Abdomen）验证通用性")
        lines.append("- **选项 B（激进）：** 重新审视 v2，尝试 lambda_tv 0.06-0.07 或 k=4, τ=6.5")
        lines.append("- **选项 C（算法改进）：** 联系 3DGS 专家，引入 Dropout、Gradient Penalty 等新技术")

    lines.extend([
        "",
        "---",
        "",
        "## 4. 技术分析",
        "",
        "### 4.1 正则化强度影响",
        "",
    ])

    tv_exps = [r for r in completed if "tv" in r["id"]]
    if tv_exps:
        lines.append("| lambda_tv | 测试 PSNR | 泛化差距 | 观察 |")
        lines.append("|-----------|----------|---------|------|")
        for r in tv_exps:
            tv_val = "0.10" if "0.10" in r["id"] else "0.12"
            lines.append(f"| {tv_val} | {r['test_psnr']:.2f} | {r['generalization_gap']:.2f} | - |")
    else:
        lines.append("**⚠️ TV 正则化实验尚未完成**")

    lines.extend([
        "",
        "### 4.2 医学约束影响",
        "",
    ])

    med_exps = [r for r in completed if ("k_" in r["id"] or "tau" in r["id"])]
    if med_exps:
        lines.append("| 参数 | 测试 PSNR | 测试 SSIM | 观察 |")
        lines.append("|------|----------|----------|------|")
        for r in med_exps:
            param = "k=5" if "k_5" in r["id"] else "τ=7.0"
            lines.append(f"| {param} | {r['test_psnr']:.2f} | {r['test_ssim']:.4f} | - |")
    else:
        lines.append("**⚠️ 医学约束实验尚未完成**")

    lines.extend([
        "",
        "---",
        "",
        "## 【需要您的决策】",
        "",
        "### 问题 1：是否满意阶段 1 结果？",
        "- **选项 A：** 满意，批准阶段 2 实验（按上述建议组合）",
        "- **选项 B：** 不满意，执行应对方案（见第 3 节）",
        "- **选项 C：** 部分满意，调整阶段 2 方案（请说明）",
        "",
        "### 问题 2：下一步行动？",
        "- **选项 A：** 执行阶段 2 组合实验",
        "- **选项 B：** 在其他器官上验证最佳配置",
        "- **选项 C：** 探索更长训练迭代（50k, 100k）",
        "- **选项 D：** 联系 3DGS 专家，探索新算法",
        "",
        "---",
        "",
        "**✋ 等待用户确认后继续**",
    ])

    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("FSGS v4+ 实验结果汇总")
    print("=" * 60)
    print()

    all_results = []

    print("收集实验结果...")
    for exp_id in EXPERIMENT_IDS:
        print(f"  检查 {exp_id}...", end=" ")
        result = collect_experiment_results(exp_id)
        all_results.append(result)
        print(result["status"])

    print()
    print(f"✅ 共收集 {len(all_results)} 个实验，{len([r for r in all_results if r['status'] == '已完成'])} 个已完成")
    print()

    print("生成汇总报告...")
    report = generate_markdown_report(all_results)

    # 保存报告
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 汇总报告已保存：{REPORT_OUTPUT}")
    print()
    print("=" * 60)
    print("快速查看结果：")
    print("=" * 60)

    for r in all_results:
        if r["status"] == "已完成":
            grade = classify_result(r)
            print(f"  {r['id']:20s} | PSNR {r['test_psnr']:.2f} | SSIM {r['test_ssim']:.4f} | 差距 {r['generalization_gap']:.2f} dB | {grade}")

    print()
    print("详细分析请查看：", REPORT_OUTPUT)


if __name__ == "__main__":
    main()
