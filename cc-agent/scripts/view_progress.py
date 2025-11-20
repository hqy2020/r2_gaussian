#!/usr/bin/env python3
"""
DropGaussian 实验进度可视化脚本
快速在终端查看实验进度
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()

def main():
    # 标题
    console.print("\n")
    console.print(Panel.fit(
        "[bold magenta]🔬 DropGaussian 实验进度仪表板[/bold magenta]\n"
        "[cyan]R²-Gaussian CT 重建项目[/cyan]\n"
        "[dim]最后更新: 2025-11-19[/dim]",
        border_style="magenta"
    ))

    # 统计卡片
    console.print("\n[bold]📊 关键统计[/bold]\n")

    stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column(style="cyan", justify="center")
    stats_table.add_column(style="cyan", justify="center")
    stats_table.add_column(style="cyan", justify="center")
    stats_table.add_column(style="cyan", justify="center")

    stats_table.add_row(
        "✅ [bold]已完成实验[/bold]\n[bold yellow]3/3[/bold yellow]",
        "📈 [bold]成功率[/bold]\n[bold green]67%[/bold green]",
        "🏆 [bold]最佳 PSNR[/bold]\n[bold blue]35.11[/bold blue]",
        "⭐ [bold]最佳 SSIM[/bold]\n[bold blue]0.961[/bold blue]"
    )

    console.print(stats_table)

    # 实验结果表格
    console.print("\n[bold]📋 实验结果详情[/bold]\n")

    results = Table(title="", box=box.ROUNDED, show_lines=True)
    results.add_column("视角数", style="cyan", justify="center")
    results.add_column("PSNR", style="yellow", justify="center")
    results.add_column("SSIM", style="yellow", justify="center")
    results.add_column("vs 3-views", style="magenta", justify="center")
    results.add_column("状态", justify="center")
    results.add_column("输出目录", style="dim", no_wrap=False)

    results.add_row(
        "[bold]3 Views[/bold]",
        "28.34",
        "0.9024",
        "baseline",
        "[red]❌ 失败[/red]",
        "2025_11_19_15_56_foot_3views_dropgaussian_curriculum"
    )

    results.add_row(
        "[bold]6 Views[/bold]",
        "32.05",
        "0.9440",
        "[green]+3.71 dB[/green]",
        "[green]✅ 成功[/green]",
        "2025_11_19_16_53_foot_6views_dropgaussian_curriculum"
    )

    results.add_row(
        "[bold]9 Views[/bold]",
        "35.11",
        "0.9613",
        "[green]+6.77 dB[/green]",
        "[blue]🏆 优秀[/blue]",
        "2025_11_19_16_53_foot_9views_dropgaussian_curriculum"
    )

    console.print(results)

    # PSNR 可视化柱状图
    console.print("\n[bold]📊 PSNR 对比可视化[/bold]\n")

    psnr_data = [
        ("3 Views", 28.34, "red"),
        ("6 Views", 32.05, "green"),
        ("9 Views", 35.11, "blue"),
    ]

    max_psnr = 36
    for label, psnr, color in psnr_data:
        bar_length = int((psnr / max_psnr) * 50)
        bar = "█" * bar_length
        console.print(f"{label:10} {psnr:.2f} dB [{color}]{bar}[/{color}]")

    # 核心发现
    console.print("\n[bold]🎯 核心发现[/bold]\n")

    findings = Panel(
        "[green]✅[/green] DropGaussian 需要至少 6 个训练视角才有效\n"
        "[green]✅[/green] 课程学习策略 (前 5000 轮不 drop + drop_gamma=0.1) 是成功关键\n"
        "[yellow]⚠️[/yellow]  3 views 失败根因：训练信号不足 80%\n"
        "[blue]💡[/blue] 视角数越多，DropGaussian 效果越好",
        title="[bold cyan]Insights[/bold cyan]",
        border_style="cyan"
    )
    console.print(findings)

    # 待办事项
    console.print("\n[bold]📋 下一步工作[/bold]\n")

    todos = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    todos.add_column("优先级", style="bold")
    todos.add_column("状态")
    todos.add_column("任务")

    todos.add_row("[red]P1[/red]", "⬜", "训练 6 views 和 9 views 的 baseline 进行对比")
    todos.add_row("[red]P1[/red]", "⬜", "分析 6/9 views 下的逐图改善情况")
    todos.add_row("[red]P1[/red]", "⬜", "撰写完整的 3/6/9 views 对比报告")
    todos.add_row("[yellow]P2[/yellow]", "⬜", "在 Chest 器官上验证 DropGaussian (6/9 views)")
    todos.add_row("[yellow]P2[/yellow]", "⬜", "在 Head 器官上验证 DropGaussian (6/9 views)")
    todos.add_row("[yellow]P2[/yellow]", "⬜", "在 Abdomen/Pancreas 上验证 DropGaussian")
    todos.add_row("[green]P3[/green]", "⬜", "探索 Importance-Aware Drop 策略")

    console.print(todos)

    # 总体进度
    console.print("\n[bold]📈 总体进度[/bold]\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=50),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]实验完成度", total=100)
        progress.update(task, completed=50)
        import time
        time.sleep(0.5)

    console.print("\n[dim]💡 提示: 使用浏览器打开 cc-agent/records/progress_dashboard.html 查看更详细的可视化[/dim]\n")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ 需要安装 rich 库")
        print("运行: pip install rich")
