#!/usr/bin/env python3
"""
检查 TensorBoard 事件文件中是否包含 CoR-GS metrics

用法:
    python check_tensorboard_corgs.py <path_to_events_file>
"""

import sys
import os

try:
    from tensorboard.backend.event_processing import event_accumulator
    import tensorflow as tf
except ImportError:
    print("错误: 需要安装 tensorboard 和 tensorflow")
    print("安装命令: pip install tensorboard tensorflow")
    sys.exit(1)


def check_corgs_metrics(events_file):
    """检查事件文件中的 CoR-GS metrics"""

    if not os.path.exists(events_file):
        print(f"错误: 文件不存在 {events_file}")
        return

    print(f"正在读取事件文件: {events_file}")
    print("=" * 80)

    # 加载事件文件
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()

    # 获取所有标量标签
    scalar_tags = ea.Tags()['scalars']

    print(f"总共找到 {len(scalar_tags)} 个标量 metrics\n")

    # 查找 CoR-GS 相关 metrics
    corgs_tags = [tag for tag in scalar_tags if 'corgs' in tag.lower()]

    if corgs_tags:
        print("✅ 找到 CoR-GS metrics:")
        print("-" * 80)
        for tag in corgs_tags:
            events = ea.Scalars(tag)
            print(f"\n标签: {tag}")
            print(f"  数据点数量: {len(events)}")
            if events:
                print(f"  首次记录: step={events[0].step}, value={events[0].value:.6f}")
                if len(events) > 1:
                    print(f"  最后记录: step={events[-1].step}, value={events[-1].value:.6f}")

                # 打印所有数据点
                print(f"  所有数据点:")
                for event in events:
                    print(f"    Step {event.step}: {event.value:.6f}")
        print("\n" + "=" * 80)
        print("✅ CoR-GS metrics 已成功记录！")
    else:
        print("⚠️ 未找到 CoR-GS metrics")
        print("-" * 80)
        print("可能的原因:")
        print("  1. log_corgs_metrics 函数未成功完成")
        print("  2. TensorBoard writer 未正确保存")
        print("  3. 计算过程中发生错误")
        print("\n建议检查训练日志中的 DEBUG-CORGS 输出")

    # 打印一些其他重要 metrics 作为对比
    print("\n" + "=" * 80)
    print("其他主要 metrics（前 10 个）:")
    print("-" * 80)
    for tag in scalar_tags[:10]:
        events = ea.Scalars(tag)
        print(f"{tag}: {len(events)} 个数据点")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_tensorboard_corgs.py <events_file>")
        print("\n示例:")
        print("  python check_tensorboard_corgs.py output/foot_corgs_test/events.out.tfevents.*")
        sys.exit(1)

    events_file = sys.argv[1]
    check_corgs_metrics(events_file)
