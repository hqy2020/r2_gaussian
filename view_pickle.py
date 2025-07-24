import pickle
import argparse
import os
import pprint

def load_and_print_pickle(file_path):
    """加载并打印 pickle 文件的全部内容"""
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("\n✅ 文件成功加载！数据类型：", type(data))
    print("\n📋 全部内容如下：\n")

    # 使用 pprint 格式化打印（防止 dict 太长一行）
    pp = pprint.PrettyPrinter(depth=5, width=120)
    pp.pprint(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="查看并打印完整的 pickle 文件内容")
    parser.add_argument('path', type=str, help="pickle 文件路径")
    args = parser.parse_args()

    load_and_print_pickle(args.path)
