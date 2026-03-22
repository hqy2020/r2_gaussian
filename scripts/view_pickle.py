import pickle
import argparse
import os
import pprint
import numpy as np

def print_nested_shapes(obj, prefix=""):
    """
    递归检查字典或数组中的字段，输出第一维长度（例如 'train -> projections': 3）
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            print_nested_shapes(v, prefix + f"{k} → ")
    elif isinstance(obj, (np.ndarray, list)):
        try:
            first_dim = len(obj)
            print(f"📏 {prefix[:-3]} 的第一维长度: {first_dim}")
        except:
            pass

def load_and_print_pickle(file_path):
    """加载并打印 pickle 文件的全部内容"""
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("\n✅ 文件成功加载！数据类型：", type(data))

    # 输出嵌套字段中的第一维长度
    print("\n📏 嵌套数组的第一维长度统计：")
    print_nested_shapes(data)

    print("\n📋 全部内容如下：\n")
    pp = pprint.PrettyPrinter(depth=5, width=120)
    pp.pprint(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="查看并打印完整的 pickle 文件内容，并输出嵌套数组长度")
    parser.add_argument('path', type=str, help="pickle 文件路径")
    args = parser.parse_args()

    load_and_print_pickle(args.path)
