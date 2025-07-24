import pickle
import numpy as np
from pprint import pprint

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_keys(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2
    return only_in_1, only_in_2, common_keys

def deep_compare(key, val1, val2, prefix=""):
    location = f"{prefix}.{key}" if prefix else key

    # 类型不一致
    if type(val1) != type(val2):
        return f"Type mismatch at '{location}': {type(val1)} vs {type(val2)}", False

    # 字典类型
    if isinstance(val1, dict):
        only1, only2, common = compare_keys(val1, val2)
        if only1 or only2:
            return f"Dict key mismatch at '{location}': only in 1: {only1}, only in 2: {only2}", False
        for k in common:
            msg, equal = deep_compare(k, val1[k], val2[k], prefix=location)
            if not equal:
                return msg, False
        return None, True

    # list 或 tuple
    if isinstance(val1, (list, tuple)):
        if len(val1) != len(val2):
            return f"Length mismatch at '{location}': {len(val1)} vs {len(val2)}", False
        for i, (a, b) in enumerate(zip(val1, val2)):
            msg, equal = deep_compare(f"[{i}]", a, b, prefix=location)
            if not equal:
                return msg, False
        return None, True

    # numpy 数组
    if isinstance(val1, np.ndarray):
        if val1.shape != val2.shape:
            return f"Shape mismatch at '{location}': {val1.shape} vs {val2.shape}", False
        if not np.allclose(val1, val2):
            return f"Array value mismatch at '{location}'", False
        return None, True

    # 其他值
    if val1 != val2:
        return f"Value mismatch at '{location}': {val1} vs {val2}", False

    return None, True

def main(path1, path2):
    print(f"Loading: {path1}")
    data1 = load_pickle(path1)
    print(f"Loading: {path2}")
    data2 = load_pickle(path2)

    if not isinstance(data1, dict) or not isinstance(data2, dict):
        print("Top-level object is not a dict.")
        print(f"Type1: {type(data1)}, Type2: {type(data2)}")
        return

    only1, only2, common = compare_keys(data1, data2)
    print("\n🔴 Keys only in first file:")
    pprint(only1)
    print("\n🔴 Keys only in second file:")
    pprint(only2)

    print("\n🟡 Differences in common keys:")
    has_diff = False
    identical_keys = []
    for key in common:
        msg, is_equal = deep_compare(key, data1[key], data2[key])
        if not is_equal:
            has_diff = True
            print(msg)
        else:
            identical_keys.append(key)

    if not has_diff:
        print("✅ No differences found in common keys.")

    print("\n🟢 Identical common keys:")
    pprint(identical_keys)

if __name__ == "__main__":
    path1 = "data/1/chest_50.pickle"
    path2 = "data/r2-sax-nerf/0_chest_cone.pickle"
    main(path1, path2)
