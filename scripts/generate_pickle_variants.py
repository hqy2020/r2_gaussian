import os
import pickle
import numpy as np

def generate_variant(input_path, output_dir, train_nums):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    total_views = data['numTrain']  # 一般为 50
    angles = np.array(data['train']['angles'])
    projections = np.array(data['train']['projections'])

    # 保留原始 val
    val_data = data['val']
    val_num = data['numVal']

    for train_num in train_nums:
        if train_num >= total_views:
            raise ValueError(f"train_num={train_num} should be less than total_views={total_views}")

        # 等间隔选择 train 索引（包含首尾）
        if train_num == 1:
            train_indices = [0]
        else:
            train_indices = [round(i * (total_views - 1) / (train_num - 1)) for i in range(train_num)]

        new_data = dict(data)  # 复制顶层结构
        new_data['numTrain'] = len(train_indices)
        new_data['numVal'] = val_num

        new_data['train'] = {
            'angles': angles[train_indices],
            'projections': projections[train_indices]
        }
        new_data['val'] = val_data  # 保持 val 不变

        basename = os.path.basename(input_path).replace('.pickle', '')
        output_name = f"{basename}_{train_num}views.pickle"
        output_path = os.path.join(output_dir, output_name)

        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(new_data, f)

        print(f"✅ Saved: {output_path}")


# ========== 使用方式 ==========
input_dir = "/home/qyhu/Documents/r2_gaussian/data/1"      # 原始 50 视角 pickle 所在目录
output_dir = "/home/qyhu/Documents/r2_gaussian/data/369"  # 生成的新文件存储目录
organ_names = ["chest", "foot", "abdomen", "head", "pancreas"]
train_nums = [3, 6, 9]

for organ in organ_names:
    input_path = os.path.join(input_dir, f"{organ}_50.pickle")
    generate_variant(input_path, output_dir, train_nums)
