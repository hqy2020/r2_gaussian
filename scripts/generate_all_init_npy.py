import os

pickle_dir = "/home/qyhu/Documents/r2_gaussian/data/369"
organ_names = ["chest", "foot", "abdomen", "head", "pancreas"]
train_nums = [3, 6, 9]

for organ in organ_names:
    for num in train_nums:
        pickle_path = os.path.join(pickle_dir, f"{organ}_50_{num}views.pickle")
        cmd = f"python initialize_pcd.py --data {pickle_path}"
        print(f"🚀 初始化: {pickle_path}")
        os.system(cmd)
