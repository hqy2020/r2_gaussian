import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from r2_gaussian.utils.plot_utils import show_two_volume
from r2_gaussian.utils.general_utils import t2a

# 加载两个体积数据
vol2 = np.load("/home/qyhu/Documents/SAX-NeRF-master/logs/Lineformer/foot_50/2025_01_07_08_58_21/eval/epoch_03000/image_pred.npy")

# # 截断 vol2 第三维以匹配 vol1
# vol2 = vol2[:, :, :128]

# 显示 shape
print("Volume 1 shape:", vol1.shape)
print("Volume 2 shape:", vol2.shape)

# 显示并比较两个体积（调用原函数）
show_two_volume(vol1, vol2, title1="foot-gt", title2="foot-pred")

# ✅ 追加保存
plt.savefig("volume_compare.png")
plt.close()
print("✅ 图像已保存为 volume_compare.png")
