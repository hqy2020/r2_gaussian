
# 本脚本用于绘制论文中 Fig.1 的三维体数据图像，需要有显示器或支持离屏渲染。

import numpy as np
import pyvista as pv


# 体数据（npy文件）路径
# volume_path = "/home/qyhu/Documents/r2_gaussian/output/4f5e679d-5/point_cloud/iteration_30000/vol_pred.npy"
volume_path = "/home/qyhu/Documents/r2_gaussian/output/3a20ea66-3/point_cloud/iteration_30000/vol_gt.npy"
# 渲染后图片保存路径
save_path = "volume-gt.png"


# Pyvista 可视化参数（可根据实际情况调整）
cpos = [  # 相机参数（位置、焦点、上方向）
    (-458.0015547298666, -207.26124611865254, 324.4699978427509),  # 相机位置
    (129.02644270914504, 111.50694084289574, 98.55158287937994),   # 观察目标点
    (0.0, 0.0, 79.59633400474613),                                 # 上方向
]
window_size = [800, 1000]  # 渲染窗口大小
colormap = "viridis"       # 体渲染配色方案


# 加载体数据（3D numpy 数组）
volume = np.load(volume_path)
# 只显示一半体素，便于观察内部结构
half_size = volume.shape[0] // 2
volume[:half_size, :, :] = 0  # 将前半部分置零，显示内部
clim = [0.0, 1.0]  # 体素值显示范围

# 创建 PyVista 渲染器，设置窗口大小、抗锯齿、离屏渲染（不弹窗，直接保存图片）
plotter = pv.Plotter(window_size=window_size, line_smoothing=True, off_screen=True)
# 添加体渲染对象
plotter.add_volume(volume, cmap=colormap, opacity="linear", clim=clim)
# 设置相机参数
plotter.camera_position = cpos
# 渲染并保存图片到指定路径
plotter.show(screenshot=save_path)
