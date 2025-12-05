"""
切片浏览可视化

使用 Plotly 创建交互式切片浏览器
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, List


def create_slice_viewer(
    volume: np.ndarray,
    axis: str = "z",
    colorscale: str = "gray",
    title: str = "Slice Viewer",
    slice_idx: Optional[int] = None
) -> go.Figure:
    """
    创建交互式切片浏览器

    Args:
        volume: 3D体积 [nx, ny, nz] 或 [D, H, W]
        axis: 切片方向 (x/y/z)
        colorscale: 颜色映射
        title: 标题
        slice_idx: 初始切片索引，None 表示中间

    Returns:
        Plotly Figure 对象
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(axis.lower(), 2)

    n_slices = volume.shape[ax]

    if slice_idx is None:
        slice_idx = n_slices // 2

    # 获取当前切片
    def get_slice(idx):
        if ax == 0:
            return volume[idx, :, :]
        elif ax == 1:
            return volume[:, idx, :]
        else:
            return volume[:, :, idx]

    # 创建初始图像
    initial_slice = get_slice(slice_idx)

    fig = go.Figure(
        data=go.Heatmap(
            z=initial_slice,
            colorscale=colorscale,
            showscale=True,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>"
        )
    )

    # 创建滑块步骤
    steps = []
    for i in range(n_slices):
        step = dict(
            method="update",
            args=[{"z": [get_slice(i)]}],
            label=str(i)
        )
        steps.append(step)

    sliders = [dict(
        active=slice_idx,
        currentvalue={"prefix": f"{axis.upper()} Slice: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
        xanchor="left"
    )]

    fig.update_layout(
        title=dict(text=title, x=0.5),
        sliders=sliders,
        width=600,
        height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_slice_comparison(
    volume1: np.ndarray,
    volume2: np.ndarray,
    title1: str = "Ground Truth",
    title2: str = "Prediction",
    axis: str = "z",
    colorscale: str = "gray",
    slice_idx: Optional[int] = None
) -> go.Figure:
    """
    创建并排切片对比视图

    Args:
        volume1: 第一个3D体积（如真值）
        volume2: 第二个3D体积（如预测）
        title1: 第一个标题
        title2: 第二个标题
        axis: 切片方向
        colorscale: 颜色映射
        slice_idx: 初始切片索引

    Returns:
        Plotly Figure 对象
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(axis.lower(), 2)

    n_slices = volume1.shape[ax]
    if slice_idx is None:
        slice_idx = n_slices // 2

    def get_slice(vol, idx):
        if ax == 0:
            return vol[idx, :, :]
        elif ax == 1:
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    # 计算差异
    diff = np.abs(volume1 - volume2)

    # 创建子图
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(title1, title2, "Difference"),
        horizontal_spacing=0.05
    )

    # 添加热图
    slice1 = get_slice(volume1, slice_idx)
    slice2 = get_slice(volume2, slice_idx)
    slice_diff = get_slice(diff, slice_idx)

    # 统一颜色范围
    vmin = min(volume1.min(), volume2.min())
    vmax = max(volume1.max(), volume2.max())

    fig.add_trace(
        go.Heatmap(z=slice1, colorscale=colorscale, zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=slice2, colorscale=colorscale, zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=slice_diff, colorscale="Reds", showscale=True),
        row=1, col=3
    )

    # 创建滑块
    steps = []
    for i in range(n_slices):
        s1 = get_slice(volume1, i)
        s2 = get_slice(volume2, i)
        sd = get_slice(diff, i)
        step = dict(
            method="update",
            args=[{"z": [s1, s2, sd]}],
            label=str(i)
        )
        steps.append(step)

    sliders = [dict(
        active=slice_idx,
        currentvalue={"prefix": f"{axis.upper()} Slice: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
        xanchor="left"
    )]

    fig.update_layout(
        title=dict(text="Volume Comparison", x=0.5),
        sliders=sliders,
        width=1200,
        height=500,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # 设置等比例
    for i in range(1, 4):
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)

    return fig


def create_segmentation_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    axis: str = "x",
    slice_idx: Optional[int] = None
) -> go.Figure:
    """
    创建原始图像与分割结果的同步切片浏览器

    Args:
        image: 原始3D图像 [D, H, W]
        prediction: 分割预测 [D, H, W]
        axis: 切片方向 (x/y/z)
        slice_idx: 初始切片索引

    Returns:
        Plotly Figure 对象，包含并排的两个图和同步滑块
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(axis.lower(), 0)

    n_slices = image.shape[ax]
    if slice_idx is None:
        slice_idx = n_slices // 2

    def get_slice(vol, idx):
        if ax == 0:
            return vol[idx, :, :]
        elif ax == 1:
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    # 创建子图 1x2 布局
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("原始图像", "分割结果"),
        horizontal_spacing=0.05
    )

    # 添加初始切片
    slice_img = get_slice(image, slice_idx)
    slice_pred = get_slice(prediction, slice_idx).astype(float)

    # 原始图像 - 灰度
    fig.add_trace(
        go.Heatmap(
            z=slice_img,
            colorscale="gray",
            showscale=False,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # 分割结果 - Viridis
    fig.add_trace(
        go.Heatmap(
            z=slice_pred,
            colorscale="Viridis",
            showscale=True,
            zmin=0,
            zmax=3,
            hovertemplate="x: %{x}<br>y: %{y}<br>class: %{z:.0f}<extra></extra>"
        ),
        row=1, col=2
    )

    # 创建滑块步骤 - 同步更新两个图
    steps = []
    for i in range(n_slices):
        s_img = get_slice(image, i)
        s_pred = get_slice(prediction, i).astype(float)
        step = dict(
            method="update",
            args=[{"z": [s_img, s_pred]}],
            label=str(i)
        )
        steps.append(step)

    sliders = [dict(
        active=slice_idx,
        currentvalue={"prefix": f"{axis.upper()} Slice: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
        xanchor="left"
    )]

    fig.update_layout(
        sliders=sliders,
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # 设置等比例并去除边框
    for i in range(1, 3):
        fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=i)
        fig.update_yaxes(showline=False, zeroline=False, showgrid=False,
                         scaleanchor=f"x{i}" if i > 1 else "x", scaleratio=1, row=1, col=i)

    return fig


def create_orthogonal_views(
    volume: np.ndarray,
    colorscale: str = "gray",
    title: str = "Orthogonal Views"
) -> go.Figure:
    """
    创建三视图（轴向/冠状/矢状）

    Args:
        volume: 3D体积 [D, H, W] 或 [nx, ny, nz]
        colorscale: 颜色映射
        title: 标题

    Returns:
        Plotly Figure 对象
    """
    d, h, w = volume.shape

    # 中间切片
    axial = volume[d // 2, :, :]      # Z方向
    coronal = volume[:, h // 2, :]    # Y方向
    sagittal = volume[:, :, w // 2]   # X方向

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Axial (Z)", "Coronal (Y)", "Sagittal (X)"),
        horizontal_spacing=0.05
    )

    vmin, vmax = volume.min(), volume.max()

    fig.add_trace(
        go.Heatmap(z=axial, colorscale=colorscale, zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=coronal, colorscale=colorscale, zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=sagittal, colorscale=colorscale, zmin=vmin, zmax=vmax, showscale=True),
        row=1, col=3
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=1200,
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    volume = np.random.rand(64, 128, 128)

    fig = create_slice_viewer(volume, axis="z", title="Test Slice Viewer")
    fig.write_html("test_slice_viewer.html")
    print("Saved to test_slice_viewer.html")

    fig2 = create_orthogonal_views(volume, title="Test Orthogonal Views")
    fig2.write_html("test_orthogonal.html")
    print("Saved to test_orthogonal.html")
