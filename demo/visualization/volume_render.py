"""
体绘制可视化

使用 Plotly 和 Marching Cubes 创建3D体积可视化
"""
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from typing import Optional, Tuple, List


def create_volume_render(
    volume: np.ndarray,
    threshold: float = 0.3,
    opacity: float = 0.3,
    colorscale: str = "Viridis",
    title: str = "Volume Rendering",
    step_size: int = 2
) -> go.Figure:
    """
    创建体绘制可视化 (基于等值面)

    Args:
        volume: 3D体积 [nx, ny, nz]
        threshold: 等值面阈值 (0-1 归一化后)
        opacity: 透明度
        colorscale: 颜色映射
        title: 标题
        step_size: Marching Cubes 步长（越大越快但越粗糙）

    Returns:
        Plotly Figure 对象
    """
    # 归一化
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # 使用 Marching Cubes 提取等值面
    try:
        verts, faces, normals, values = measure.marching_cubes(
            vol_norm,
            level=threshold,
            step_size=step_size
        )
    except Exception as e:
        # 如果失败，返回提示
        fig = go.Figure()
        fig.add_annotation(
            text=f"无法生成等值面 (阈值={threshold})",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, width=700, height=700)
        return fig

    # 创建 Mesh3d
    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=opacity,
            colorscale=colorscale,
            intensity=values,
            showscale=True,
            colorbar=dict(title="Intensity"),
            name="Volume",
            lighting=dict(
                ambient=0.4,
                diffuse=0.5,
                specular=0.1,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=0)
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_multi_threshold_volume(
    volume: np.ndarray,
    thresholds: List[float] = [0.2, 0.5, 0.8],
    colors: List[str] = ["blue", "green", "red"],
    opacities: List[float] = [0.1, 0.2, 0.5],
    title: str = "Multi-threshold Volume"
) -> go.Figure:
    """
    创建多阈值体绘制

    Args:
        volume: 3D体积
        thresholds: 阈值列表
        colors: 颜色列表
        opacities: 透明度列表
        title: 标题

    Returns:
        Plotly Figure 对象
    """
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    fig = go.Figure()

    for thresh, color, opacity in zip(thresholds, colors, opacities):
        try:
            verts, faces, _, values = measure.marching_cubes(
                vol_norm, level=thresh, step_size=2
            )
            x, y, z = verts.T
            i, j, k = faces.T

            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=opacity,
                color=color,
                name=f"Threshold {thresh:.2f}"
            ))
        except:
            continue

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=700,
        height=700,
        showlegend=True
    )

    return fig


def create_volume_scatter(
    volume: np.ndarray,
    threshold: float = 0.3,
    sample_rate: float = 0.1,
    colorscale: str = "Viridis",
    title: str = "Volume Points"
) -> go.Figure:
    """
    创建体积点云可视化（适用于稀疏数据）

    Args:
        volume: 3D体积
        threshold: 显示阈值
        sample_rate: 采样率
        colorscale: 颜色映射
        title: 标题

    Returns:
        Plotly Figure 对象
    """
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # 获取高于阈值的点
    mask = vol_norm > threshold
    indices = np.where(mask)

    # 采样
    n_points = len(indices[0])
    if n_points > 10000:
        sample_idx = np.random.choice(n_points, int(n_points * sample_rate), replace=False)
        x = indices[0][sample_idx]
        y = indices[1][sample_idx]
        z = indices[2][sample_idx]
        values = vol_norm[x, y, z]
    else:
        x, y, z = indices
        values = vol_norm[mask]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=values,
                colorscale=colorscale,
                opacity=0.6,
                colorbar=dict(title="Intensity")
            ),
            name="Volume Points"
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=700,
        height=700
    )

    return fig


# 测试代码
if __name__ == "__main__":
    # 创建测试数据 - 一个球形
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    z = np.linspace(-1, 1, 64)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    volume = np.exp(-(X**2 + Y**2 + Z**2) / 0.3)

    fig = create_volume_render(volume, threshold=0.3, title="Test Volume Render")
    fig.write_html("test_volume_render.html")
    print("Saved to test_volume_render.html")
