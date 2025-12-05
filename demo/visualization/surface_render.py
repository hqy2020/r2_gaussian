"""
分割结果表面重建可视化

用于显示多器官分割结果的3D表面
"""
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from typing import Dict, List, Optional


# ACDC 心脏分割器官配置
ORGAN_CONFIGS = {
    1: {
        "name": "右心室 (RV)",
        "color": "rgba(255, 100, 100, 0.7)",  # 红色
        "opacity": 0.7
    },
    2: {
        "name": "心肌 (Myocardium)",
        "color": "rgba(100, 255, 100, 0.7)",  # 绿色
        "opacity": 0.7
    },
    3: {
        "name": "左心室 (LV)",
        "color": "rgba(100, 100, 255, 0.7)",  # 蓝色
        "opacity": 0.7
    }
}

# CT 重建颜色配置
CT_COLOR = "rgba(200, 200, 200, 0.5)"


def create_segmentation_surface(
    segmentation: np.ndarray,
    organ_configs: Dict = None,
    title: str = "Segmentation 3D View",
    step_size: int = 1
) -> go.Figure:
    """
    从分割结果创建3D表面可视化

    Args:
        segmentation: 分割mask [D, H, W]，值为类别ID
        organ_configs: 器官配置字典，None 使用默认 ACDC 配置
        title: 标题
        step_size: Marching Cubes 步长

    Returns:
        Plotly Figure 对象
    """
    if organ_configs is None:
        organ_configs = ORGAN_CONFIGS

    fig = go.Figure()

    # 为每个器官类别创建表面
    for label_id, config in organ_configs.items():
        mask = (segmentation == label_id).astype(np.float32)

        if mask.sum() < 100:  # 跳过太小的区域
            continue

        try:
            verts, faces, normals, _ = measure.marching_cubes(
                mask,
                level=0.5,
                step_size=step_size
            )
        except Exception as e:
            print(f"Warning: Could not create surface for {config['name']}: {e}")
            continue

        x, y, z = verts.T
        i, j, k = faces.T

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=config["color"],
            opacity=config["opacity"],
            name=config["name"],
            showlegend=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.5,
                specular=0.2,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=200, z=0)
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Slice)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        width=700,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_ct_surface(
    volume: np.ndarray,
    threshold: float = 0.3,
    title: str = "CT Reconstruction 3D View",
    step_size: int = 2
) -> go.Figure:
    """
    从CT重建体积创建3D表面可视化

    Args:
        volume: CT 体积 [nx, ny, nz]
        threshold: 等值面阈值
        title: 标题
        step_size: Marching Cubes 步长

    Returns:
        Plotly Figure 对象
    """
    # 归一化
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            vol_norm,
            level=threshold,
            step_size=step_size
        )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"无法生成表面 (阈值={threshold})",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, width=700, height=700)
        return fig

    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=CT_COLOR,
            opacity=0.6,
            name="CT Reconstruction",
            lighting=dict(
                ambient=0.4,
                diffuse=0.5,
                specular=0.3,
                roughness=0.3
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


def create_comparison_surface(
    seg_gt: np.ndarray,
    seg_pred: np.ndarray,
    organ_id: int = 1,
    title: str = "Segmentation Comparison"
) -> go.Figure:
    """
    创建分割结果对比可视化

    Args:
        seg_gt: 真值分割
        seg_pred: 预测分割
        organ_id: 要比较的器官ID
        title: 标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    # 真值表面
    mask_gt = (seg_gt == organ_id).astype(np.float32)
    if mask_gt.sum() > 100:
        try:
            verts, faces, _, _ = measure.marching_cubes(mask_gt, level=0.5, step_size=1)
            x, y, z = verts.T
            i, j, k = faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color="rgba(0, 255, 0, 0.5)",
                opacity=0.5,
                name="Ground Truth"
            ))
        except:
            pass

    # 预测表面
    mask_pred = (seg_pred == organ_id).astype(np.float32)
    if mask_pred.sum() > 100:
        try:
            verts, faces, _, _ = measure.marching_cubes(mask_pred, level=0.5, step_size=1)
            x, y, z = verts.T
            i, j, k = faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color="rgba(255, 0, 0, 0.5)",
                opacity=0.5,
                name="Prediction"
            ))
        except:
            pass

    organ_name = ORGAN_CONFIGS.get(organ_id, {}).get("name", f"Class {organ_id}")

    fig.update_layout(
        title=dict(text=f"{title} - {organ_name}", x=0.5),
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


# 测试代码
if __name__ == "__main__":
    # 创建测试分割数据
    seg = np.zeros((32, 64, 64), dtype=np.int32)

    # 创建几个简单的器官形状
    x = np.arange(64)
    y = np.arange(64)
    z = np.arange(32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 器官1: 球形
    dist1 = np.sqrt((X - 20) ** 2 + (Y - 32) ** 2 + (Z - 16) ** 2)
    seg[dist1 < 10] = 1

    # 器官2: 另一个球形
    dist2 = np.sqrt((X - 40) ** 2 + (Y - 32) ** 2 + (Z - 16) ** 2)
    seg[dist2 < 12] = 2

    # 器官3: 中心球形
    dist3 = np.sqrt((X - 32) ** 2 + (Y - 32) ** 2 + (Z - 16) ** 2)
    seg[(dist3 < 8) & (seg == 0)] = 3

    fig = create_segmentation_surface(seg, title="Test Segmentation Surface")
    fig.write_html("test_segmentation_surface.html")
    print("Saved to test_segmentation_surface.html")
