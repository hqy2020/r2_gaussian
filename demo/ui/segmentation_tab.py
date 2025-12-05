"""
分割模块 UI

语义分割 Gradio 界面
"""
import os
import sys
import numpy as np
import gradio as gr
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# 确保项目路径在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.backend.segmentation import (
    SegmentationInference,
    SegmentationResult,
    ORGAN_NAMES,
    load_acdc_cases,
    get_acdc_data_path
)
from demo.backend.data_manager import data_manager
from demo.visualization.slice_viewer import create_slice_viewer
from demo.visualization.surface_render import create_segmentation_surface


# 全局状态
class SegmentationState:
    model: Optional[SegmentationInference] = None
    result: Optional[SegmentationResult] = None
    model_loaded: bool = False


state = SegmentationState()


def initialize_model(progress=gr.Progress()) -> str:
    """初始化分割模型（如果尚未加载）"""
    if state.model_loaded:
        return "模型已就绪"

    try:
        progress(0, desc="正在加载分割模型...")
        state.model = SegmentationInference()

        progress(0.5, desc="加载权重...")
        if state.model.load_model():
            state.model_loaded = True
            progress(1.0, desc="完成")
            return "分割模型加载成功!"
        else:
            return "模型加载失败"
    except Exception as e:
        return f"加载错误: {str(e)}"


def run_segmentation(case_name: str, progress=gr.Progress()) -> Tuple:
    """执行分割"""
    # 确保模型已加载
    if not state.model_loaded:
        init_result = initialize_model(progress)
        if not state.model_loaded:
            return None, None, None, init_result, None, None, None

    if not case_name:
        return None, None, None, "请选择一个案例", None, None, None

    try:
        progress(0, desc="正在分割...")

        # 获取数据路径
        data_path = get_acdc_data_path(case_name)
        if not os.path.exists(data_path):
            return None, None, None, f"数据文件不存在: {data_path}", None, None, None

        progress(0.3, desc="加载数据...")
        state.result = state.model.segment_from_h5(data_path)

        progress(0.6, desc="生成可视化...")

        # 创建切片可视化
        mid_slice = state.result.image.shape[0] // 2
        original_fig = create_slice_viewer(
            state.result.image,
            axis="z",
            title="原始图像"
        )

        # 创建分割结果可视化（转为float显示）
        seg_float = state.result.prediction.astype(np.float32)
        segmented_fig = create_slice_viewer(
            seg_float,
            axis="z",
            colorscale="Viridis",
            title="分割结果"
        )

        progress(0.8, desc="生成3D视图...")

        # 创建3D表面
        surface_fig = create_segmentation_surface(
            state.result.prediction,
            title="分割3D视图"
        )

        progress(1.0, desc="完成")

        # Dice 分数
        dice_rv = dice_myo = dice_lv = 0.0
        if state.result.dice_scores:
            dice_rv = round(state.result.dice_scores.get(1, 0), 4)
            dice_myo = round(state.result.dice_scores.get(2, 0), 4)
            dice_lv = round(state.result.dice_scores.get(3, 0), 4)

        status = (
            f"分割完成!\n"
            f"- 图像尺寸: {state.result.image.shape}\n"
            f"- 切片数量: {state.result.image.shape[0]}\n"
            f"- 检测到的类别: {np.unique(state.result.prediction).tolist()}"
        )

        return (
            original_fig,
            segmented_fig,
            surface_fig,
            status,
            dice_rv,
            dice_myo,
            dice_lv
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"分割失败: {str(e)}", None, None, None


def update_surface_view() -> Tuple:
    """更新3D表面视图"""
    if state.result is None:
        return None, "请先执行分割"

    try:
        fig = create_segmentation_surface(
            state.result.prediction,
            title="分割3D视图"
        )
        return fig, "3D视图更新成功"
    except Exception as e:
        return None, f"更新失败: {str(e)}"


def export_segmentation(format_type: str):
    """导出分割结果"""
    if state.result is None:
        return None

    try:
        if format_type == "NPY":
            path = tempfile.mktemp(suffix=".npy")
            np.save(path, state.result.prediction)
        elif format_type == "NIfTI":
            import SimpleITK as sitk
            path = tempfile.mktemp(suffix=".nii.gz")
            img = sitk.GetImageFromArray(state.result.prediction.astype(np.float32))
            img.SetSpacing((1, 1, 10))
            sitk.WriteImage(img, path)
        else:
            return None

        return path

    except Exception as e:
        print(f"导出失败: {e}")
        return None


def create_segmentation_tab():
    """创建分割模块 Tab"""

    # 获取 ACDC 案例列表
    acdc_cases = load_acdc_cases()

    with gr.Row():
        # 左侧：控制面板
        with gr.Column(scale=1):
            gr.Markdown("### 数据选择")

            # 案例选择
            case_dropdown = gr.Dropdown(
                choices=acdc_cases[:20] if len(acdc_cases) > 20 else acdc_cases,  # 限制显示数量
                label="选择 ACDC 案例",
                value=acdc_cases[0] if acdc_cases else None,
                info="心脏MRI分割数据"
            )

            # 控制按钮
            segment_btn = gr.Button("执行分割", variant="primary")

            # 状态显示
            status_text = gr.Textbox(
                label="状态",
                interactive=False,
                lines=4
            )

            gr.Markdown("---")
            gr.Markdown("### 分割指标")

            # Dice 分数显示
            with gr.Row():
                dice_rv = gr.Number(label="Dice (RV)", interactive=False)
            with gr.Row():
                dice_myo = gr.Number(label="Dice (MYO)", interactive=False)
            with gr.Row():
                dice_lv = gr.Number(label="Dice (LV)", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 器官说明")
            gr.Markdown("""
            - **RV**: 右心室 (红色)
            - **MYO**: 心肌 (绿色)
            - **LV**: 左心室 (蓝色)
            """)

            gr.Markdown("---")
            gr.Markdown("### 导出")

            with gr.Row():
                export_npy_btn = gr.Button("导出 NPY")
                export_nii_btn = gr.Button("导出 NIfTI")

            download_file = gr.File(label="下载文件", visible=True)

        # 右侧：可视化区域
        with gr.Column(scale=2):
            gr.Markdown("### 分割结果")

            with gr.Tabs():
                with gr.TabItem("切片浏览"):
                    with gr.Row():
                        original_plot = gr.Plot(label="原始图像")
                        segmented_plot = gr.Plot(label="分割结果")

                with gr.TabItem("3D可视化"):
                    surface_plot = gr.Plot(label="3D表面重建")
                    refresh_3d_btn = gr.Button("刷新3D视图")

    # 事件绑定
    segment_btn.click(
        run_segmentation,
        inputs=[case_dropdown],
        outputs=[
            original_plot,
            segmented_plot,
            surface_plot,
            status_text,
            dice_rv,
            dice_myo,
            dice_lv
        ]
    )

    refresh_3d_btn.click(
        update_surface_view,
        inputs=[],
        outputs=[surface_plot, status_text]
    )

    export_npy_btn.click(
        lambda: export_segmentation("NPY"),
        inputs=[],
        outputs=[download_file]
    )

    export_nii_btn.click(
        lambda: export_segmentation("NIfTI"),
        inputs=[],
        outputs=[download_file]
    )


# 测试代码
if __name__ == "__main__":
    with gr.Blocks(title="语义分割") as demo:
        gr.Markdown("# 语义分割测试")
        create_segmentation_tab()

    demo.launch(server_name="0.0.0.0", server_port=7861)
