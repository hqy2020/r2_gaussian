"""
重建模块 UI

CT 三维重建 Gradio 界面
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

from demo.backend.reconstruction import ReconstructionInference, ReconstructionResult
from demo.backend.data_manager import data_manager
from demo.visualization.slice_viewer import create_slice_viewer, create_slice_comparison
from demo.visualization.volume_render import create_volume_render
from demo.visualization.surface_render import create_ct_surface


# 全局状态
class ReconstructionState:
    model: Optional[ReconstructionInference] = None
    result: Optional[ReconstructionResult] = None
    current_preset: Optional[str] = None


state = ReconstructionState()


def load_model(preset_name: str, progress=gr.Progress()) -> str:
    """加载重建模型"""
    if not preset_name:
        return "请选择一个预设案例"

    progress(0, desc="正在加载模型...")

    preset = data_manager.get_reconstruction_preset(preset_name)
    if preset is None:
        return f"未找到预设: {preset_name}"

    if not os.path.exists(preset.model_path):
        return f"模型路径不存在: {preset.model_path}"

    if not os.path.exists(preset.data_path):
        return f"数据路径不存在: {preset.data_path}"

    try:
        progress(0.3, desc="初始化模型...")
        state.model = ReconstructionInference(preset.model_path, preset.data_path)

        progress(0.6, desc="加载检查点...")
        if state.model.load_model():
            state.current_preset = preset_name
            info = state.model.get_sample_info()
            progress(1.0, desc="完成")
            return (
                f"模型加载成功!\n"
                f"- 迭代次数: {info['loaded_iter']}\n"
                f"- 高斯数量: {info['n_gaussians']:,}\n"
                f"- 体积尺寸: {info['volume_shape']}\n"
                f"- 训练视角: {info['n_train_views']}\n"
                f"- 测试视角: {info['n_test_views']}"
            )
        else:
            return "模型加载失败"
    except Exception as e:
        return f"加载错误: {str(e)}"


def run_reconstruction(progress=gr.Progress()) -> Tuple:
    """执行重建"""
    if state.model is None:
        return None, "请先加载模型", None, None

    try:
        progress(0, desc="正在重建...")
        progress(0.5, desc="执行3D查询...")

        state.result = state.model.reconstruct()

        progress(0.8, desc="生成可视化...")

        # 默认显示切片浏览器
        fig = create_slice_viewer(
            state.result.volume,
            axis="z",
            title="重建结果 - 切片浏览"
        )

        progress(1.0, desc="完成")

        psnr = state.result.psnr if state.result.psnr else 0
        ssim = state.result.ssim if state.result.ssim else 0

        return (
            fig,
            f"重建完成! 体积尺寸: {state.result.volume.shape}",
            round(psnr, 4),
            round(ssim, 4)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"重建失败: {str(e)}", None, None


def update_visualization(viz_type: str, threshold: float = 0.3) -> Tuple:
    """更新可视化类型"""
    if state.result is None:
        return None, "请先执行重建"

    try:
        if viz_type == "切片浏览":
            fig = create_slice_viewer(
                state.result.volume,
                axis="z",
                title="重建结果 - 切片浏览"
            )
        elif viz_type == "体绘制":
            fig = create_volume_render(
                state.result.volume,
                threshold=threshold,
                title="重建结果 - 体绘制"
            )
        elif viz_type == "表面重建":
            fig = create_ct_surface(
                state.result.volume,
                threshold=threshold,
                title="重建结果 - 表面重建"
            )
        elif viz_type == "GT对比" and state.result.volume_gt is not None:
            fig = create_slice_comparison(
                state.result.volume_gt,
                state.result.volume,
                title1="Ground Truth",
                title2="Prediction"
            )
        else:
            return None, "不支持的可视化类型"

        return fig, f"可视化类型: {viz_type}"

    except Exception as e:
        return None, f"可视化失败: {str(e)}"


def export_volume(format_type: str):
    """导出体积数据"""
    if state.result is None:
        return None

    try:
        # 创建临时文件
        if format_type == "NPY":
            path = tempfile.mktemp(suffix=".npy")
            np.save(path, state.result.volume)
        elif format_type == "NIfTI":
            import SimpleITK as sitk
            path = tempfile.mktemp(suffix=".nii.gz")
            img = sitk.GetImageFromArray(state.result.volume.transpose(2, 0, 1))
            sitk.WriteImage(img, path)
        else:
            return None

        return path

    except Exception as e:
        print(f"导出失败: {e}")
        return None


def create_reconstruction_tab():
    """创建重建模块 Tab"""

    with gr.Row():
        # 左侧：控制面板
        with gr.Column(scale=1):
            gr.Markdown("### 数据选择")

            # 预设案例选择
            preset_dropdown = gr.Dropdown(
                choices=data_manager.get_reconstruction_preset_names(),
                label="选择预设案例",
                value=None,
                info="选择预训练好的模型和对应数据"
            )

            # 预设描述
            preset_info = gr.Textbox(
                label="案例描述",
                interactive=False,
                lines=2
            )

            # 控制按钮
            with gr.Row():
                load_btn = gr.Button("加载模型", variant="primary", scale=2)
                recon_btn = gr.Button("执行重建", variant="secondary", scale=2)

            # 状态显示
            status_text = gr.Textbox(
                label="状态",
                interactive=False,
                lines=6
            )

            gr.Markdown("---")
            gr.Markdown("### 可视化选项")

            viz_type = gr.Radio(
                choices=["切片浏览", "体绘制", "表面重建", "GT对比"],
                value="切片浏览",
                label="可视化类型"
            )

            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="等值面阈值",
                info="用于体绘制和表面重建"
            )

            update_viz_btn = gr.Button("更新可视化")

            gr.Markdown("---")
            gr.Markdown("### 导出")

            with gr.Row():
                export_npy_btn = gr.Button("导出 NPY")
                export_nii_btn = gr.Button("导出 NIfTI")

            download_file = gr.File(label="下载文件", visible=True)

        # 右侧：可视化区域
        with gr.Column(scale=2):
            gr.Markdown("### 可视化结果")

            # 主可视化区域
            viz_plot = gr.Plot(label="3D 可视化")

            # 指标显示
            with gr.Row():
                psnr_display = gr.Number(label="PSNR (dB)", interactive=False)
                ssim_display = gr.Number(label="SSIM", interactive=False)

    # 事件绑定
    def update_preset_info(preset_name):
        if preset_name:
            preset = data_manager.get_reconstruction_preset(preset_name)
            if preset:
                return preset.description
        return ""

    preset_dropdown.change(
        update_preset_info,
        inputs=[preset_dropdown],
        outputs=[preset_info]
    )

    load_btn.click(
        load_model,
        inputs=[preset_dropdown],
        outputs=[status_text]
    )

    recon_btn.click(
        run_reconstruction,
        inputs=[],
        outputs=[viz_plot, status_text, psnr_display, ssim_display]
    )

    update_viz_btn.click(
        update_visualization,
        inputs=[viz_type, threshold_slider],
        outputs=[viz_plot, status_text]
    )

    export_npy_btn.click(
        lambda: export_volume("NPY"),
        inputs=[],
        outputs=[download_file]
    )

    export_nii_btn.click(
        lambda: export_volume("NIfTI"),
        inputs=[],
        outputs=[download_file]
    )


# 测试代码
if __name__ == "__main__":
    with gr.Blocks(title="CT 三维重建") as demo:
        gr.Markdown("# CT 三维重建测试")
        create_reconstruction_tab()

    demo.launch(server_name="0.0.0.0", server_port=7860)
