"""
智能辅助诊断平台 - 简化版应用
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import gradio as gr

from demo.backend.reconstruction import ReconstructionInference
from demo.backend.segmentation import SegmentationInference, load_acdc_cases, get_acdc_data_path
from demo.backend.data_manager import data_manager
from demo.visualization.slice_viewer import create_slice_viewer, create_segmentation_comparison
from demo.visualization.volume_render import create_volume_render
from demo.visualization.surface_render import create_segmentation_surface, create_ct_surface
from demo.visualization.sam_prompt_viz import generate_sam_prompt_examples


# 全局状态
recon_model = None
recon_result = None
seg_model = None
seg_result = None


def load_reconstruction_model(preset_name):
    """加载重建模型"""
    global recon_model
    if not preset_name:
        return "请选择预设案例"

    preset = data_manager.get_reconstruction_preset(preset_name)
    if not preset:
        return f"未找到预设: {preset_name}"

    try:
        recon_model = ReconstructionInference(preset.model_path, preset.data_path)
        if recon_model.load_model():
            info = recon_model.get_sample_info()
            return f"加载成功! 高斯数: {info['n_gaussians']}, 体积: {info['volume_shape']}"
        return "加载失败"
    except Exception as e:
        return f"错误: {str(e)}"


def run_reconstruction():
    """执行重建"""
    global recon_model, recon_result
    if recon_model is None:
        return None, "请先加载模型", 0, 0

    try:
        recon_result = recon_model.reconstruct()
        fig = create_slice_viewer(recon_result.volume, axis="z", title="重建结果")
        psnr = recon_result.psnr or 0
        ssim = recon_result.ssim or 0
        return fig, "重建完成!", round(psnr, 4), round(ssim, 4)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"重建失败: {str(e)}", 0, 0


def update_recon_viz(viz_type, threshold):
    """更新重建可视化"""
    global recon_result
    if recon_result is None:
        return None

    try:
        if viz_type == "切片浏览":
            return create_slice_viewer(recon_result.volume, axis="z", title="切片浏览")
        elif viz_type == "体绘制":
            return create_volume_render(recon_result.volume, threshold=threshold, title="体绘制")
        elif viz_type == "表面重建":
            return create_ct_surface(recon_result.volume, threshold=threshold, title="表面重建")
    except Exception as e:
        return None


def run_segmentation(case_name):
    """执行分割"""
    global seg_model, seg_result

    if not case_name:
        return None, None, None, None, "请选择案例"

    try:
        if seg_model is None:
            seg_model = SegmentationInference()
            seg_model.load_model()

        data_path = get_acdc_data_path(case_name)
        seg_result = seg_model.segment_from_h5(data_path)

        # 创建联动切片浏览器（原始图像和分割结果并排，共用滑块）
        comparison_fig = create_segmentation_comparison(
            seg_result.image,
            seg_result.prediction,
            axis="x"
        )

        # 生成 SAM 提示效果示例图
        sam_examples = generate_sam_prompt_examples(
            seg_result.image,
            seg_result.prediction,
            axis=0
        )

        return (
            comparison_fig,
            sam_examples["point"],
            sam_examples["box"],
            sam_examples["mask"],
            "分割完成!"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"错误: {str(e)}"


def create_app():
    """创建应用"""
    with gr.Blocks(title="智能辅助诊断平台", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # 智能辅助诊断平台
        ### 面向稀疏数据的医学影像场景理解和重建方法研究
        """)

        with gr.Tabs():
            # CT重建标签页
            with gr.TabItem("CT 三维重建"):
                with gr.Row():
                    with gr.Column(scale=1):
                        recon_preset = gr.Dropdown(
                            choices=data_manager.get_reconstruction_preset_names(),
                            label="选择预设案例"
                        )
                        load_btn = gr.Button("加载模型", variant="primary")
                        recon_btn = gr.Button("执行重建")
                        recon_status = gr.Textbox(label="状态", lines=3)

                        viz_type = gr.Radio(
                            choices=["切片浏览", "体绘制", "表面重建"],
                            value="切片浏览",
                            label="可视化类型"
                        )
                        threshold = gr.Slider(0.1, 0.9, 0.3, label="阈值")
                        update_btn = gr.Button("更新可视化")

                    with gr.Column(scale=2):
                        recon_plot = gr.Plot(label="可视化")
                        with gr.Row():
                            psnr = gr.Number(label="PSNR")
                            ssim = gr.Number(label="SSIM")

                load_btn.click(
                    fn=load_reconstruction_model,
                    inputs=[recon_preset],
                    outputs=[recon_status],
                    api_name="load_model"
                )
                recon_btn.click(
                    fn=run_reconstruction,
                    inputs=[],
                    outputs=[recon_plot, recon_status, psnr, ssim],
                    api_name="run_reconstruction"
                )
                update_btn.click(
                    fn=update_recon_viz,
                    inputs=[viz_type, threshold],
                    outputs=[recon_plot],
                    api_name="update_viz"
                )

            # 语义分割标签页
            with gr.TabItem("语义分割"):
                with gr.Row():
                    with gr.Column(scale=1):
                        acdc_cases = load_acdc_cases()
                        seg_case = gr.Dropdown(
                            choices=acdc_cases,
                            label="选择ACDC案例（心脏MRI）"
                        )
                        seg_btn = gr.Button("执行分割", variant="primary")
                        seg_status = gr.Textbox(label="状态", lines=2)

                        gr.Markdown("### SAM 提示效果展示")
                        gr.Markdown("*基于分割结果自动生成的提示示例*")
                        point_prompt_img = gr.Image(label="点提示 (Point Prompt)", height=150)
                        box_prompt_img = gr.Image(label="框提示 (Box Prompt)", height=150)
                        mask_prompt_img = gr.Image(label="掩码提示 (Mask Prompt)", height=150)

                    with gr.Column(scale=2):
                        comparison_plot = gr.Plot(label="切片浏览（同步联动）")

                seg_btn.click(
                    fn=run_segmentation,
                    inputs=[seg_case],
                    outputs=[comparison_plot, point_prompt_img, box_prompt_img, mask_prompt_img, seg_status],
                    api_name="run_segmentation"
                )

            # 使用说明
            with gr.TabItem("使用说明"):
                gr.Markdown("""
                ## CT 三维重建
                1. 选择预设案例（足部-3/6/9视角）
                2. 点击"加载模型"
                3. 点击"执行重建"
                4. 使用可视化选项浏览结果

                ## 语义分割
                1. 选择ACDC案例（心脏MRI数据）
                2. 点击"执行分割"
                3. 使用同步滑块浏览原始图像和分割结果
                4. 查看SAM提示效果展示：
                   - **点提示**：绿色点标注器官中心，红色X标注背景
                   - **框提示**：彩色矩形框标注器官边界
                   - **掩码提示**：半透明颜色覆盖器官区域
                """)

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("智能辅助诊断平台 (简化版)")
    print("=" * 60)

    app = create_app()
    # 尝试使用7860端口，如果被占用则自动选择其他可用端口
    try:
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except OSError:
        print("端口7860被占用，使用自动分配的端口...")
        app.launch(server_name="0.0.0.0", server_port=None, share=False)
