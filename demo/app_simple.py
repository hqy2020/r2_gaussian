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
from demo.visualization.slice_viewer import create_slice_viewer
from demo.visualization.volume_render import create_volume_render
from demo.visualization.surface_render import create_segmentation_surface, create_ct_surface


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
        return None, None, None, "请选择案例", 0, 0, 0

    try:
        if seg_model is None:
            seg_model = SegmentationInference()
            seg_model.load_model()

        data_path = get_acdc_data_path(case_name)
        seg_result = seg_model.segment_from_h5(data_path)

        # 可视化
        original_fig = create_slice_viewer(seg_result.image, axis="z", title="原始图像")
        seg_fig = create_slice_viewer(seg_result.prediction.astype(float), axis="z", colorscale="Viridis", title="分割结果")
        surface_fig = create_segmentation_surface(seg_result.prediction, title="3D表面")

        # Dice
        dice_rv = seg_result.dice_scores.get(1, 0) if seg_result.dice_scores else 0
        dice_myo = seg_result.dice_scores.get(2, 0) if seg_result.dice_scores else 0
        dice_lv = seg_result.dice_scores.get(3, 0) if seg_result.dice_scores else 0

        return original_fig, seg_fig, surface_fig, "分割完成!", round(dice_rv, 4), round(dice_myo, 4), round(dice_lv, 4)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"错误: {str(e)}", 0, 0, 0


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

                load_btn.click(load_reconstruction_model, [recon_preset], [recon_status])
                recon_btn.click(run_reconstruction, [], [recon_plot, recon_status, psnr, ssim])
                update_btn.click(update_recon_viz, [viz_type, threshold], [recon_plot])

            # 语义分割标签页
            with gr.TabItem("语义分割"):
                with gr.Row():
                    with gr.Column(scale=1):
                        acdc_cases = load_acdc_cases()
                        seg_case = gr.Dropdown(
                            choices=acdc_cases[:20] if len(acdc_cases) > 20 else acdc_cases,
                            label="选择ACDC案例"
                        )
                        seg_btn = gr.Button("执行分割", variant="primary")
                        seg_status = gr.Textbox(label="状态", lines=2)

                        gr.Markdown("### Dice分数")
                        dice_rv = gr.Number(label="RV (右心室)")
                        dice_myo = gr.Number(label="MYO (心肌)")
                        dice_lv = gr.Number(label="LV (左心室)")

                    with gr.Column(scale=2):
                        with gr.Row():
                            original_plot = gr.Plot(label="原始图像")
                            seg_plot = gr.Plot(label="分割结果")
                        surface_plot = gr.Plot(label="3D表面")

                seg_btn.click(
                    run_segmentation,
                    [seg_case],
                    [original_plot, seg_plot, surface_plot, seg_status, dice_rv, dice_myo, dice_lv]
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
                1. 选择ACDC案例
                2. 点击"执行分割"
                3. 查看分割结果和3D表面
                """)

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("智能辅助诊断平台 (简化版)")
    print("=" * 60)

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
