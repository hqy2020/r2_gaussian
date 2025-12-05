"""
智能辅助诊断平台 - Gradio 主应用

面向稀疏数据的医学影像场景理解和重建方法研究
"""
import os
import sys
from pathlib import Path

# 确保项目路径在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from demo.ui.reconstruction_tab import create_reconstruction_tab
from demo.ui.segmentation_tab import create_segmentation_tab
from demo.config import GRADIO_THEME, SERVER_PORT


# 自定义 CSS
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.main-title {
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 20px;
}

.tab-nav {
    margin-bottom: 20px;
}

footer {
    text-align: center;
    padding: 20px;
    color: #888;
}
"""


def create_app():
    """创建 Gradio 应用"""

    with gr.Blocks(
        title="智能辅助诊断平台",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as app:

        # 标题区域
        gr.Markdown(
            """
            # 智能辅助诊断平台
            ### 面向稀疏数据的医学影像场景理解和重建方法研究

            本平台集成了两个核心模块：
            - **CT 三维重建**：基于 3D Gaussian Splatting 的稀疏视角CT重建 (R²-Gaussian)
            - **语义分割**：基于 UNet + SAM 的半监督医学图像分割
            """,
            elem_classes=["main-title"]
        )

        # 主标签页
        with gr.Tabs(elem_classes=["tab-nav"]):
            with gr.TabItem("🔄 CT 三维重建"):
                gr.Markdown("""
                > 从稀疏的X-ray投影重建完整的3D CT体积
                > - 支持 3/6/9 视角输入
                > - 基于 3D Gaussian Splatting 技术
                """)
                create_reconstruction_tab()

            with gr.TabItem("🎯 语义分割"):
                gr.Markdown("""
                > 自动分割医学图像中的器官结构
                > - 支持心脏MRI分割 (ACDC数据集)
                > - 分割类别：右心室(RV)、心肌(MYO)、左心室(LV)
                """)
                create_segmentation_tab()

            with gr.TabItem("📖 使用说明"):
                gr.Markdown("""
                ## 使用说明

                ### CT 三维重建模块

                1. **选择预设案例**：从下拉菜单选择预训练好的模型（如"足部-3视角"）
                2. **加载模型**：点击"加载模型"按钮，等待模型初始化完成
                3. **执行重建**：点击"执行重建"按钮，系统将从稀疏投影重建3D体积
                4. **浏览结果**：
                   - 切片浏览：使用滑块浏览各个切片
                   - 体绘制：3D等值面渲染
                   - 表面重建：提取表面网格
                   - GT对比：与真值对比（如果有）
                5. **导出结果**：支持导出NPY或NIfTI格式

                ### 语义分割模块

                1. **选择案例**：从ACDC数据集选择一个患者案例
                2. **执行分割**：点击"执行分割"按钮
                3. **查看结果**：
                   - 切片浏览：对比原始图像和分割结果
                   - 3D可视化：查看多器官3D表面
                4. **查看指标**：Dice分数显示各器官分割精度
                5. **导出结果**：支持导出NPY或NIfTI格式

                ---

                ## 技术说明

                ### CT 三维重建 (R²-Gaussian)

                - **输入**：3/6/9个视角的X-ray投影图像
                - **技术**：3D Gaussian Splatting + 可微体素化
                - **输出**：256×256×256 的3D CT体积
                - **指标**：PSNR (峰值信噪比)、SSIM (结构相似性)

                ### 语义分割 (UNet + SAM)

                - **输入**：2D医学图像切片
                - **技术**：UNet编码解码器 + SAM辅助
                - **输出**：像素级分割掩码
                - **指标**：Dice系数

                ---

                ## 关于

                本系统是毕业论文《面向稀疏数据的医学影像场景理解和重建方法研究》的演示平台。

                - **CT重建代码**：基于 R²-Gaussian (NeurIPS 2024)
                - **分割代码**：基于半监督分割方法
                """)

        # 页脚
        gr.Markdown(
            """
            ---
            <center>
            智能辅助诊断平台 v0.1.0 | 面向稀疏数据的医学影像场景理解和重建方法研究
            </center>
            """,
            elem_classes=["footer"]
        )

    return app


def main():
    """主函数"""
    print("=" * 60)
    print("智能辅助诊断平台")
    print("面向稀疏数据的医学影像场景理解和重建方法研究")
    print("=" * 60)

    app = create_app()

    print(f"\n启动服务器...")
    print(f"访问地址: http://localhost:{SERVER_PORT}")
    print(f"局域网地址: http://0.0.0.0:{SERVER_PORT}")
    print("\n按 Ctrl+C 停止服务器")

    app.launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
