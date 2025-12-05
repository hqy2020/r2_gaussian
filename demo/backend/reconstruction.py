"""
重建模块推理接口封装

基于 R²-Gaussian 的稀疏视角CT三维重建
"""
import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.image_utils import metric_vol


@dataclass
class ReconstructionResult:
    """重建结果"""
    volume: np.ndarray          # 重建体积 [nx, ny, nz]
    volume_gt: Optional[np.ndarray] = None  # 真值体积（如果有）
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    projections: Optional[List[np.ndarray]] = None  # 投影图像列表


class ReconstructionInference:
    """R²-Gaussian 重建推理封装类"""

    def __init__(self, model_path: str, data_path: str = None, device: str = "cuda"):
        """
        初始化重建模型

        Args:
            model_path: 训练好的模型目录 (如 output/2025_11_17_foot_3views_baseline_30k)
            data_path: 数据文件路径 (如 data/369/foot_50_3views.pickle)，可选
            device: 推理设备
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = device

        self.gaussians: Optional[GaussianModel] = None
        self.scene: Optional[Scene] = None
        self.pipeline: Optional[PipelineParams] = None
        self.scanner_cfg: Optional[Dict] = None
        self.vol_gt: Optional[torch.Tensor] = None
        self.loaded_iter: int = -1

        self._initialized = False

    def load_model(self, iteration: int = -1) -> bool:
        """
        加载模型检查点

        Args:
            iteration: 加载的迭代次数，-1 表示最新

        Returns:
            是否加载成功
        """
        try:
            # 确定数据路径
            if self.data_path is None:
                # 从模型目录读取配置
                cfg_path = os.path.join(self.model_path, "cfg_args")
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        import argparse
                        cfg_text = f.read()
                        # 解析配置
                        # 简单起见，使用 Scene 的检测逻辑
                        pass

            # 创建 ModelParams
            class Args:
                def __init__(self, model_path, source_path):
                    self.model_path = model_path
                    self.source_path = source_path
                    self.eval = True
                    self.ply_path = ""
                    self.scale_bound = None
                    self.images = "images"
                    self.resolution = 1
                    self.data_device = "cuda"
                    self.white_background = False
                    self.sh_degree = 3

            # 读取模型配置
            cfg_args_path = os.path.join(self.model_path, "cfg_args")
            if os.path.exists(cfg_args_path):
                with open(cfg_args_path, 'r') as f:
                    cfg_content = f.read()
                    # 解析 Namespace 格式
                    import re
                    source_match = re.search(r"source_path='([^']+)'", cfg_content)
                    if source_match:
                        source_path = source_match.group(1)
                    else:
                        source_path = self.data_path
            else:
                source_path = self.data_path

            if source_path is None:
                raise ValueError("需要提供数据路径 data_path 或确保模型目录包含 cfg_args")

            args = Args(self.model_path, source_path)

            # 加载场景（获取 scanner_cfg 和 vol_gt）
            self.scene = Scene(args, shuffle=False)
            self.scanner_cfg = self.scene.scanner_cfg
            self.vol_gt = self.scene.vol_gt

            # 初始化 GaussianModel
            self.gaussians = GaussianModel(None)
            self.loaded_iter = initialize_gaussian(self.gaussians, args, iteration)

            # Pipeline 参数
            class PipeArgs:
                convert_SHs_python = False
                compute_cov3D_python = False
                debug = False

            self.pipeline = PipeArgs()

            self._initialized = True
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reconstruct(self) -> ReconstructionResult:
        """
        执行3D体积重建

        Returns:
            ReconstructionResult 包含重建结果
        """
        if not self._initialized:
            raise RuntimeError("请先调用 load_model() 加载模型")

        with torch.no_grad():
            # 执行重建
            query_pkg = query(
                self.gaussians,
                self.scanner_cfg["offOrigin"],
                self.scanner_cfg["nVoxel"],
                self.scanner_cfg["sVoxel"],
                self.pipeline,
            )
            vol_pred = query_pkg["vol"]

            # 转换为 numpy
            vol_pred_np = t2a(vol_pred)

            # 计算指标（如果有真值）
            psnr = None
            ssim = None
            vol_gt_np = None

            if self.vol_gt is not None:
                vol_gt_np = t2a(self.vol_gt)
                psnr, _ = metric_vol(self.vol_gt, vol_pred, "psnr")
                ssim, _ = metric_vol(self.vol_gt, vol_pred, "ssim")

            return ReconstructionResult(
                volume=vol_pred_np,
                volume_gt=vol_gt_np,
                psnr=psnr,
                ssim=ssim
            )

    def render_projections(self, views: str = "train") -> List[np.ndarray]:
        """
        渲染投影图像

        Args:
            views: "train" 或 "test"

        Returns:
            投影图像列表
        """
        if not self._initialized:
            raise RuntimeError("请先调用 load_model() 加载模型")

        cameras = self.scene.getTrainCameras() if views == "train" else self.scene.getTestCameras()

        projections = []
        with torch.no_grad():
            for camera in cameras:
                render_pkg = render(camera, self.gaussians, self.pipeline)
                proj = render_pkg["render"].cpu().numpy()[0]  # [H, W]
                projections.append(proj)

        return projections

    def get_sample_info(self) -> Dict:
        """获取当前样本信息"""
        if not self._initialized:
            return {}

        return {
            "model_path": self.model_path,
            "loaded_iter": self.loaded_iter,
            "n_gaussians": self.gaussians.get_xyz.shape[0] if self.gaussians else 0,
            "volume_shape": list(self.scanner_cfg["nVoxel"]) if self.scanner_cfg else None,
            "n_train_views": len(self.scene.getTrainCameras()) if self.scene else 0,
            "n_test_views": len(self.scene.getTestCameras()) if self.scene else 0,
        }


def load_preset(preset_name: str) -> Tuple[str, str]:
    """
    根据预设名称获取数据和模型路径

    Args:
        preset_name: 预设名称

    Returns:
        (data_path, model_path)
    """
    from demo.config import RECONSTRUCTION_PRESETS

    for preset in RECONSTRUCTION_PRESETS:
        if preset.name == preset_name:
            return preset.data_path, preset.model_path

    raise ValueError(f"未找到预设: {preset_name}")


# 测试代码
if __name__ == "__main__":
    # 测试重建推理
    model_path = str(PROJECT_ROOT / "output/2025_11_17_foot_3views_baseline_30k")
    data_path = str(PROJECT_ROOT / "data/369/foot_50_3views.pickle")

    print(f"Testing with model: {model_path}")
    print(f"Data: {data_path}")

    inference = ReconstructionInference(model_path, data_path)

    if inference.load_model():
        print("Model loaded successfully!")
        print(f"Sample info: {inference.get_sample_info()}")

        result = inference.reconstruct()
        print(f"Reconstruction complete!")
        print(f"Volume shape: {result.volume.shape}")
        print(f"PSNR: {result.psnr:.4f}" if result.psnr else "PSNR: N/A")
        print(f"SSIM: {result.ssim:.4f}" if result.ssim else "SSIM: N/A")
    else:
        print("Failed to load model")
