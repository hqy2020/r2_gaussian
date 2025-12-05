"""
分割模块推理接口封装

基于 UNet + SAM 的半监督医学图像语义分割
"""
import os
import sys
import numpy as np
import torch
import h5py
from pathlib import Path
from scipy.ndimage import zoom
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# 配置
SEGMENTATION_CODE_PATH = "/home/qyhu/Documents/new_PG_semi"
DEFAULT_MODEL_PATH = "/home/qyhu/Documents/new_PG_semi/exp/acdc/CP_SAM/unet/7/ema_0/best.pth"

# 器官名称
ORGAN_NAMES = {
    0: "背景 (Background)",
    1: "右心室 (RV)",
    2: "心肌 (Myocardium)",
    3: "左心室 (LV)"
}

ORGAN_COLORS = {
    0: (0, 0, 0),       # 背景 - 黑色
    1: (255, 0, 0),     # RV - 红色
    2: (0, 255, 0),     # MYO - 绿色
    3: (0, 0, 255)      # LV - 蓝色
}


@dataclass
class SegmentationResult:
    """分割结果"""
    prediction: np.ndarray      # 分割预测 [D, H, W]，值为类别ID
    image: np.ndarray           # 原始图像 [D, H, W]
    label: Optional[np.ndarray] = None  # 真值标签（如果有）
    dice_scores: Optional[Dict[int, float]] = None  # 各类别 Dice 分数


class SegmentationInference:
    """UNet 分割推理封装类"""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        初始化分割模型

        Args:
            model_path: 模型权重路径
            device: 推理设备
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device
        self.input_size = (256, 256)
        self.num_classes = 4  # 背景 + 3个器官

        self.model = None
        self._initialized = False

    def load_model(self) -> bool:
        """
        加载分割模型

        Returns:
            是否加载成功
        """
        try:
            # 添加分割代码路径
            if SEGMENTATION_CODE_PATH not in sys.path:
                sys.path.insert(0, SEGMENTATION_CODE_PATH)

            from model.unet import UNet

            # 创建模型
            self.model = UNet(in_chns=1, class_num=self.num_classes).to(self.device)

            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint['model']
            # 处理 DataParallel 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()

            print(f"分割模型加载成功: {self.model_path}")
            self._initialized = True
            return True

        except Exception as e:
            print(f"加载分割模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def segment_slice(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割单张切片

        Args:
            image: 输入图像 [H, W]

        Returns:
            (prediction, confidence)
            prediction: 分割结果 [H, W]，值为类别ID (0-3)
            confidence: 置信度 [H, W]
        """
        if not self._initialized:
            raise RuntimeError("请先调用 load_model() 加载模型")

        h, w = image.shape

        # 缩放到模型输入尺寸
        image_resized = zoom(image, (256 / h, 256 / w), order=0)

        # 归一化（如果需要）
        if image_resized.max() > 1.0:
            image_resized = image_resized / 255.0

        # 转换为张量
        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            pred = pred.squeeze(0).cpu().numpy()
            confidence = confidence.squeeze(0).cpu().numpy()

        # 缩放回原尺寸
        pred_resized = zoom(pred, (h / 256, w / 256), order=0)
        conf_resized = zoom(confidence, (h / 256, w / 256), order=0)

        return pred_resized.astype(np.int32), conf_resized.astype(np.float32)

    def segment_volume(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割整个3D体积 (逐切片)

        Args:
            volume: 输入体积 [D, H, W]

        Returns:
            (prediction, confidence)
            prediction: 分割结果 [D, H, W]
            confidence: 置信度 [D, H, W]
        """
        prediction = np.zeros_like(volume, dtype=np.int32)
        confidence = np.zeros_like(volume, dtype=np.float32)

        for i in range(volume.shape[0]):
            pred, conf = self.segment_slice(volume[i])
            prediction[i] = pred
            confidence[i] = conf

        return prediction, confidence

    def segment_from_h5(self, h5_path: str) -> SegmentationResult:
        """
        从 HDF5 文件加载数据并分割

        Args:
            h5_path: HDF5 文件路径

        Returns:
            SegmentationResult
        """
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]
            label = f['label'][:] if 'label' in f else None

        prediction, _ = self.segment_volume(image)

        # 计算 Dice 分数（如果有真值）
        dice_scores = None
        if label is not None:
            dice_scores = {}
            for cls in [1, 2, 3]:
                pred_mask = (prediction == cls)
                gt_mask = (label == cls)
                intersection = (pred_mask & gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum()
                dice_scores[cls] = 2 * intersection / (union + 1e-8)

        return SegmentationResult(
            prediction=prediction,
            image=image,
            label=label,
            dice_scores=dice_scores
        )

    def create_colored_mask(self, prediction: np.ndarray, slice_idx: int = None) -> np.ndarray:
        """
        创建彩色分割掩码

        Args:
            prediction: 分割预测 [D, H, W] 或 [H, W]
            slice_idx: 如果是3D，指定切片索引

        Returns:
            RGB 图像 [H, W, 3]
        """
        if prediction.ndim == 3:
            if slice_idx is None:
                slice_idx = prediction.shape[0] // 2
            pred_slice = prediction[slice_idx]
        else:
            pred_slice = prediction

        h, w = pred_slice.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for cls, color in ORGAN_COLORS.items():
            mask = pred_slice == cls
            colored[mask] = color

        return colored


def load_acdc_cases() -> List[str]:
    """获取 ACDC 数据集中的所有案例"""
    acdc_path = "/home/qyhu/ACDC/data"
    cases = []
    if os.path.exists(acdc_path):
        for f in os.listdir(acdc_path):
            if f.endswith('.h5'):
                cases.append(f.replace('.h5', ''))
    return sorted(cases)


def get_acdc_data_path(case_name: str) -> str:
    """获取 ACDC 数据文件路径"""
    return f"/home/qyhu/ACDC/data/{case_name}.h5"


# 测试代码
if __name__ == "__main__":
    print("Testing segmentation inference...")

    inference = SegmentationInference()

    if inference.load_model():
        print("Model loaded successfully!")

        # 测试分割
        test_h5 = "/home/qyhu/ACDC/data/patient001_frame01.h5"
        if os.path.exists(test_h5):
            result = inference.segment_from_h5(test_h5)
            print(f"Image shape: {result.image.shape}")
            print(f"Prediction shape: {result.prediction.shape}")
            print(f"Unique classes: {np.unique(result.prediction)}")

            if result.dice_scores:
                print("Dice scores:")
                for cls, score in result.dice_scores.items():
                    print(f"  {ORGAN_NAMES[cls]}: {score:.4f}")
        else:
            print(f"Test file not found: {test_h5}")
    else:
        print("Failed to load model")
