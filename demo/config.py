"""
配置管理
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 分割模型路径
SEGMENTATION_MODEL_PATH = "/home/qyhu/Documents/new_PG_semi/exp/acdc/CP_SAM/unet/7/ema_0/best.pth"
SEGMENTATION_CODE_PATH = "/home/qyhu/Documents/new_PG_semi"

# ACDC数据路径
ACDC_DATA_PATH = "/home/qyhu/ACDC/data"


@dataclass
class ReconstructionPreset:
    """重建预设案例"""
    name: str
    organ: str
    views: int
    data_path: str
    model_path: str
    description: str = ""

    def __post_init__(self):
        # 转换为绝对路径
        if not os.path.isabs(self.data_path):
            self.data_path = str(PROJECT_ROOT / self.data_path)
        if not os.path.isabs(self.model_path):
            self.model_path = str(PROJECT_ROOT / self.model_path)


@dataclass
class SegmentationPreset:
    """分割预设案例"""
    name: str
    data_path: str
    description: str = ""


# 预设案例配置
RECONSTRUCTION_PRESETS: List[ReconstructionPreset] = [
    ReconstructionPreset(
        name="足部-3视角",
        organ="foot",
        views=3,
        data_path="data/369/foot_50_3views.pickle",
        model_path="output/2025_11_17_foot_3views_baseline_30k",
        description="足部CT稀疏重建，使用3个视角的X-ray投影"
    ),
    ReconstructionPreset(
        name="足部-6视角",
        organ="foot",
        views=6,
        data_path="data/369/foot_50_6views.pickle",
        model_path="output/2025_11_20_foot_6views_baseline",
        description="足部CT稀疏重建，使用6个视角的X-ray投影"
    ),
    ReconstructionPreset(
        name="足部-9视角",
        organ="foot",
        views=9,
        data_path="data/369/foot_50_9views.pickle",
        model_path="output/2025_11_20_foot_9views_baseline",
        description="足部CT稀疏重建，使用9个视角的X-ray投影"
    ),
]

SEGMENTATION_PRESETS: List[SegmentationPreset] = [
    SegmentationPreset(
        name="心脏ACDC-患者001",
        data_path=f"{ACDC_DATA_PATH}/patient001_frame01.h5",
        description="心脏MRI分割，包含左心室(LV)、右心室(RV)、心肌(MYO)"
    ),
    SegmentationPreset(
        name="心脏ACDC-患者002",
        data_path=f"{ACDC_DATA_PATH}/patient002_frame01.h5",
        description="心脏MRI分割，包含左心室(LV)、右心室(RV)、心肌(MYO)"
    ),
    SegmentationPreset(
        name="心脏ACDC-患者003",
        data_path=f"{ACDC_DATA_PATH}/patient003_frame01.h5",
        description="心脏MRI分割，包含左心室(LV)、右心室(RV)、心肌(MYO)"
    ),
]


# Gradio主题配置
GRADIO_THEME = "soft"
SERVER_PORT = 7860
