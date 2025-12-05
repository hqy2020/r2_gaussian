"""
预设案例数据管理
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from demo.config import (
    PROJECT_ROOT,
    RECONSTRUCTION_PRESETS,
    SEGMENTATION_PRESETS,
    ACDC_DATA_PATH,
    ReconstructionPreset,
    SegmentationPreset
)


class DataManager:
    """预设数据管理器"""

    def __init__(self):
        self.reconstruction_presets = RECONSTRUCTION_PRESETS
        self.segmentation_presets = SEGMENTATION_PRESETS
        self._validate_presets()

    def _validate_presets(self):
        """验证预设路径是否存在"""
        for preset in self.reconstruction_presets:
            if not os.path.exists(preset.data_path):
                print(f"Warning: Data not found: {preset.data_path}")
            if not os.path.exists(preset.model_path):
                print(f"Warning: Model not found: {preset.model_path}")

        for preset in self.segmentation_presets:
            if not os.path.exists(preset.data_path):
                print(f"Warning: Data not found: {preset.data_path}")

    def get_reconstruction_preset_names(self) -> List[str]:
        """获取所有重建预设名称"""
        return [p.name for p in self.reconstruction_presets]

    def get_segmentation_preset_names(self) -> List[str]:
        """获取所有分割预设名称"""
        return [p.name for p in self.segmentation_presets]

    def get_reconstruction_preset(self, name: str) -> Optional[ReconstructionPreset]:
        """获取重建预设"""
        for preset in self.reconstruction_presets:
            if preset.name == name:
                return preset
        return None

    def get_segmentation_preset(self, name: str) -> Optional[SegmentationPreset]:
        """获取分割预设"""
        for preset in self.segmentation_presets:
            if preset.name == name:
                return preset
        return None

    def list_acdc_cases(self) -> List[str]:
        """列出所有 ACDC 案例"""
        if not os.path.exists(ACDC_DATA_PATH):
            return []
        cases = []
        for f in os.listdir(ACDC_DATA_PATH):
            if f.endswith('.h5'):
                cases.append(f.replace('.h5', ''))
        return sorted(cases)

    def get_acdc_path(self, case_name: str) -> str:
        """获取 ACDC 数据文件路径"""
        return os.path.join(ACDC_DATA_PATH, f"{case_name}.h5")


# 全局实例
data_manager = DataManager()


# 测试代码
if __name__ == "__main__":
    dm = DataManager()

    print("Reconstruction presets:")
    for name in dm.get_reconstruction_preset_names():
        preset = dm.get_reconstruction_preset(name)
        exists_data = os.path.exists(preset.data_path)
        exists_model = os.path.exists(preset.model_path)
        print(f"  - {name}: data={exists_data}, model={exists_model}")

    print("\nSegmentation presets:")
    for name in dm.get_segmentation_preset_names():
        preset = dm.get_segmentation_preset(name)
        exists = os.path.exists(preset.data_path)
        print(f"  - {name}: exists={exists}")

    print("\nACDC cases:")
    cases = dm.list_acdc_cases()
    print(f"  Total: {len(cases)} cases")
    if cases:
        print(f"  First 5: {cases[:5]}")
