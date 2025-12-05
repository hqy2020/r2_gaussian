"""
SAM 提示效果可视化

生成点提示、框提示、掩码提示的示例图
"""
import numpy as np
from typing import Dict, Optional, Tuple
import cv2


def normalize_image(image: np.ndarray) -> np.ndarray:
    """将图像归一化到 0-255 uint8"""
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def get_organ_center(mask: np.ndarray, organ_id: int) -> Optional[Tuple[int, int]]:
    """获取器官的质心位置"""
    organ_mask = (mask == organ_id).astype(np.uint8)
    if organ_mask.sum() == 0:
        return None

    # 计算质心
    coords = np.where(organ_mask > 0)
    cy = int(np.mean(coords[0]))
    cx = int(np.mean(coords[1]))
    return (cx, cy)


def get_organ_bbox(mask: np.ndarray, organ_id: int) -> Optional[Tuple[int, int, int, int]]:
    """获取器官的边界框 (x1, y1, x2, y2)"""
    organ_mask = (mask == organ_id).astype(np.uint8)
    if organ_mask.sum() == 0:
        return None

    coords = np.where(organ_mask > 0)
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    return (x1, y1, x2, y2)


def create_point_prompt_image(
    image: np.ndarray,
    mask: np.ndarray,
    organ_colors: Dict[int, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    创建点提示示例图

    Args:
        image: 灰度图像切片 [H, W]
        mask: 分割掩码 [H, W]，值为器官ID
        organ_colors: 器官ID到RGB颜色的映射

    Returns:
        RGB图像 [H, W, 3]
    """
    if organ_colors is None:
        organ_colors = {
            1: (255, 100, 100),   # RV - 红色
            2: (100, 255, 100),   # MYO - 绿色
            3: (100, 100, 255),   # LV - 蓝色
        }

    # 转换为RGB
    img_norm = normalize_image(image)
    rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # 在每个器官中心画点
    for organ_id, color in organ_colors.items():
        center = get_organ_center(mask, organ_id)
        if center is not None:
            # 画实心圆点（正样本）
            cv2.circle(rgb, center, 8, color, -1)
            # 画白色边框
            cv2.circle(rgb, center, 8, (255, 255, 255), 2)

    # 在背景区域画一个负样本点（红色X）
    bg_mask = (mask == 0).astype(np.uint8)
    if bg_mask.sum() > 0:
        coords = np.where(bg_mask > 0)
        # 选择一个离器官较远的背景点
        idx = len(coords[0]) // 4
        neg_y, neg_x = coords[0][idx], coords[1][idx]
        # 画X标记
        cv2.drawMarker(rgb, (neg_x, neg_y), (255, 50, 50),
                       cv2.MARKER_TILTED_CROSS, 12, 3)

    return rgb


def create_box_prompt_image(
    image: np.ndarray,
    mask: np.ndarray,
    organ_colors: Dict[int, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    创建框提示示例图

    Args:
        image: 灰度图像切片 [H, W]
        mask: 分割掩码 [H, W]
        organ_colors: 器官ID到RGB颜色的映射

    Returns:
        RGB图像 [H, W, 3]
    """
    if organ_colors is None:
        organ_colors = {
            1: (255, 100, 100),   # RV - 红色
            2: (100, 255, 100),   # MYO - 绿色
            3: (100, 100, 255),   # LV - 蓝色
        }

    # 转换为RGB
    img_norm = normalize_image(image)
    rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # 为每个器官画边界框
    for organ_id, color in organ_colors.items():
        bbox = get_organ_bbox(mask, organ_id)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # 稍微扩大边界框
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(rgb.shape[1] - 1, x2 + pad)
            y2 = min(rgb.shape[0] - 1, y2 + pad)
            # 画矩形框
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)

    return rgb


def create_mask_prompt_image(
    image: np.ndarray,
    mask: np.ndarray,
    organ_colors: Dict[int, Tuple[int, int, int]] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    创建掩码提示示例图

    Args:
        image: 灰度图像切片 [H, W]
        mask: 分割掩码 [H, W]
        organ_colors: 器官ID到RGB颜色的映射
        alpha: 掩码透明度

    Returns:
        RGB图像 [H, W, 3]
    """
    if organ_colors is None:
        organ_colors = {
            1: (255, 100, 100),   # RV - 红色
            2: (100, 255, 100),   # MYO - 绿色
            3: (100, 100, 255),   # LV - 蓝色
        }

    # 转换为RGB
    img_norm = normalize_image(image)
    rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB).astype(np.float32)

    # 创建彩色掩码叠加层
    overlay = np.zeros_like(rgb)

    for organ_id, color in organ_colors.items():
        organ_mask = (mask == organ_id)
        if organ_mask.sum() > 0:
            overlay[organ_mask] = color

    # 混合原图和叠加层
    result = rgb * (1 - alpha) + overlay * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def generate_sam_prompt_examples(
    image: np.ndarray,
    mask: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """
    生成 SAM 三种提示方式的示例图

    Args:
        image: 3D图像体积 [D, H, W]
        mask: 3D分割掩码 [D, H, W]
        slice_idx: 切片索引，None表示中间切片
        axis: 切片轴

    Returns:
        {
            "point": 点提示示例图 [H, W, 3],
            "box": 框提示示例图 [H, W, 3],
            "mask": 掩码提示示例图 [H, W, 3]
        }
    """
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2

    # 获取切片
    if axis == 0:
        img_slice = image[slice_idx]
        mask_slice = mask[slice_idx]
    elif axis == 1:
        img_slice = image[:, slice_idx]
        mask_slice = mask[:, slice_idx]
    else:
        img_slice = image[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

    # 生成三种提示图
    return {
        "point": create_point_prompt_image(img_slice, mask_slice),
        "box": create_box_prompt_image(img_slice, mask_slice),
        "mask": create_mask_prompt_image(img_slice, mask_slice)
    }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    image = np.random.rand(10, 128, 128) * 255
    mask = np.zeros((10, 128, 128), dtype=np.int32)
    mask[:, 40:60, 30:50] = 1  # RV
    mask[:, 50:80, 50:80] = 2  # MYO
    mask[:, 55:75, 55:75] = 3  # LV

    examples = generate_sam_prompt_examples(image, mask)

    for name, img in examples.items():
        cv2.imwrite(f"test_sam_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved test_sam_{name}.png, shape: {img.shape}")
