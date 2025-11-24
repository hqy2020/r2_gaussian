#!/usr/bin/env python3
"""
实验：尝试在 X 射线图像上提取 SIFT 特征点
目的：证明传统特征检测在 X 射线图像上失效
"""
import numpy as np
import cv2
import pickle
import sys
import os.path as osp

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.general_utils import t2a


def test_sift_on_xray(data_path):
    """在 X 射线图像上测试 SIFT 特征检测"""

    print("=" * 60)
    print("实验：X 射线图像的特征点检测能力测试")
    print("=" * 60)

    # 加载 X 射线投影数据
    import argparse
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args([])
    model_params = model_params.extract(args)
    model_params.source_path = data_path
    scene = Scene(model_params, shuffle=False)

    # 获取第一张投影图像
    camera = scene.train_cameras[0]
    xray_image = t2a(camera.original_image)[0]  # (H, W)

    print(f"\n📊 图像信息:")
    print(f"  分辨率: {xray_image.shape}")
    print(f"  值范围: [{xray_image.min():.4f}, {xray_image.max():.4f}]")
    print(f"  均值: {xray_image.mean():.4f}")
    print(f"  标准差: {xray_image.std():.4f}")

    # 归一化到 [0, 255] uint8
    xray_norm = ((xray_image - xray_image.min()) /
                 (xray_image.max() - xray_image.min()) * 255).astype(np.uint8)

    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create()

    # 检测特征点
    keypoints, descriptors = sift.detectAndCompute(xray_norm, None)

    print(f"\n🔍 SIFT 特征检测结果:")
    print(f"  检测到的特征点数量: {len(keypoints)}")
    print(f"  图像像素总数: {xray_image.size}")
    print(f"  特征点密度: {len(keypoints) / xray_image.size * 1000:.2f} per 1000 pixels")

    # 分析特征点分布
    if len(keypoints) > 0:
        responses = np.array([kp.response for kp in keypoints])
        sizes = np.array([kp.size for kp in keypoints])

        print(f"\n📈 特征点质量分析:")
        print(f"  响应强度均值: {responses.mean():.4f}")
        print(f"  响应强度中位数: {np.median(responses):.4f}")
        print(f"  响应强度最大值: {responses.max():.4f}")
        print(f"  特征尺度均值: {sizes.mean():.2f} pixels")

        # 绘制特征点（保存图片）
        img_with_keypoints = cv2.drawKeypoints(
            xray_norm, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        output_path = "xray_sift_features.png"
        cv2.imwrite(output_path, img_with_keypoints)
        print(f"\n✅ 特征点可视化已保存: {output_path}")

    # 对比：模拟一个 RGB 图像的特征点密度
    print(f"\n📊 对比参考:")
    print(f"  典型 RGB 图像特征点密度: 5-20 per 1000 pixels")
    print(f"  COLMAP 推荐最小密度: 3 per 1000 pixels")
    print(f"  当前 X 射线图像密度: {len(keypoints) / xray_image.size * 1000:.2f} per 1000 pixels")

    # 判断
    if len(keypoints) < xray_image.size * 0.003:  # < 3 per 1000
        print(f"\n❌ 结论: 特征点过少，COLMAP 将失败！")
        print(f"  原因: X 射线图像缺乏纹理和明显边缘")
    else:
        print(f"\n⚠️  结论: 特征点勉强够数，但质量可能很差")

    print("\n" + "=" * 60)

    return len(keypoints), xray_image.size


if __name__ == "__main__":
    # 测试 Foot-3 views 数据
    data_path = "data/369/foot_50_3views.pickle"

    if not osp.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    test_sift_on_xray(data_path)
