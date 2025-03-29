import cv2
import numpy as np
import argparse

def apply_filters(input_path, output_dir):
    """
    应用多种滤波算法并保存结果
    :param input_path: 输入图片路径
    :param output_dir: 输出目录路径
    """
    # 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        return
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 灰度化处理（可选）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 均值滤波
    mean_filter = cv2.blur(img, (5, 5))
    cv2.imwrite(f"{output_dir}/mean_filter.jpg", mean_filter)
    
    # 2. 高斯滤波
    gaussian_filter = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/gaussian_filter.jpg", gaussian_filter)
    
    # 3. 中值滤波
    median_filter = cv2.medianBlur(img, 5)
    cv2.imwrite(f"{output_dir}/median_filter.jpg", median_filter)
    
    # 4. 双边滤波（保边滤波）
    bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(f"{output_dir}/bilateral_filter.jpg", bilateral_filter)
    
    print(f"所有滤波结果已保存至 {output_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图像滤波演示')
    parser.add_argument("--input", default="D:/work/opencv-practice/projects/02_image_filter/original.jpg", help="输入图片路径")
    parser.add_argument("--output", default="filter_results", help="输出目录路径")
    args = parser.parse_args()
    
    apply_filters(args.input, args.output)
