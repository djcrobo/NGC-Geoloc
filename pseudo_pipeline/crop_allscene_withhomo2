import os
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None
def get_image_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, width, height):
    """
    将经纬度转换为像素坐标。

    参数:
    - lat (float): 纬度
    - lon (float): 经度
    - min_lat (float): 图像边界最小纬度
    - max_lat (float): 图像边界最大纬度
    - min_lon (float): 图像边界最小经度
    - max_lon (float): 图像边界最大经度
    - width (int): 图像宽度（像素）
    - height (int): 图像高度（像素）

    返回:
    - (int, int): 对应的像素坐标 (x, y)
    """
    # 检查经纬度是否在边界内
    if not (min_lat <= lat <= max_lat) or not (min_lon <= lon <= max_lon):
        raise ValueError(f"经纬度 ({lat}, {lon}) 超出图像边界。")

    # 线性比例转换
    x = ((lon - min_lon) / (max_lon - min_lon)) * (width - 1)
    y = ((max_lat - lat) / (max_lat - min_lat)) * (height - 1)  # 北向上
    return int(x), int(y)

def visualize_points(dom_path, gps_points):
    """
    可视化多个 GPS 点在 DOM 图像上的位置。

    参数:
    - dom_path (str): GeoTIFF 图像的路径
    - gps_points (list of dict): 每个字典包含 'lat', 'lon', 'pixel_x', 'pixel_y'
    """
    # 加载 DOM 图像
    with rasterio.open(dom_path) as dom:
        dom_image = dom.read([1, 2, 3])  # 读取 RGB 通道
        dom_height, dom_width = dom_image.shape[1], dom_image.shape[2]
        dom_image = dom_image.transpose(1, 2, 0)  # 转换为 (H, W, 3)

    plt.figure(figsize=(20, 20))  # 增大图像显示尺寸
    plt.imshow(dom_image)

    # 绘制所有 GPS 点
    for idx, point in enumerate(gps_points):
        x, y = point['pixel_x'], point['pixel_y']
        plt.scatter(x, y, 
                    c='red', 
                    marker='o', 
                    edgecolors='black',  # 添加黑色边框
                    linewidth=0.5,       # 边框宽度
                    label='GPS Point' if idx == 0 else "", 
                    s=30,                # 标记大小
                    alpha=0.7)            # 设置透明度以便于叠加
        # 添加标注（去除背景）
        plt.text(x + 5, y + 5, f"P{point['num']}", 
                 color='yellow', 
                 fontsize=9, 
                 bbox=dict(facecolor='none', edgecolor='none'))  # 去除背景

    plt.title('GPS Points Visualization on DOM', fontsize=20)
    plt.legend()
    plt.xlabel("X (Pixel)", fontsize=16)
    plt.ylabel("Y (Pixel)", fontsize=16)
    plt.axis("equal")
    plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
    plt.show()

def read_gps_from_csv(csv_path):
    """
    从 CSV 文件读取 GPS 点，并标准化列名为小写。

    参数:
    - csv_path (str): CSV 文件的路径

    返回:
    - list of dict: 每个字典包含 'num', 'lat', 'lon', 'phi1' 等字段
    """
    df = pd.read_csv(csv_path)
    # 将所有列名转换为小写并去除空格
    df.columns = [col.strip().lower() for col in df.columns]
    print("标准化后的列名：", df.columns.tolist())
    
    # 打印前5条数据以确认
    print("前5条 GPS 点数据：")
    print(df.head())
    
    gps_points = df.to_dict(orient='records')
    return gps_points

def crop_and_rotate_region(dom_image_path, pixel_x, pixel_y, crop_width, crop_height, output_dir, angle, image_width, image_height, num, homographies_dir):
    """
    根据像素坐标裁剪并旋转图像，并保存为PNG格式，同时计算并保存单应性矩阵。

    参数:
    - dom_image_path (str): 输入图像路径（支持如PNG、JPEG等格式）
    - pixel_x (int): 裁剪中心点的 x 坐标（像素）
    - pixel_y (int): 裁剪中心点的 y 坐标（像素）
    - crop_width (int): 裁剪宽度（像素）
    - crop_height (int): 裁剪高度（像素）
    - output_dir (str): 裁剪后图像保存目录（文件夹路径）
    - angle (float): 逆时针旋转角度（度）
    - image_width (int): 原始图像宽度
    - image_height (int): 原始图像高度
    - num (int): CSV 中的行编号，用于命名输出文件
    - homographies_dir (str): 单应性矩阵保存目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(homographies_dir, exist_ok=True)  # 创建单应性矩阵保存目录

    try:
        # 打开图像
        large_image = Image.open(dom_image_path)
        image_array = np.array(large_image)
    except Exception as e:
        print(f"无法打开图像 {dom_image_path}: {e}")
        return

    cx = pixel_x 
    cy = pixel_y
    x1 = cx - crop_width // 2
    y1 = cy - crop_height // 2
    x2 = cx + crop_width // 2
    y2 = cy + crop_height // 2

    # 计算裁剪框四个角的坐标
    ori_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    ones_ori = np.ones((ori_points.shape[0], 1))
    ori_points_homogeneous = np.hstack([ori_points, ones_ori])
    
    # 获取旋转矩阵 (注意角度取负以匹配坐标系)
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1)  # 2x3 矩阵
    H_rotate0=cv2.getRotationMatrix2D((crop_width // 2, crop_height // 2), -angle, 1)#这里应该改成正的angle  吗？
    # 将旋转矩阵扩展为 3x3 单应性矩阵
    H_rotate = np.vstack([H_rotate0, [0, 0, 1]])  # 3x3 矩阵
    
    # 创建平移矩阵，将裁剪区域的左上角映射回原始图像
    H_translate = np.array([
        [1, 0, x1],
        [0, 1, y1],
        [0, 0, 1]
    ])
    
    # 组合变换得到最终的单应性矩阵
    H = H_translate @ H_rotate  # 矩阵乘法
    # import pdb;pdb.set_trace()
    #这里的H的旋转中心点和平移距离有错误！需要修改
    
    # 保存单应性矩阵
    homography_filename = f"homography_crop_{num}.npy"
    homography_path = os.path.join(homographies_dir, homography_filename)
    np.save(homography_path, H)
    
    # 通过矩阵乘法计算新的点
    ori_transformed_points = ori_points_homogeneous.dot(M.T)
    transformed_points = ori_transformed_points[:, :2]

    # 定义检查点是否在图像边界内的函数
    def is_within_image_bounds(points, img_width, img_height):
        for point in points:
            x, y = point
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                return False  # 有点超出边界
        return True  # 所有点都在边界内

    # 检查旋转后角点是否在裁剪图像范围内
    if not is_within_image_bounds(transformed_points, image_width, image_height):
        print(f"Rotated crop for pixel ({pixel_x}, {pixel_y}) with angle {angle}° exceeds crop bounds. Skipping.")
        return

    y_coords, x_coords = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')

    grid_points = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
    ones = np.ones((grid_points.shape[0], 1))
    grid_points_homogeneous = np.hstack([grid_points, ones])

    # 变换坐标点
    transformed_grid_points = grid_points_homogeneous.dot(M.T)

    transformed_x = transformed_grid_points[:, 0].reshape(y_coords.shape)
    transformed_y = transformed_grid_points[:, 1].reshape(x_coords.shape)

    # 合并为 (2, height, width) 的矩阵
    transformed_coords = np.stack([transformed_x, transformed_y], axis=0)

    # 重映射图像
    remapped_image = cv2.remap(
        image_array,
        transformed_x.astype(np.float32),
        transformed_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # 使用黑色填充边界
    )

    remapped_image_pil = Image.fromarray(remapped_image)
    # 检查四个角点是否在图像范围内

    # 如果有角点超出范围，则不生成图像
    if not is_within_image_bounds(transformed_points, image_width, image_height):
        print("有角点超出图像范围，跳过生成图像2")
    else:
        remapped_image_pil.save(
            os.path.join(output_dir, f"crop_{num}.png")
        )
        print(f"裁剪并旋转后的图像保存为 {os.path.join(output_dir, f'crop_{num}.png')}")
        print(f"单应性矩阵保存为 {homography_path}")

def main():
    # 替换为你的 GeoTIFF 文件路径
    # dom_path = "/home/ubuntu/Downloads/UAV_VisLoc_dataset/06/satellite06.tif"

    # # 替换为你的 CSV 文件路径
    # csv_path = "/home/ubuntu/Downloads/UAV_VisLoc_dataset/06/06.csv"

    # # 替换为你希望保存裁剪图像的文件夹路径
    # output_folder = "/home/ubuntu/Desktop/crop_pic3"

    # 替换为你希望保存单应性矩阵的文件夹路径
    # homographies_dir = "/home/ubuntu/Desktop/homographies"
    dom_path = "/mnt/data1/PublicDatasets/UAV_VisLoc_dataset/05/satellite05.tif"
    

    # 替换为你的 CSV 文件路径
    csv_path = "/mnt/data1/PublicDatasets/UAV_VisLoc_dataset/05/05.csv"

    # 替换为你希望保存裁剪图像的文件夹路径
    output_folder = "/mnt/data1/UserData/dengjc23/map_coordinate_regression/datasets/05/db/ref_image"

    # 替换为你希望保存单应性矩阵的文件夹路径
    homographies_dir = "/mnt/data1/UserData/dengjc23/map_coordinate_regression/datasets/05/query/aerial/homographies05"

    # 自定义裁剪宽度和高度（像素）
    # crop_width = 2100
    # crop_height = 1400
    crop_width = 1500
    crop_height = 1000
    

    # 检查文件是否存在
    if not os.path.isfile(dom_path):
        print(f"GeoTIFF 文件未找到: {dom_path}")
        return
    if not os.path.isfile(csv_path):
        print(f"CSV 文件未找到: {csv_path}")
        return

    # 读取 GPS 点
    gps_points = read_gps_from_csv(csv_path)
    print(f"从 CSV 读取了 {len(gps_points)} 个 GPS 点。")

    try:
        # 获取 GeoTIFF 的边界和尺寸
        with rasterio.open(dom_path) as dom:
            min_lon, min_lat, max_lon, max_lat = dom.bounds.left, dom.bounds.bottom, dom.bounds.right, dom.bounds.top
            width = dom.width
            height = dom.height

            print(f"GeoTIFF 边界:")
            print(f"最小经度: {min_lon}, 最大经度: {max_lon}")
            print(f"最小纬度: {min_lat}, 最大纬度: {max_lat}")
            print(f"图像尺寸: 宽度 = {width} 像素, 高度 = {height} 像素")

    except Exception as e:
        print(f"处理 GeoTIFF 文件时出错: {e}")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(homographies_dir, exist_ok=True)  # 创建单应性矩阵保存目录

    # 将每个 GPS 点转换为像素坐标并裁剪图像
    valid_gps_points = []
    for idx, point in enumerate(gps_points):
        try:
            lat = point['lat']
            lon = point['lon']
            phi1 = point['phi1']  # 读取 'phi1' 作为旋转角度
            x, y = get_image_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, width, height)
            point['pixel_x'] = x
            point['pixel_y'] = y
            valid_gps_points.append(point)
            print(f"GPS 点 {point['num']} (Lat: {lat}, Lon: {lon}) 转换为像素坐标 (X: {x}, Y: {y}), 旋转角度: {phi1}°")

            # 定义裁剪后的图像保存路径
            output_path = os.path.join(output_folder, f"crop_{point['num']}.png")

            # 裁剪并旋转图像，同时计算并保存单应性矩阵
            crop_and_rotate_region(
                dom_image_path=dom_path,
                pixel_x=x,
                pixel_y=y,
                crop_width=crop_width,
                crop_height=crop_height,
                output_dir=output_folder,
                angle=phi1,  # 使用 'phi1' 作为旋转角度
                image_width=width,
                image_height=height,
                num=point['num'],  # 使用 CSV 中的 'num' 作为编号
                homographies_dir=homographies_dir  # 传递单应性矩阵保存目录
            )
        
        except KeyError as e:
            print(f"GPS 点 {idx+1} 缺少必要的键: {e}. 点内容: {point}")
        except ValueError as ve:
            print(f"GPS 点 {idx+1} 经纬度超出图像边界 ({lat}, {lon}). Skipping.")
        except Exception as e:
            print(f"转换或裁剪 GPS 点 {idx+1} 时出错: {e}. 点内容: {point}")

    print(f"有效的 GPS 点数量: {len(valid_gps_points)}")

    if len(valid_gps_points) == 0:
        print("没有有效的 GPS 点可供可视化。")
        return

    # 可视化所有有效的 GPS 点
    visualize_points(dom_path, valid_gps_points)

if __name__ == "__main__":
    main()
