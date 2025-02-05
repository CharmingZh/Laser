#!/usr/bin/00_env python3
# -*- coding: utf-8 -*-

"""
线激光三维重建 (相机坐标系版本) + 点云着色 (jet 映射)

本脚本：
1) 从 JSON 文件中读取：camera_matrix, dist_coeffs, laser_plane_params(a,b,c,d), ROI, 阈值等
2) 从 ./data_source/captures 文件夹读取若干激光图 (1.png,2.png,...)，逐帧排序
3) 对每帧：去畸变 -> ROI -> 灰度 -> 按列查找最亮像素(u,v)
   -> 在相机坐标系下求射线与激光平面(a*x_c + b*y_c + c*z_c + d=0)的交点(X_c,Y_c,Z_c)
4) 如果需要“平均每帧移动距离”(默认 0.943 mm/帧)，假设相机坐标系下 Y_c 就是物体前进方向，
   则对 Y_c 累加 offset=(frame_id - min_frame_id)*0.943
5) 将每帧 3D 点用红色标记在图像中显示(可选)，然后合并到最终点云
6) **对合并后的点云做 jet 颜色映射**，并保存点云、用 Open3D 显示
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import os
import glob
import json
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
import time

# 如果需要图形界面
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 用于给点云上色
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#######################################
# 可调参数
#######################################
AVERAGE_SPEED_MM_PER_FRAME = -0.732  # 在相机系下 Y_c 轴的每帧平移
DEBUG_PRINT_PER_FRAME      = False   # 是否打印每帧提取点数
SHOW_VISUALIZATION         = 1424   # 是否弹出窗口显示标记后的图像
SKIP_FRAME                 = 50
image_folder = "../data_source/captures.shape.1"
# image_folder = "../data_source/captures.meat"

#######################################
# 函数：对点云按某一坐标轴上色 (jet)
#######################################
def colorize_point_cloud(pcd, axis='z'):
    """
    根据点云某一轴(默认 z 轴)的值进行上色.
    使用 jet 色彩空间.
    """
    points = np.asarray(pcd.points)

    if len(points) == 0:
        print("[WARN] colorize_point_cloud: Empty point cloud!")
        return pcd

    # 选择用于上色的轴
    if axis == 'x':
        color_axis = points[:, 0]
    elif axis == 'y':
        color_axis = points[:, 1]
    else:
        color_axis = points[:, 2]

    # 归一化到 [0, 1]
    norm = mcolors.Normalize(vmin=color_axis.min(), vmax=color_axis.max())
    normalized_colors = norm(color_axis)

    # 使用 jet 色彩映射
    colormap = plt.get_cmap('jet')
    colors = colormap(normalized_colors)[:, :3]  # 去掉 alpha 通道

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Point cloud colored by '{axis}' axis using jet colormap.")
    return pcd

#######################################
# 1) 从 JSON 读取所有必要参数（相机系）
#######################################
def load_camera_coords_params(json_path: str):
    """
    从 JSON 中读取:
      - camera_params: camera_matrix, dist_coeffs
      - laser_plane_params: a,b,c,d (已在相机坐标系下标定好的激光平面)
      - thresholds/roi: ROI, 阈值
    不使用 world_extrinsics (R_world, t_world).
    返回一个包含所需信息的 dict.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 1. 相机内参 + 畸变
    camera_matrix = np.array(cfg["camera_params"]["camera_matrix"], dtype=np.float32)
    dist_coeffs   = np.array(cfg["camera_params"]["dist_coeffs"],   dtype=np.float32)

    # 2. 激光平面 (a*x_c + b*y_c + c*z_c + d = 0) 在相机系下
    (a_1, b_1, c_1, d_1) = cfg["laser_plane_params_1"]
    (a_2, b_2, c_2, d_2) = cfg["laser_plane_params_2"]

    # 3. 阈值(如果 JSON 里没有，也可以自己写死)
    # threshold_1 = cfg.get("threshold_1", 10)
    # threshold_2 = cfg.get("threshold_2", 10)
    threshold_1 = cfg["thresholds"]["RECON_LASER_DETECT_THRESHOLD_1"]
    threshold_2 = cfg["thresholds"]["RECON_LASER_DETECT_THRESHOLD_2"]

    # 4. ROI
    roi_cfg = cfg.get("roi", {})
    top_1    = roi_cfg.get("DETECTION_TOP_1", 0)
    bottom_1 = roi_cfg.get("DETECTION_BOTTOM_1", 99999)
    left_1   = roi_cfg.get("LASER_LINE_LEFT_BOUND_1", 0)
    right_1  = roi_cfg.get("LASER_LINE_RIGHT_BOUND_1", 99999)

    # 4. ROI
    roi_cfg  = cfg.get("roi", {})
    top_2    = roi_cfg.get("DETECTION_TOP_2", 0)
    bottom_2 = roi_cfg.get("DETECTION_BOTTOM_2", 99999)
    left_2   = roi_cfg.get("LASER_LINE_LEFT_BOUND_2", 0)
    right_2  = roi_cfg.get("LASER_LINE_RIGHT_BOUND_2", 99999)


    # 自定义修改
    threshold_1 = 6
    threshold_2 = 5

    top_1 = 530
    bottom_1 = 700
    left_1 = 350
    right_1 = 1000

    top_2 = 725
    bottom_2 = 950
    left_2 = 350
    right_2 = 1000



    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs":   dist_coeffs,

        "plane_1":         (a_1, b_1, c_1, d_1),
        "threshold_1":     threshold_1,
        "roi_top_1":       top_1,
        "roi_bottom_1":    bottom_1,
        "roi_left_1":      left_1,
        "roi_right_1":     right_1,

        "plane_2":         (a_2, b_2, c_2, d_2),
        "threshold_2":     threshold_2,
        "roi_top_2":       top_2,
        "roi_bottom_2":    bottom_2,
        "roi_left_2":      left_2,
        "roi_right_2":     right_2
    }


#######################################
# 2) 主流程: 在相机坐标系下重建 + 点云上色
#######################################
def reconstruct_laser_in_camera_coords(
    image_folder: str,
    config_path: str,
    output_ply: str = "camera_coords_laser_colored.ply"
):
    """
    使用相机系下的激光平面(a,b,c,d)，不再做 R_world 或 t_world.
    逐帧检测激光像素 -> 射线-平面相交 -> (X_c, Y_c, Z_c).
    若需要, 对 Y_c 做 "AVERAGE_SPEED_MM_PER_FRAME" * (frame_id - min_frame_id).
    最后对点云进行 jet 映射上色并保存.
    """
    # A) 读取参数
    cfg = load_camera_coords_params(config_path)
    K            = cfg["camera_matrix"]
    dist         = cfg["dist_coeffs"]
    # (a,b,c,d)    = cfg["plane"]

    (a_1, b_1, c_1, d_1) = cfg["plane_1"]
    (a_2, b_2, c_2, d_2) = cfg["plane_2"]

    threshold_1    = cfg["threshold_1"]
    top_1          = cfg["roi_top_1"]
    bottom_1       = cfg["roi_bottom_1"]
    left_bound_1   = cfg["roi_left_1"]
    right_bound_1  = cfg["roi_right_1"]


    threshold_2    = cfg["threshold_2"]
    top_2          = cfg["roi_top_2"]
    bottom_2       = cfg["roi_bottom_2"]
    left_bound_2   = cfg["roi_left_2"]
    right_bound_2  = cfg["roi_right_2"]


    # threshold    = cfg["threshold"]
    # top          = cfg["roi_top"]
    # bottom       = cfg["roi_bottom"]
    # left_bound   = cfg["roi_left"]
    # right_bound  = cfg["roi_right"]

    # B) 获取图像列表
    images = glob.glob(os.path.join(image_folder, "*.png"))
    if not images:
        print(f"[ERROR] No .png images found in {image_folder}")
        return None
    # 按帧号排序
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    min_frame_id = int(os.path.splitext(os.path.basename(images[0]))[0])

    all_points_3d_1 = []

    # Line 1
    # C) 逐帧处理
    time_lst = []
    for img_path in tqdm(images, desc="Processing"):
        fname = os.path.basename(img_path)
        frame_id = int(os.path.splitext(fname)[0])

        # (可选：如果需要跳过某些帧，可以加逻辑，如:)
        if frame_id < SKIP_FRAME:
            continue

        # 1) 读取图像 & 去畸变
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        img_undist = cv2.undistort(img, K, dist, None, K)

        # 2) 转灰度
        gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
        _, _, gray = cv2.split(img_undist)

        # 3) ROI
        h, w = gray.shape
        gray[:top_1, :] = 0
        if bottom_1 < h:
            gray[bottom_1:, :] = 0

        # 2. 设置中值滤波核大小
        kernel_size = 5
        # 3. 进行中值滤波
        gray = cv2.medianBlur(gray, kernel_size)

        vis_img = img_undist.copy()
        frame_pts_camera = []

        time_start = time.time()

        # 4) 按列查找激光像素
        col_start = max(left_bound_1, 0)
        col_end   = min(right_bound_1, w)

        for u in range(col_start, col_end):
            col_data = gray[:, u]
            max_val = col_data.max()
            if max_val < threshold_1:
                continue

            rows = np.where(col_data == max_val)[0]
            if len(rows) == 0:
                continue

            # 使用高斯滤波平滑亮度曲线
            smoothed = gaussian_filter1d(col_data.astype(float), sigma=7)

            # 找到平滑后亮度的最大值及其位置
            peak_idx = np.argmax(smoothed)

            # 使用邻域拟合高斯曲线获取亚像素位置
            if 1 <= peak_idx < len(smoothed) - 1:
                alpha = smoothed[peak_idx - 1]
                beta = smoothed[peak_idx]
                gamma = smoothed[peak_idx + 1]
                # 简单的二次插值
                p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                v_subpixel = peak_idx + p
            else:
                v_subpixel = peak_idx

            v = v_subpixel  # 保留为浮点数进行后续计算

            # 标记红色（可视化调试用）
            if 0 <= v < h:
                vis_img[int(v), u] = (0, 255, 0)

            # 计算在相机系下(X_c, Y_c, Z_c)
            pix_hom = np.array([u, v, 1.0], dtype=np.float32)
            ray_dir_cam = np.linalg.inv(K) @ pix_hom
            ray_dir_cam /= np.linalg.norm(ray_dir_cam)
            dx, dy, dz = ray_dir_cam

            denom = a_1 * dx + b_1 * dy + c_1 * dz
            if abs(denom) < 1e-12:
                continue

            t = -d_1 / denom
            if t < 0:
                # 在相机后方
                continue

            X_c = t * dx
            Y_c = t * dy
            Z_c = t * dz
            frame_pts_camera.append([X_c, Y_c, Z_c])

        if not frame_pts_camera:
            if DEBUG_PRINT_PER_FRAME:
                print(f"[INFO] Frame {frame_id} => no data_source points.")
            continue

        frame_pts_camera = np.array(frame_pts_camera, dtype=np.float32)

        # 5) 假设 Y_c 是运动方向 => 累加 offset
        offset_frames = frame_id - min_frame_id
        y_shift       = offset_frames * AVERAGE_SPEED_MM_PER_FRAME
        frame_pts_camera[:, 1] -= y_shift

        if DEBUG_PRINT_PER_FRAME:
            print(f"Frame {frame_id} => detected {len(frame_pts_camera)} points.")

        time_lst.append((time.time() - time_start) * 1000)

        if SHOW_VISUALIZATION:
            cv2.imshow("Laser Marked", vis_img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        all_points_3d_1.append(frame_pts_camera)

    print("平均每帧查找激光耗时 (ms)：", np.mean(time_lst))

    # D) 合并
    if not all_points_3d_1:
        print("[ERROR] No valid points from all frames.")
        return None

    merged_points = np.vstack(all_points_3d_1)
    print(f"[SUMMARY] Total points: {len(merged_points)}")

    # E) 写到Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)

    # ★★★ 在这里对点云按 z 轴上色 (你也可以选 'x' 或 'y') ★★★
    pcd_colored_1 = colorize_point_cloud(pcd, axis='z')

    # 保存
    o3d.io.write_point_cloud("point_cloud_1.ply", pcd_colored_1)
    print(f"[INFO] Colored point cloud saved to: {output_ply}")



    # Line 2
    all_points_3d_2 = []
    # C) 逐帧处理
    time_lst = []
    for img_path in tqdm(images, desc="Processing"):
        fname = os.path.basename(img_path)
        frame_id = int(os.path.splitext(fname)[0])

        # (可选：如果需要跳过某些帧，可以加逻辑，如:)
        # if frame_id < 210:
        #     continue

        # 1) 读取图像 & 去畸变
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        img_undist = cv2.undistort(img, K, dist, None, K)

        # 2) 转灰度
        gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
        gray, _, _ = cv2.split(img_undist)

        # 3) ROI
        h, w = gray.shape
        gray[:top_2, :] = 0
        if bottom_2 < h:
            gray[bottom_2:, :] = 0

        # 2. 设置中值滤波核大小
        # 通常3或5较常用；核越大平滑效果越强，但细节损失也越明显
        kernel_size = 3

        # 3. 进行中值滤波
        gray = cv2.medianBlur(gray, kernel_size)

        vis_img = img_undist.copy()
        frame_pts_camera = []

        time_start = time.time()

        # 4) 按列查找激光像素
        col_start = max(left_bound_2, 0)
        col_end = min(right_bound_2, w)

        for u in range(col_start, col_end):
            col_data = gray[:, u]
            max_val = col_data.max()
            if max_val < threshold_2:
                continue

            rows = np.where(col_data == max_val)[0]
            if len(rows) == 0:
                continue

            # 使用高斯滤波平滑亮度曲线
            smoothed = gaussian_filter1d(col_data.astype(float), sigma=15)

            # 找到平滑后亮度的最大值及其位置
            peak_idx = np.argmax(smoothed)

            # 使用邻域拟合高斯曲线获取亚像素位置
            if 1 <= peak_idx < len(smoothed) - 1:
                alpha = smoothed[peak_idx - 1]
                beta = smoothed[peak_idx]
                gamma = smoothed[peak_idx + 1]
                # 简单的二次插值
                p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                v_subpixel = peak_idx + p
            else:
                v_subpixel = peak_idx

            v = v_subpixel  # 保留为浮点数进行后续计算

            # 标记红色（可视化调试用）
            if 0 <= v < h:
                vis_img[int(v), u] = (0, 255, 0)

            # 计算在相机系下(X_c, Y_c, Z_c)
            pix_hom = np.array([u, v, 1.0], dtype=np.float32)
            ray_dir_cam = np.linalg.inv(K) @ pix_hom
            ray_dir_cam /= np.linalg.norm(ray_dir_cam)
            dx, dy, dz = ray_dir_cam

            denom = a_2 * dx + b_2 * dy + c_2 * dz
            if abs(denom) < 1e-12:
                continue

            t = -d_2 / denom
            if t < 0:
                # 在相机后方
                continue

            X_c = t * dx
            Y_c = t * dy
            Z_c = t * dz
            frame_pts_camera.append([X_c, Y_c, Z_c])

        if not frame_pts_camera:
            if DEBUG_PRINT_PER_FRAME:
                print(f"[INFO] Frame {frame_id} => no data_source points.")
            continue

        frame_pts_camera = np.array(frame_pts_camera, dtype=np.float32)

        # 5) 假设 Y_c 是运动方向 => 累加 offset
        offset_frames = frame_id - min_frame_id
        y_shift = offset_frames * AVERAGE_SPEED_MM_PER_FRAME
        frame_pts_camera[:, 1] -= y_shift

        if DEBUG_PRINT_PER_FRAME:
            print(f"Frame {frame_id} => detected {len(frame_pts_camera)} points.")

        time_lst.append((time.time() - time_start) * 1000)

        if SHOW_VISUALIZATION:
            cv2.imshow("Laser Marked", vis_img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        all_points_3d_2.append(frame_pts_camera)

    print("平均每帧查找激光耗时 (ms)：", np.mean(time_lst))

    # D) 合并
    if not all_points_3d_2:
        print("[ERROR] No valid points from all frames.")
        return None

    merged_points = np.vstack(all_points_3d_2)
    print(f"[SUMMARY] Total points: {len(merged_points)}")

    # E) 写到Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)

    # ★★★ 在这里对点云按 z 轴上色 (你也可以选 'x' 或 'y') ★★★
    pcd_colored_2 = colorize_point_cloud(pcd, axis='z')


    ###

    # 保存
    o3d.io.write_point_cloud("point_cloud_2.ply", pcd_colored_2)
    print(f"[INFO] Colored point cloud saved to: {output_ply}")

    # 关闭OpenCV窗口
    cv2.destroyAllWindows()

    # # Open3D可视化
    # o3d.visualization.draw_geometries([pcd_colored], window_name="Colored Laser in Camera Coords")


def main():
    # 1) 假设图像在 ./data_source/captures
    # image_folder = "../data_source/capture_target_4"
    # image_folder = "../data_source/captures"

    # 2) JSON 文件(含 camera_params, laser_plane_params, roi, thresholds等)
    config_path  = "../data_source/system_calibration_result/laser_params_export.json"

    # 3) 输出点云文件
    output_ply   = "output.ply"

    reconstruct_laser_in_camera_coords(
        image_folder, config_path, output_ply
    )

if __name__=="__main__":
    main()
