import os
import glob
import numpy as np
import cv2

# ====== 使用 Qt5Agg 后端 ======
import matplotlib
matplotlib.use('Qt5Agg')  # 也可使用 'TkAgg' 或其他可视化后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

import json  # 确保在函数外部导入

# 定义 START_FRAME 和 END_FRAME
START_FRAME = 400
END_FRAME   = 700


def load_camera_and_board_params(config_path: str):
    """
    从 JSON 或其它配置文件中读取相机标定参数、标定板信息。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    camera_matrix = np.array(cfg["camera_params"]["camera_matrix"], dtype=np.float32)
    dist_coeffs   = np.array(cfg["camera_params"]["dist_coeffs"], dtype=np.float32)

    pattern_size      = tuple(cfg["calib_board"]["CALIB_BOARD_SIZE"])  # 例如 (6,8)
    circle_distance   = cfg["calib_board"]["CALIB_BOARD_DIST"]         # 圆心距离 (mm)

    DETECTION_TOP     = cfg["roi"]["DETECTION_TOP_1"]
    DETECTION_BOTTOM  = cfg["roi"]["DETECTION_BOTTOM_2"]
    BINARY_THRESHOLD  = cfg["thresholds"]["BINARY_THRESHOLD"]

    # BINARY_THRESHOLD  = 200

    GAUSSIAN_KERNEL   = 3  # 可自行配置或从 cfg 中获取

    return (camera_matrix, dist_coeffs,
            pattern_size, circle_distance,
            DETECTION_TOP, DETECTION_BOTTOM,
            BINARY_THRESHOLD, GAUSSIAN_KERNEL)


def load_timestamps(timestamp_file: str):
    """
    读取时间戳文件到 {帧号: timestamp} 字典。
    文件格式示例:
        1,1737852335.6760855
        2,1737852335.6936588
        ...
    """
    timestamps_dict = {}
    with open(timestamp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, t_str = line.split(',')
            idx = int(idx_str)
            t   = float(t_str)
            timestamps_dict[idx] = t
    return timestamps_dict


def compute_average_movement(indices, y_list, start_frame, end_frame):
    """
    计算从 start_frame 到 end_frame 在 y_list 上的平均每帧移动量 (mm/帧)。

    indices      : np.array[int]   记录帧号
    y_list       : np.array[float] 标定板在相机坐标系下的 tvec[1] 值（mm）
    start_frame  : int             起始帧号 (如 100)
    end_frame    : int             结束帧号 (如 400)

    return: float, 表示 (mm/帧)
    """
    # 使用布尔掩码选择范围内的帧
    mask = (indices >= start_frame) & (indices <= end_frame)
    selected_y = y_list[mask]

    if len(selected_y) < 2:
        print(f"   >>> [WARNING] 选定范围内的有效帧不足，无法计算平均移动。")
        return 0.0

    # 计算相邻帧的位移
    mm_per_frame = np.diff(selected_y)
    average_movement = np.mean(mm_per_frame)
    return average_movement


def compute_board_vertical_displacement_per_frame(
    image_folder: str,
    timestamp_file: str,
    config_path: str
):
    """
    依次检测标定板的圆心并 solvePnP，得到 tvec[1] (假设即竖直方向, 单位 mm)；
    然后只关心相邻帧在 tvec[1] 上的差值 (mm/frame)。

    同时返回对应帧的时间戳列表。
    """
    (camera_matrix, dist_coeffs,
     pattern_size, circle_distance,
     DETECTION_TOP, DETECTION_BOTTOM,
     BINARY_THRESHOLD, GAUSSIAN_KERNEL) = load_camera_and_board_params(config_path)

    # 准备标定板 3D 坐标 objp
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern_size[0], 0:pattern_size[1]]
                   .T.reshape(-1, 2)) * circle_distance

    # 读取时间戳
    timestamps_dict = load_timestamps(timestamp_file)

    # 按编号顺序读取图像
    images = glob.glob(os.path.join(image_folder, '*.png'))
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    y_list   = []  # tvec[1] 数值
    indices  = []  # 帧号
    timestamps_list = []  # 对应帧的时间戳

    for img_path in tqdm(images, desc="   >>> Reading Images..."):
        filename = os.path.basename(img_path)
        idx_str  = os.path.splitext(filename)[0]
        try:
            idx      = int(idx_str)
        except ValueError:
            print(f"[WARN] 无法解析帧号: {img_path}")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取图像: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 限制 ROI
        gray[:DETECTION_TOP, :]    = 0
        gray[DETECTION_BOTTOM:, :] = 0

        # 预处理
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        # 设置自定义阈值
        _, gray_thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        # 检测圆心
        found, centers = cv2.findCirclesGrid(gray_thresh, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        if not found:
            print(f"[INFO] 未检测到圆心, 跳过: {img_path}")
            continue

        # solvePnP => (rvec, tvec)
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=objp,
            imagePoints=centers,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            print(f"[WARN] solvePnP 失败: {img_path}")
            continue

        tvec = tvec.flatten()  # (3,)
        y_pos = tvec[1]        # 假设相机坐标系下 Y 为竖直分量 (mm)

        indices.append(idx)
        y_list.append(y_pos)
        # 获取对应帧的时间戳
        if idx in timestamps_dict:
            timestamps_list.append(timestamps_dict[idx])
        else:
            print(f"[WARN] 帧号 {idx} 没有对应的时间戳。")
            timestamps_list.append(None)  # 或者其他处理方式

    # 如果无法获取足够帧，直接返回 None
    if len(y_list) < 2:
        print("[ERROR] 有效数据帧不足，无法计算 mm/frame。")
        return None, None, None, None, None, None

    # 相邻帧在竖直方向上的位移差 => mm/frame
    mm_per_frame_list = np.diff(y_list).tolist()  # Convert to list

    # 相邻帧的时间戳差
    dt_list = []
    valid_mm_per_frame = []
    for i in range(len(timestamps_list) - 1):
        t1 = timestamps_list[i]
        t2 = timestamps_list[i + 1]
        if t1 is None or t2 is None:
            print(f"[WARN] 帧号 {indices[i]} 或 {indices[i+1]} 没有有效时间戳，跳过移动距离计算。")
            dt_list.append(None)
            valid_mm_per_frame.append(None)
        else:
            dt = t2 - t1
            if dt <= 0:
                print(f"[WARN] 帧号 {indices[i]} 到 {indices[i+1]} 的时间差 <= 0，跳过。")
                dt_list.append(None)
                valid_mm_per_frame.append(None)
            else:
                dt_list.append(dt)
                valid_mm_per_frame.append(mm_per_frame_list[i])

    return indices, y_list, mm_per_frame_list, dt_list, valid_mm_per_frame, timestamps_list


def calculate_rmse(actual, predicted):
    """
    计算均方根误差 (RMSE)

    参数:
        actual (np.array): 实际值
        predicted (np.array): 预测值或参考值

    返回:
        float: RMSE 值
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mape(actual, predicted):
    """
    计算平均绝对百分比误差 (MAPE)

    参数:
        actual (np.array): 实际值
        predicted (np.array): 预测值或参考值

    返回:
        float: MAPE 值（百分比）
    """
    # 避免除以零
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def main():
    image_folder   = "../data_source/captures.calib"
    timestamp_file = "../data_source/captures.calib/timestamps.txt"
    config_path    = "../data_source/system_calibration_result/laser_params_export.json"

    # 计算每帧的竖直位移和时间差
    result = compute_board_vertical_displacement_per_frame(
        image_folder, timestamp_file, config_path
    )

    if result is None or len(result) < 6:
        print("   >>> [ERROR] 数据加载失败。")
        return

    indices, y_list, mm_per_frame_list, dt_list, valid_mm_per_frame, timestamps_list = result

    # 将 indices 和 y_list 转换为 NumPy 数组
    indices = np.array(indices)
    y_list = np.array(y_list)
    mm_per_frame_list = np.array(mm_per_frame_list)
    dt_list = np.array(dt_list)
    valid_mm_per_frame = np.array(valid_mm_per_frame)

    # 打印调试信息
    print(f"indices 类型: {type(indices)}, 示例内容: {indices[:10]}")
    print(f"y_list 类型: {type(y_list)}, 示例内容: {y_list[:10]}")
    print(f"mm_per_frame_list 类型: {type(mm_per_frame_list)}, 示例内容: {mm_per_frame_list[:10]}")
    print(f"dt_list 类型: {type(dt_list)}, 示例内容: {dt_list[:10]}")
    print(f"valid_mm_per_frame 类型: {type(valid_mm_per_frame)}, 示例内容: {valid_mm_per_frame[:10]}")

    # 计算指定帧范围内的平均每帧移动距离
    avg_mm_per_frame = compute_average_movement(indices, y_list, START_FRAME, END_FRAME)
    print(f"从第{START_FRAME}帧到第{END_FRAME}帧，平均每帧移动: {avg_mm_per_frame:.3f} mm/帧")

    # 计算指定帧范围内的基于时间戳的平均移动速度 (mm/s)
    # 修正 mask_time 以匹配 valid_mm_per_frame 和 dt_list 的长度
    mask_time = (indices[:-1] >= START_FRAME) & (indices[:-1] <= END_FRAME - 1)  # 修正这里
    valid_mask = mask_time & (~np.isnan(valid_mm_per_frame)) & (dt_list > 0)

    # 提取有效的 mm/frame 和 dt
    selected_mm_per_frame = valid_mm_per_frame[mask_time]
    selected_dt = dt_list[mask_time]

    # 进一步过滤有效数据
    valid_entries = (~np.isnan(selected_mm_per_frame)) & (selected_dt > 0)
    selected_mm_per_frame = selected_mm_per_frame[valid_entries]
    selected_dt = selected_dt[valid_entries]

    if len(selected_dt) > 0:
        movement_per_time = selected_mm_per_frame / selected_dt  # mm/s
        avg_movement_per_time = np.mean(movement_per_time)
        print(f"从第{START_FRAME}帧到第{END_FRAME}帧，基于时间戳的平均移动速度: {avg_movement_per_time:.3f} mm/s")
    else:
        avg_movement_per_time = 0.0
        print(f"从第{START_FRAME}帧到第{END_FRAME}帧，基于时间戳的平均移动速度: 数据不足，无法计算。")

    # === 计算 RMSE 和 MAPE ===
    if len(selected_dt) > 0:
        # 以平均移动速度作为参考值
        reference_movement = np.full_like(movement_per_time, avg_movement_per_time)

        rmse = calculate_rmse(movement_per_time, reference_movement)
        mape = calculate_mape(movement_per_time, reference_movement)

        print(f"RMSE（基于时间戳的移动速度）: {rmse:.3f} mm/s")
        print(f"MAPE（基于时间戳的移动速度）: {mape:.2f}%")
    else:
        print("   >>> [WARNING] 数据不足，无法计算 RMSE 和 MAPE。")

    # === 可视化：以帧号为横坐标 ===
    # 创建3个子图，垂直排列
    fig, ax = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    # 1) 标定板竖直位置 vs. 帧号
    ax[0].plot(indices, y_list, marker='o', label='tvec_y (mm)')
    ax[0].set_ylabel("Vertical Position (mm)")
    ax[0].legend()
    ax[0].grid(True)

    # 2) 相邻帧竖直位移 vs. 帧号（中点）
    #    mm_per_frame_list 的长度比 indices 少 1
    mid_indices = (indices[:-1] + indices[1:]) / 2.0
    ax[1].plot(mid_indices, mm_per_frame_list, marker='x', color='r', label='Vertical Movement (mm/frame)')
    ax[1].set_ylabel("Movement (mm/frame)")
    ax[1].legend()
    ax[1].grid(True)

    # 3) 第二个子图的局部放大视图
    # 确保 START_FRAME 和 END_FRAME 在 mid_indices 范围内
    if START_FRAME < mid_indices[0] or END_FRAME > mid_indices[-1]:
        print("   >>> [WARNING] START_FRAME 或 END_FRAME 超出 mid_indices 范围。请检查设置。")
    else:
        # 使用布尔掩码选择范围内的帧
        mask = (mid_indices >= START_FRAME) & (mid_indices <= END_FRAME)
        selected_mid_indices = mid_indices[mask]
        selected_mm_per_frame = mm_per_frame_list[mask]

        if len(selected_mm_per_frame) < 1:
            print("   >>> [WARNING] 选定范围内的有效帧不足，无法绘制局部放大视图。")
        else:
            # 计算平均值
            average_mm = np.mean(selected_mm_per_frame)

            # 绘制局部放大视图
            ax[2].plot(selected_mid_indices, selected_mm_per_frame, marker='x', color='orange', label='Selected Vertical Movement (mm/frame)')
            ax[2].axhline(y=average_mm, color='g', linestyle='--', label=f'Average = {average_mm:.3f} mm/frame')

            ax[2].set_xlabel("Frame Index")
            ax[2].set_ylabel("Movement (mm/frame)")
            ax[2].legend()
            ax[2].grid(True)

    plt.suptitle("Calibration Board Vertical Movement (X-axis = Frame Index)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应标题
    plt.show()

    # === 新的画板：每个时间测量间隔内的移动距离 ===
    # 计算每个时间测量间隔内的移动距离 (mm/s)
    # movement_per_time = dy / dt

    # 过滤无效数据（dt 为 None 或 0）
    valid_mask_time = (dt_list > 0) & (~np.isnan(valid_mm_per_frame))
    movement_per_time_full = valid_mm_per_frame[valid_mask_time] / dt_list[valid_mask_time]

    # 选择对应的时间点（使用中点时间）
    # movement_per_time_full 对应 frame i -> frame i+1 的移动距离，使用 frame i 和 frame i+1 的时间戳计算中点时间
    timestamps_mid = []
    for i in range(len(timestamps_list) - 1):
        t1 = timestamps_list[i]
        t2 = timestamps_list[i + 1]
        if t1 is not None and t2 is not None:
            mid = (t1 + t2) / 2.0
            timestamps_mid.append(mid)
        else:
            timestamps_mid.append(None)
    timestamps_mid = np.array(timestamps_mid)

    # 现在，使用 valid_mask_time 来过滤出有效的 mid_timestamps
    selected_timestamps = timestamps_mid[valid_mask_time]

    # 创建新的画板
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(selected_timestamps, movement_per_time_full, marker='s', linestyle='-', color='m',
             label='Movement per Time Interval (mm/s)')
    if len(movement_per_time_full) > 0:
        average_movement_per_time_full = np.mean(movement_per_time_full)
        ax2.axhline(y=average_movement_per_time_full, color='c', linestyle='--',
                    label=f'Average = {average_movement_per_time_full:.3f} mm/s')
    else:
        print("   >>> [WARNING] 没有有效的移动距离数据用于绘制。")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Movement Distance (mm/s)")
    ax2.set_title("Movement per Time Interval")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # === 保存平均移动距离（可选） ===
    # 保存平均每帧移动距离和基于时间的平均移动速度到 JSON 文件
    movement_data = {
        "avg_mm_per_frame": avg_mm_per_frame,
        "avg_mm_per_sec": avg_movement_per_time if len(selected_dt) > 0 else None
    }
    with open("average_movement.json", "w") as f:
        json.dump(movement_data, f)
    print("[INFO] 平均移动距离已保存到 'average_movement.json' 文件。")

    # === 打印第二段代码需要的基于时间戳的移动速度 ===
    if avg_movement_per_time > 0:
        print(f"从第{START_FRAME}帧到第{END_FRAME}帧，第二段代码需要的基于时间戳的平均移动速度: {avg_movement_per_time:.3f} mm/s")
    else:
        print(f"从第{START_FRAME}帧到第{END_FRAME}帧，基于时间戳的平均移动速度数据不足，无法提供第二段代码需要的值。")


if __name__ == "__main__":
    main()
