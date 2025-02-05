import json

import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

import matplotlib
matplotlib.use('Qt5Agg')  # 或 'TkAgg'，确保支持交互窗口

import open3d as o3d

from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d


# 1 Calibrating Camera
CALIB_BOARD_SIZE = (6, 6)
CALIB_BOARD_DIST = 10

DETECTION_TOP    = 0
DETECTION_BOTTOM = 1424

GAUSSIAN_KERNEL  = 7
BINARY_THRESHOLD = 170

CALIBRATION_VIZ  = False
VIZ_FALSE_ONLY   = False
CALIBRATION_SAVE = False

CAL_EACH_REPROJ_ERR  = False
EVERY_REPROJ_DISPLAY = False
EVERY_REPROJ_VIZ     = False

# 2 World Coordinate Define
NUMBERS_TO_FIT_AXIS = 1

WORLD_CAMERA_COORD_DISPLAY = False

# 3 Calibrating Laser Line && Plane
DETECTION_TOP_LASER_1    = 500
DETECTION_BOTTOM_LASER_1 = 700
LASER_LINE_LEFT_BOUND_LASER_1  = 500
LASER_LINE_RIGHT_BOUND_LASER_1 = 900
LASER_BINARY_THRESHOLD_1 = 1

DETECTION_TOP_LASER_2    = 700
DETECTION_BOTTOM_LASER_2 = 1424
LASER_LINE_LEFT_BOUND_LASER_2  = 500
LASER_LINE_RIGHT_BOUND_LASER_2 = 900
LASER_BINARY_THRESHOLD_2 = 1

LASER_CALIB_BOARD_DETECTED_DISPLAY = False
LASER_LINES_THRESH_DISPLAY         = True
LASER_LINES_FITTED_DISPLAY         = True

DISPLAY_SIZE = (1424, 1424)



# 控制可视化组件的显示
BOARD_POINTS_DISPLAY      = False
BOARD_PLANES_DISPLAY      = False
LASER_CALIB_PLANE_DISPLAY = True
LASER_LINES_DISPLAY       = True
LASER_PLANE_DISPLAY       = True


# ------------------------------
# 1. Camera Calibration
# ------------------------------
def camera_calibration(image_path, pattern_size, circle_distance):
    """
    使用圆形标定板进行相机标定。
    参数：
        image_path: 标定板图片路径（支持通配符）
        pattern_size: 圆标定板的内点阵大小 (行, 列)
        circle_distance: 圆心之间的真实间距(单位:mm)，用于计算3D坐标
    返回：
        mtx   : 相机内参矩阵 (3x3)
        dist  : 畸变系数 (1x5)
        rvecs : 每张标定图对应的旋转向量列表
        tvecs : 每张标定图对应的平移向量列表
        objp  : 单块标定板上所有圆心在“物体坐标系”下的 3D 坐标 (pattern_size[0]*pattern_size[1], 3)
    """
    print("# 1 Camera Calibration.")
    # 单块标定板的 3D 物体坐标
    objp        = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[
                   0:pattern_size[0],
                   0:pattern_size[1]
                   ].T.reshape(-1, 2)) * circle_distance

    objpoints = []  # 存放所有图对应的 objp
    imgpoints = []  # 存放所有图对应的 2D 圆心

    valid_images = []  # 存放成功检测到圆心的图像文件名

    # Read all images in the folder.
    images = glob.glob(image_path)
    # Sort all images by sequences.
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not images:
        raise ValueError(f"   >>> [ERROR] No images found in the path: {image_path}")

    w, h = 0, 0
    for fname in tqdm(images, desc="   >>> Reading Images..."):
        img = cv2.imread(fname)
        w, h = img.shape[1], img.shape[0]
        if img is None:
            print(f"   >>> [ERROR] Can't Read Image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Limit Detection ROI Area
        gray[0:DETECTION_TOP  , :] = 0
        gray[DETECTION_BOTTOM:, :] = 0

        # Image Augmentation
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        gray = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        # Circle Points Detection
        ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(centers)
            valid_images.append(fname)

            # Image For Visualization
            vis_img = img.copy()
            cv2.drawChessboardCorners(vis_img, pattern_size, centers, ret)

            if CALIBRATION_VIZ:
                # 创建一个可调整大小的窗口
                cv2.namedWindow(f"Detected Circles - {fname}", cv2.WINDOW_NORMAL)

                # 调整窗口大小 (宽度, 高度)
                cv2.resizeWindow(f"Detected Circles - {fname}", 300, 900)
                cv2.imshow(f"Detected Circles - {fname}", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if CALIBRATION_SAVE:
                save_path = "temp_results/camera_calibration/" + os.path.basename(fname)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_img)
        else:
            print(f"   >>> [ERROR] Non detected circle point in {os.path.basename(fname)}")
            if CALIBRATION_VIZ is True or VIZ_FALSE_ONLY is True:
                # 创建一个可调整大小的窗口
                cv2.namedWindow(f"Not Found - {fname}", cv2.WINDOW_NORMAL)

                # 调整窗口大小 (宽度, 高度)
                cv2.resizeWindow(f"Not Found - {fname}", 300, 900)

                cv2.imshow(f"Not Found - {fname}", gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise ValueError("   >>> [ERROR] Can't Calibrate because of Non detected circle point.")

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("   >>> # 1 [SUMMARY] Total Re-Projection Error =", ret)

    ################################################
    ## Calculating Re-Proj Errors in Each Images. ##
    ################################################
    per_image_errors_original  = []  # 原始重投影误差（不考虑畸变）
    per_image_errors_corrected = []  # 修正后重投影误差（考虑并修正畸变）

    if CAL_EACH_REPROJ_ERR is True:
        for i in range(len(objpoints)):
            # 将3D点投影到图像平面（不考虑畸变参数）
            imgpoints2_original, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, None)
            imgpoints2_original = imgpoints2_original.reshape(-1, 2)

            # 将3D点投影到图像平面（考虑畸变参数）
            imgpoints2_corrected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2_corrected = imgpoints2_corrected.reshape(-1, 2)

            # 原始检测点
            actual_imgpoints = imgpoints[i].reshape(-1, 2)

            # 计算欧氏距离
            error_original = np.linalg.norm(actual_imgpoints - imgpoints2_original, axis=1)
            mean_error_original = error_original.mean()
            per_image_errors_original.append(mean_error_original)

            error_corrected = np.linalg.norm(actual_imgpoints - imgpoints2_corrected, axis=1)
            mean_error_corrected = error_corrected.mean()
            per_image_errors_corrected.append(mean_error_corrected)

            if EVERY_REPROJ_DISPLAY is True:
                print(
                    f"      >>> Re-Projection Error (Original)  in Image \t{os.path.basename(valid_images[i])}:\t{mean_error_original:.4f} pixels")
                print(
                    f"      >>> Re-Projection Error (Corrected) in Image \t{os.path.basename(valid_images[i])}:\t{mean_error_corrected:.4f} pixels")

    ################################################
    ## Visualization Re-Proj Err. in Each Images. ##
    ################################################
    if EVERY_REPROJ_VIZ is True and CAL_EACH_REPROJ_ERR is True:
        # Generating x-axis identifiers
        x_labels = [f"Board {i}" for i in range(1, len(valid_images) + 1)]

        # Visualizing the errors
        plt.figure(figsize=(12, 7))

        # 绘制修正后的重投影误差
        plt.plot(x_labels, per_image_errors_corrected, marker='o', color='green', label="Corrected Reprojection Error")

        # 绘制原始的重投影误差
        plt.plot(x_labels, per_image_errors_original, marker='x', color='red', label="Original Reprojection Error")

        # 绘制总的重投影误差
        plt.axhline(y=ret, color='blue', linestyle='--', label="Total Reprojection Error")

        # Adding titles and labels
        plt.title("Reprojection Error Visualization", fontsize=14)
        plt.xlabel("Calibration Board Identifier", fontsize=12)
        plt.ylabel("Reprojection Error (pixels)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Display the plot
        plt.show()

    # New Camera Intrinsic Matrix
    mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), alpha=1
    )

    return mtx, dist, rvecs, tvecs, objp


# ------------------------------
# 2. World Coordinate Define
# ------------------------------
def define_world_coordinate_system(tvecs, rvecs, objp):
    """
    根据前10个平移向量和旋转向量定义新的世界坐标系。
        y轴：标定板移动的主要方向
        z轴：拟合平面的法向量
        x轴：y轴和z轴的叉积
    """
    print("# 2 World Coordinate Define.")
    # 拟合平面
    normal, centroid = fit_plane_to_tvecs(tvecs)

    # 跟踪第一个标定点在前 NUMBERS_TO_FIT_AXIS 个标定板中的位置
    first_point_movements = []
    for i in range(min(NUMBERS_TO_FIT_AXIS, len(rvecs))):
        rvec = rvecs[i].reshape(3, 1)
        tvec = tvecs[i].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        first_point_cam = (R @ objp[0].reshape(3, 1) + tvec).flatten()
        first_point_movements.append(first_point_cam)
    first_point_movements = np.array(first_point_movements)

    # 拟合移动方向作为y轴
    y_axis = fit_line_to_point_movements(first_point_movements)

    # z轴为平面法向量
    z_axis = normal

    # x轴为y轴和z轴的叉积
    # x_axis = (-1) * np.cross(y_axis, z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # 重新计算y轴以确保正交
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 形成旋转矩阵
    R_world = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3

    # 定义世界坐标系原点为第一个标定板的位置
    t_world = tvecs[0].flatten()

    return R_world, t_world


def fit_plane_to_tvecs(tvecs, threshold=1.0):
    """
    使用平移向量拟合一个平面，并剔除异常点。
    返回平面法向量和质心。
    """
    selected_tvecs = np.array(tvecs).reshape(-1, 3)

    # 初次拟合平面
    centroid = np.mean(selected_tvecs, axis=0)
    centered_tvecs = selected_tvecs - centroid
    _, _, Vt = svd(centered_tvecs)
    normal = Vt[2, :]  # 初始平面法向量

    # 剔除离平面远的点
    distances = np.abs(np.dot(centered_tvecs, normal))
    inliers = distances < threshold  # 根据阈值筛选内点

    # 重新拟合平面
    selected_tvecs = selected_tvecs[inliers]
    centroid = np.mean(selected_tvecs, axis=0)
    centered_tvecs = selected_tvecs - centroid
    _, _, Vt = svd(centered_tvecs)
    normal = Vt[2, :]

    # 确保法向量方向
    if normal[2] > 0:
        normal = -normal

    return normal, centroid


def fit_line_to_point_movements(point_movements):
    """
    拟合点的移动方向，返回方向向量。
    """
    # 计算质心
    centroid = np.mean(point_movements, axis=0)
    centered_movements = point_movements - centroid

    # 使用SVD进行线性拟合
    U, S, Vt = svd(centered_movements)
    direction = Vt[0, :]  # 主要移动方向

    # 确保方向向量的方向一致
    if direction[1] < 0:
        direction = -direction

    direction /= np.linalg.norm(direction)

    return direction


def verify_rotation_matrix(R):
    should_be_identity = np.dot(R.T, R)
    I = np.identity(3)
    return np.allclose(should_be_identity, I, atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def create_coordinate_frame(R, t, axis_length=0.5):
    """
    创建一个 Open3D 坐标系模型，应用旋转和平移。
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(t)
    return frame


def create_plane(normal, point, size=1.0, color=[0.5, 0.5, 0.5]):
    """
    创建一个平面，用于可视化。
    normal: 平面法向量
    point: 平面上的一个点
    size: 平面的尺寸
    color: 平面的颜色
    """
    # 创建平面的网格
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.01)
    mesh.paint_uniform_color(color)

    # 计算旋转矩阵，使平面的法向量与给定法向量一致
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    if s == 0:
        R = np.identity(3)
    else:
        c = np.dot(z_axis, normal)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.identity(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    mesh.rotate(R, center=(0, 0, 0))

    # 平移到指定位置
    mesh.translate(point)

    return mesh


def visualize_coordinate_systems(R_world, t_world, objp, rvecs, tvecs):
    """
    使用 Open3D 可视化原始相机坐标系、新的世界坐标系、拟合的平面和前十个标定板的标定点。
    """
    geometries = []

    # 相机坐标系
    R_camera = np.identity(3)
    t_camera = np.zeros(3)


    # 创建原始相机坐标系（红色）
    camera_frame = create_coordinate_frame(R_camera, t_camera, axis_length=100)
    geometries.append(camera_frame)

    # 创建新的世界坐标系（绿色）
    world_frame = create_coordinate_frame(R_world, t_world, axis_length=CALIB_BOARD_DIST*CALIB_BOARD_SIZE[0])
    geometries.append(world_frame)

    # 创建点云用于存储前十个标定板的标定点
    pcd = o3d.geometry.PointCloud()

    # 定义颜色列表（循环使用）
    colors = [
        [1, 0, 0],  # 红色
        [0, 1, 0],  # 绿色
        [0, 0, 1],  # 蓝色
        [1, 1, 0],  # 黄色
        [1, 0, 1],  # 品红
        [0, 1, 1],  # 青色
        [0.5, 0.5, 0.5],  # 灰色
        [1, 0.5, 0],  # 橙色
        [0.5, 0, 0.5],  # 紫色
        [0, 0.5, 0.5],  # 深青色
    ]

    # 遍历前十个标定板
    for i in range(min(10, len(rvecs))):
        rvec = rvecs[i].reshape(3, 1)
        tvec = tvecs[i].reshape(3, 1)

        # 计算旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 将objp转换到相机坐标系
        objp_cam = (R @ objp.T + tvec).T  # (N, 3)
        objp_world = (objp_cam - t_camera) @ R_camera.T  # (N, 3)

        # 创建点云
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(objp_world)
        color = colors[i % len(colors)]
        pcd_temp.paint_uniform_color(color)

        # 添加到几何体列表
        geometries.append(pcd_temp)

    # 打印几何体的信息以调试
    print(f"   >>> 添加了 {len(geometries)} 个几何体到可视化窗口。")
    print(f"   >>> 相机坐标系位置: {camera_frame.get_center()}")
    print(f"   >>> 世界坐标系位置: {world_frame.get_center()}")
    # print(f"拟合平面位置: {centroid}")
    print(f"   >>> 标定点云数量: {len(geometries) - 3}")  # 每个标定板一个点云

    # 可视化所有几何体
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Coordinate Systems and Calibration Points",
        width=1200,
        height=800,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=True,  # 显示坐标系的线框
        mesh_show_back_face=True
    )


def compute_plane_using_pose(centers, camera_matrix, dist_coeffs,
                             pattern_size, circle_distance,
                             R_world, t_world):
    """
    根据标定板的位姿计算其所在平面，并转换到世界坐标系。
    返回：
        平面参数 [a, b, c, d] in world coordinates
        标定板在世界坐标系中的点 (N,3)
    """
    # 构造物体坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern_size[0], 0:pattern_size[1]]
                   .T.reshape(-1, 2)) * circle_distance

    # 解畸变
    centers = cv2.undistortPoints(centers, camera_matrix, dist_coeffs,
                                  None, camera_matrix)

    # solvePnP
    ret, rvec, tvec = cv2.solvePnP(objp, centers, camera_matrix, dist_coeffs)
    if not ret:
        raise ValueError("   >>> [ERROR] 无法计算标定板的位姿！")

    R_cam, _ = cv2.Rodrigues(rvec)
    normal_cam = R_cam[:, 2]
    d_cam = -np.dot(normal_cam, tvec.squeeze())

    # 转换到世界坐标系（由于 R_world 是单位矩阵，t_world 是零向量，实际不变）
    normal_world = normal_cam
    d_world = d_cam

    # 转换标定板点到世界坐标系（不变）
    objp_cam = (R_cam @ objp.T).T + tvec.T
    objp_world = objp_cam

    return np.append(normal_world, d_world), objp_world


# ------------------------------
# 3. 标定板倾斜 & 激光平面拟合
# ------------------------------
def compute_plane_from_inclined_board(
        image_path,
        camera_matrix,
        dist_coeffs,
        laser_color,
        top_bound,
        bottom_bound,
        left_bound,
        right_bound,
        threshold,
        pattern_size=CALIB_BOARD_SIZE,
        circle_distance=CALIB_BOARD_DIST,
        R_world=None,
        t_world=None
):
    """
    根据倾斜的标定板图像计算其所在平面，并提取激光线三维信息，最后拟合激光平面(光平面)。
    所有平面参数均转换到统一的世界坐标系。
    返回：
        all_points_3d    : list[np.ndarray], 每个标定板对应的 3D 点云 (world coordinates)
        all_plane_params : list[(a,b,c,d)], 每个标定板拟合出的平面方程 in world coordinates
        laser_points     : list[np.ndarray], 每张激光图对应的一条 3D 点 (线) in world coordinates
        laser_plane_params : (a,b,c,d)，所有激光点拟合出的光平面 in world coordinates
    """
    print("# 3 Calibrating Laser Line && Plane")
    img_cb_only = []  # 仅包含标定板的图片
    # img_cb_laser = []  # 激光投射到相应的标定板的图片
    img_cb_blue = []    # 蓝色激光投射到相应的标定板的图片
    img_cb_red = []     # 红色激光投射到相应的标定板的图片

    images = glob.glob(image_path)
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for item in images:
        basename = os.path.basename(item)
        print("basename", basename)
        # 根据文件名区分标定板图像和激光图像
        if basename.startswith("1.png") or \
                basename.startswith("4.png") or \
                basename.startswith("7.png") or \
                basename.startswith("10.png") or \
                basename.startswith("13.png"):
            img_cb_only.append(item)
            print(f"   >>> Calibration Board Image {os.path.basename(item)} has been loaded.")
        elif basename.startswith("2.png") or \
                basename.startswith("5.png") or \
                basename.startswith("8.png") or \
                basename.startswith("11.png") or \
                basename.startswith("14.png"):
            img_cb_blue.append(item)
            print(f"   >>> Blue Laser Projection Image {os.path.basename(item)} has been loaded.")
        else:
            img_cb_red.append(item)
            print(f"   >>> Red Laser Projection  Image {os.path.basename(item)} has been loaded.")



    # 单块标定板物体坐标
    objp        = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)) * circle_distance

    all_points_3d    = []  # 标定板三维点
    all_plane_params = []  # 标定板平面参数
    laser_points     = []  # 多张激光图像的3D线
    drawed_rgb_img   = []

    # 先处理标定板（倾斜）图像
    for fname in img_cb_only:
        print("   >>> Processing Image - ", os.path.basename(fname))
        img = cv2.imread(fname)
        if img is None:
            print(f"      >>> [ERROR] Can't Read Image: {os.path.basename(fname)}")
            continue

        # 去畸变
        img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray[0:DETECTION_TOP,   :] = 0
        gray[DETECTION_BOTTOM:, :] = 0

        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, gray = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

        ret, centers = cv2.findCirclesGrid(
            gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )

        if ret:
            plane_params, points_3d = compute_plane_using_pose(
                centers, camera_matrix, dist_coeffs, pattern_size, circle_distance,
                R_world, t_world
            )
            all_points_3d.append(points_3d)  # 此标定板的3D点
            all_plane_params.append(plane_params)  # 此标定板的平面

            vis_img = img.copy()
            cv2.drawChessboardCorners(vis_img, pattern_size, centers, ret)
            drawed_rgb_img.append(vis_img)
            if LASER_CALIB_BOARD_DETECTED_DISPLAY is True:
                cv2.imshow(f"   >>> [INFO] Detected Circles - {fname}", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            save_path = "temp_results/laser_calibration/" + os.path.basename(fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_img)

    # 确保有平面参数可用
    if len(all_plane_params) == 0:
        print("      >>> [ERROR] 没有标定板的平面参数，无法处理激光图像。")
        return all_points_3d, all_plane_params, laser_points, None

    # 只处理与标定板平面对应的激光图像
    if laser_color == 'red':
        img_cb_blue = img_cb_red
    for fname, plane_params, img_rgb in tqdm(zip(img_cb_blue, all_plane_params, drawed_rgb_img),
                                    desc="Processing data_source images"):
        img = cv2.imread(fname)
        if img is None:
            print(f"      >>> [ERROR] 无法读取图像: {fname}")
            continue

        # 去畸变
        img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)

        # 提取绿色通道（假设激光线在绿色通道中最明显）
        g = None
        if laser_color == 'blue':
            _, _, g = cv2.split(img)
        elif laser_color == 'red':
            g, _, _ = cv2.split(img)

        # 拟合激光线(2D像素)
        m, b = laser_line_fit(fname, g, img_rgb,
                              region_top=top_bound,
                              region_bottom=bottom_bound,
                              region_left=left_bound,
                              region_right=right_bound,
                              th=threshold)

        # 将激光线从 2D 投影到 3D (与对应标定板平面求交)
        a, b_plane, c, d = plane_params

        laser_3d_points = []
        for u in range(left_bound, right_bound):
            v = int(m * u + b)

            if v < 0 or v >= img.shape[0]:
                continue  # 越界

            pixel_hom = np.array([u, v, 1], dtype=np.float32)
            ray_dir_cam = np.linalg.inv(camera_matrix) @ pixel_hom
            ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)  # 单位化

            # 将射线方向转换到世界坐标系（实际与相机坐标系一致）
            ray_dir_world = R_world @ ray_dir_cam  # R_world 是单位矩阵

            # 相机中心在世界坐标系中的位置（t_world 是零向量）
            C_world = t_world

            denom = a * ray_dir_world[0] + b_plane * ray_dir_world[1] + c * ray_dir_world[2]
            if abs(denom) < 1e-6:
                continue  # 射线与平面平行

            t = -(a * C_world[0] + b_plane * C_world[1] + c * C_world[2] + d) / denom
            if t < 0:
                continue  # 交点在射线反方向

            X = C_world[0] + t * ray_dir_world[0]
            Y = C_world[1] + t * ray_dir_world[1]
            Z = C_world[2] + t * ray_dir_world[2]
            laser_3d_points.append([X, Y, Z])

        if len(laser_3d_points) > 0:
            laser_points.append(np.array(laser_3d_points))

    # 最后：拟合所有激光线(3D) -> 激光平面
    if len(laser_points) == 0:
        print("   >>> [ERROR] 未检测到任何激光线点，无法进行激光平面拟合！")
        # 返回基础数据，但光平面用 None 表示
        return all_points_3d, all_plane_params, laser_points, None
    else:
        all_laser_3d = np.vstack(laser_points)
        X = all_laser_3d[:, 0]
        Y = all_laser_3d[:, 1]
        Z = all_laser_3d[:, 2]

        # 拟合光平面 a x + b y + c z + d = 0, 固定 c=1 => z + a x + b y + d = 0
        A = np.column_stack((X, Y, np.ones_like(X)))
        rhs = -Z
        params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        a, b, d = params
        c = 1.0
        laser_plane_params = (a, b, c, d)

        print("\n   >>> [ 激光平面拟合结果 ]")
        print("         A = %4.6f" % a)
        print("         B = %4.6f" % b)
        print("         C = %4.6f" % c)
        print("         D = %4.6f" % d)
        print("      Plane Equation: %4.2f x + %4.2f y + %4.2f z + %4.2f = 0" % (a, b, c, d))

        return all_points_3d, all_plane_params, laser_points, laser_plane_params


def laser_line_fit(fname, image, img_rgb, region_top, region_bottom, region_left, region_right, th):
    """
    输入灰度图像、阈值, 输出线性方程 (y = m*x + b) 的 m, b
    """
    img_viz = image.copy()
    img_non = np.zeros_like(img_viz)
    img_viz = cv2.merge([img_viz, img_non, img_non])

    image[:region_top,          :] = 0
    image[region_bottom:,       :] = 0
    image[:,  :region_left] = 0
    image[:, region_right:] = 0

    # 初步过滤低亮度
    image = cv2.GaussianBlur(image, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 7)
    _, binary = cv2.threshold(image, th, 255, cv2.THRESH_BINARY)


    if LASER_LINES_THRESH_DISPLAY is True:
        # 创建一个可调整大小的窗口
        cv2.namedWindow(f"bin", cv2.WINDOW_NORMAL)

        # 调整窗口大小 (宽度, 高度)
        cv2.resizeWindow(f"bin", DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        cv2.imshow("bin", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 逐列检测
    detected_points = []

    # 逐列检测
    for u in range(region_left, region_right):
        col_data = image[:, u]
        max_val = col_data.max()
        if max_val < th:
            continue

        rows = np.where(col_data == max_val)[0]
        if len(rows) == 0:
            continue

        # 使用高斯滤波平滑亮度曲线
        smoothed = gaussian_filter1d(col_data.astype(float), sigma=1)

        # 找到平滑后亮度的最大值及其位置
        peak_idx = np.argmax(smoothed)

        # 亚像素定位
        if 1 <= peak_idx < len(smoothed) - 1:
            alpha = smoothed[peak_idx - 1]
            beta = smoothed[peak_idx]
            gamma = smoothed[peak_idx + 1]
            denominator = alpha - 2 * beta + gamma
            if denominator != 0:
                p = 0.5 * (alpha - gamma) / denominator
                v_subpixel = peak_idx + p
            else:
                v_subpixel = peak_idx
        else:
            v_subpixel = peak_idx

        v = v_subpixel  # 保留为浮点数进行后续计算

        # 标记红色（可视化调试用）
        if 0 <= v < image.shape[0]:
            img_viz[int(v), u] = (0, 0, 255)  # 红色
            detected_points.append([u, v])

    # 将检测到的点拟合为直线 y = m*x + b
    if len(detected_points) < 2:
        print("   >>> [ERROR] 未检测到足够的激光点, 无法拟合直线.")
        return 0, 0, detected_points

    # 使用最小二乘法拟合
    frame_pts_camera = np.array(detected_points)
    x_coords = frame_pts_camera[:, 0]
    y_coords = frame_pts_camera[:, 1]

    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    # 简单可视化
    # line_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.addWeighted(img_rgb, 0.5, img_viz, 1, 0)
    for x_c in range(region_left, region_right):
        y_c = int(m * x_c + b)
        if 0 <= y_c < image.shape[0]:
            # cv2.circle(img_rgb, (x_c, y_c), 1, (0, 0, 255), -1)
            img_rgb[y_c, x_c] = [0, 255, 0]  # OpenCV使用BGR通道，红色为 (0, 0, 255)

    if LASER_LINES_FITTED_DISPLAY is True:
        # 创建一个可调整大小的窗口
        cv2.namedWindow("Fitted Line", cv2.WINDOW_NORMAL)

        # 调整窗口大小 (宽度, 高度)
        cv2.resizeWindow("Fitted Line", DISPLAY_SIZE[0], DISPLAY_SIZE[1])

        cv2.imshow("Fitted Line", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    save_path = "temp_results/line_fitting/" + os.path.basename(fname)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_rgb)

    return m, b


# ------------------------------
# 5. 可视化标定板平面及误差计算
# ------------------------------
def visualize_world_plane_and_compute_error(points_3d):
    """
    可视化世界坐标系的x-y平面，并计算点云在z轴方向的平均偏差和RMSE。

    参数：
        points_3d (np.ndarray): 生成的3D点云 (N, 3)
    """
    if points_3d is None or len(points_3d) == 0:
        print("没有有效的3D点用于可视化和误差计算。")
        return

    # 计算z轴方向的偏差
    z_coords = points_3d[:, 2]
    mean_z = np.mean(z_coords)
    rmse_z = np.sqrt(np.mean((z_coords - mean_z) ** 2))
    mape_z = np.mean(np.abs((z_coords - mean_z) / mean_z)) * 100 if mean_z != 0 else None

    print(f"Z轴平均偏差 (Mean Z): {mean_z:.6f} mm")
    print(f"Z轴RMSE: {rmse_z:.6f} mm")
    if mape_z is not None:
        print(f"Z轴MAPE: {mape_z:.2f}%")
    else:
        print("Z轴MAPE无法计算 (Mean Z 为 0)。")

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color([0.0, 0.5, 1.0])  # 蓝色点云

    # 创建x-y平面
    plane_size = 500  # 根据需要调整平面大小
    plane = create_world_plane(size=plane_size)
    plane.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色平面

    # 创建坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # 可视化点云、x-y平面和坐标轴
    o3d.visualization.draw_geometries([pcd, plane, coordinate_frame],
                                      window_name="3D Laser Points with World Plane and Axes",
                                      width=800, height=600)


def create_world_plane(size=500):
    """
    创建世界坐标系的x-y平面。

    参数：
        size (float): 平面大小，平面从 (-size, -size) 到 (size, size)
    返回：
        plane_mesh (open3d.geometry.TriangleMesh): 平面网格
    """
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_vertices = [
        [-size, -size, 0],
        [size, -size, 0],
        [size, size, 0],
        [-size, size, 0]
    ]
    plane_triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)
    plane_mesh.compute_vertex_normals()
    return plane_mesh


def build_plane_mesh(normal_vec, d, ref_points, color, size_expand=20):
    """
    根据法向量 normal_vec, 常数项 d, 以及参考点云 ref_points 的范围，
    构造一个 Plane Mesh (Open3D TriangleMesh) 并涂色。
    注：平面方程: normal_vec · X + d = 0
    """
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_vertices = []
    plane_triangles = []

    x_min, x_max = ref_points[:, 0].min() - size_expand, ref_points[:, 0].max() + size_expand
    y_min, y_max = ref_points[:, 1].min() - size_expand, ref_points[:, 1].max() + size_expand

    a, b, c = normal_vec

    # 网格步数可调
    Nx = 10
    Ny = 10
    Xs = np.linspace(x_min, x_max, Nx)
    Ys = np.linspace(y_min, y_max, Ny)
    for xv in Xs:
        for yv in Ys:
            # a*x + b*y + c*z + d = 0 => z = -(a*x + b*y + d)/c
            if abs(c) < 1e-9:
                z = 0.0
            else:
                z = -(a * xv + b * yv + d) / c
            plane_vertices.append([xv, yv, z])

    for rr in range(Nx - 1):
        for cc in range(Ny - 1):
            idx0 = rr * Ny + cc
            idx1 = rr * Ny + cc + 1
            idx2 = (rr + 1) * Ny + cc
            idx3 = (rr + 1) * Ny + cc + 1
            plane_triangles.append([idx0, idx1, idx2])
            plane_triangles.append([idx1, idx3, idx2])

    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)
    plane_mesh.paint_uniform_color(color)
    return plane_mesh


def build_plane_mesh_abcd(a, b, c, d, ref_points, color, size=20):
    """
    给定平面方程 a x + b y + c z + d = 0，
    根据 ref_points 的范围(或固定 size) 生成可视化网格。
    """
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_vertices = []
    plane_triangles = []

    if ref_points is not None and len(ref_points) > 0:
        x_min, x_max = ref_points[:, 0].min() - size, ref_points[:, 0].max() + size
        y_min, y_max = ref_points[:, 1].min() - size, ref_points[:, 1].max() + size
    else:
        # 如果没有参考点，则范围固定
        x_min, x_max = -size, size
        y_min, y_max = -size, size

    Nx = 10
    Ny = 10
    Xs = np.linspace(x_min, x_max, Nx)
    Ys = np.linspace(y_min, y_max, Ny)
    for xv in Xs:
        for yv in Ys:
            denom = c
            if abs(denom) < 1e-9:
                z = 0
            else:
                z = -(a * xv + b * yv + d) / c
            plane_vertices.append([xv, yv, z])

    for rr in range(Nx - 1):
        for cc in range(Ny - 1):
            idx0 = rr * Ny + cc
            idx1 = rr * Ny + cc + 1
            idx2 = (rr + 1) * Ny + cc
            idx3 = (rr + 1) * Ny + cc + 1
            plane_triangles.append([idx0, idx1, idx2])
            plane_triangles.append([idx1, idx3, idx2])

    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)
    plane_mesh.paint_uniform_color(color)
    return plane_mesh


# ------------------------------
# 6. 使用Open3D可视化 标定板 & 激光
# ------------------------------
def visualize_calib_and_laser_open3d(
        mtx, dist, rvecs, tvecs, objp,
        all_points_3d_1, all_plane_params_1,  # 标定板的 3D点 & 平面
        laser_points_1, laser_plane_params_1,  # 激光线 & 光平面
        all_points_3d_2,
        all_plane_params_2,
        laser_points_2,
        laser_plane_params_2,
        R_world, t_world,  # 世界坐标系的旋转和平移
        R_camera, t_camera,
):
    """
    综合可视化：
      1) 相机标定得到的多块标定板（投影到世界坐标系）及其平面
      2) compute_plane_from_inclined_board 得到的 all_points_3d、all_plane_params
         (这些是另一组倾斜标定板, 也一起可视化)
      3) 激光线三维点
      4) 激光平面(光平面)
      5) 在原点添加坐标轴
    """
    geometries = []

    # (A) 显示【相机标定阶段】的多张标定板
    # 用 rvecs[i], tvecs[i] 将 objp 映射到相机坐标，再转换到世界坐标
    for i in range(len(rvecs)):
        R_cam, _ = cv2.Rodrigues(rvecs[i])
        board_in_cam = (R_cam @ objp.T).T + tvecs[i].reshape(1, 3)

        # 转换到世界坐标系（R_world 是单位矩阵，t_world 是零向量，无需转换）
        board_in_world = board_in_cam

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(board_in_world)
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)

        if BOARD_POINTS_DISPLAY:
            geometries.append(pcd)

        # 同时绘制该标定板的平面(可选)
        # 计算世界坐标系下的平面参数（与相机坐标系相同）
        normal_world = R_cam[:, 2]
        d_world = -np.dot(normal_world, tvecs[i].squeeze())

        plane_mesh = build_plane_mesh(normal_world, d_world, board_in_world, color)

        if BOARD_PLANES_DISPLAY:
            geometries.append(plane_mesh)

    # (B) 显示【倾斜标定板阶段】的 3D 点 all_points_3d
    for i, pts_3d in enumerate(all_points_3d_1):
        # pts_3d 已经在世界坐标系下（与相机坐标系一致）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        # 对应的平面
        a, b, c, d = all_plane_params_1[i]
        plane_mesh = build_plane_mesh_abcd(a, b, c, d, pts_3d, color)

        if LASER_CALIB_PLANE_DISPLAY:
            geometries.append(plane_mesh)

    for i, pts_3d in enumerate(all_points_3d_2):
        # pts_3d 已经在世界坐标系下（与相机坐标系一致）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        # 对应的平面
        a, b, c, d = all_plane_params_2[i]
        plane_mesh = build_plane_mesh_abcd(a, b, c, d, pts_3d, color)

        if LASER_CALIB_PLANE_DISPLAY:
            geometries.append(plane_mesh)

    # (C) 显示激光线(3D)
    for i, line3d in enumerate(laser_points_1):
        # line3d 已经在世界坐标系下（与相机坐标系一致）
        if len(line3d) < 2:
            continue
        # 将相邻点连成线
        line_indices = [[j, j + 1] for j in range(len(line3d) - 1)]
        line_colors = [[1, 0, 0] for _ in line_indices]  # 红色线

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line3d)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        if LASER_LINES_DISPLAY:
            geometries.append(line_set)

    for i, line3d in enumerate(laser_points_2):
        # line3d 已经在世界坐标系下（与相机坐标系一致）
        if len(line3d) < 2:
            continue
        # 将相邻点连成线
        line_indices = [[j, j + 1] for j in range(len(line3d) - 1)]
        line_colors = [[0, 0, 1] for _ in line_indices]  # 红色线

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line3d)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        if LASER_LINES_DISPLAY:
            geometries.append(line_set)

    # (D) 激光平面(光平面)
    if laser_plane_params_1 is not None:
        a, b, c, d = laser_plane_params_1
        plane_mesh = build_plane_mesh_abcd(a, b, c, d,
                                           None, [0.0, 1.0, 0.0],
                                           size=100)
        if LASER_PLANE_DISPLAY:
            geometries.append(plane_mesh)

    if laser_plane_params_2 is not None:
        a, b, c, d = laser_plane_params_2
        plane_mesh = build_plane_mesh_abcd(a, b, c, d,
                                           None, [0.0, 1.0, 0.0],
                                           size=100)
        if LASER_PLANE_DISPLAY:
            geometries.append(plane_mesh)

    # (E) 添加坐标轴
    # 创建原始相机坐标系（红色）
    camera_frame = create_coordinate_frame(R_camera, t_camera, axis_length=100)
    geometries.append(camera_frame)

    # 创建新的世界坐标系（绿色）
    world_frame = create_coordinate_frame(R_world, t_world, axis_length=85)
    geometries.append(world_frame)

    print("[Open3D] 综合可视化：标定板(相机标定 + 倾斜标定) + 激光线 + 光平面 + 坐标轴")
    o3d.visualization.draw_geometries(geometries)


# ------------------------------
# 7. Pipeline 整合
# ------------------------------

# ============================
# 在这里添加新的导出函数
# ============================
def export_params_to_json(json_path,
                          camera_matrix,
                          dist_coeffs,
                          R_world,
                          t_world,
                          laser_plane_params_1,
                          laser_plane_params_2,
                          # 以下是阈值、ROI、标定板等全局信息，可根据需要修改或精简


                          LASER_BINARY_THRESHOLD_1=LASER_BINARY_THRESHOLD_1,
                          LASER_BINARY_THRESHOLD_2=LASER_BINARY_THRESHOLD_2,

                          RECON_LASER_DETECT_THRESHOLD_1=200,
                          RECON_LASER_DETECT_THRESHOLD_2=200,




                          DETECTION_TOP=0,
                          DETECTION_BOTTOM=2848,
                          LASER_LINE_LEFT_BOUND=100,
                          LASER_LINE_RIGHT_BOUND=600,
                          RECON_LEFT_BOUND=70,
                          RECON_RIGHT_BOUND=770,
                          RECON_TOP_BOUND=0,
                          RECON_BOTTOM_BOUND=2848,

                          CALIB_BOARD_SIZE=CALIB_BOARD_SIZE,
                          CALIB_BOARD_DIST=CALIB_BOARD_DIST
                          ):
    """
    将关键参数导出到 JSON 文件，包括：
      - 相机内参 (camera_matrix, dist_coeffs)
      - 世界坐标系外参 (R_world, t_world)
      - 激光平面参数 (a, b, c, d)
      - 激光检测阈值 (thresholds)
      - 图像ROI (roi)
      - 标定板参数 (calib_board)
    """
    data = {
        "camera_params": {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist()
        },
        "world_extrinsics": {
            "R_world": R_world.tolist(),
            "t_world": t_world.tolist()
        },
        "thresholds": {
            "BINARY_THRESHOLD": BINARY_THRESHOLD,
            "RECON_LASER_DETECT_THRESHOLD_1": LASER_BINARY_THRESHOLD_1,
            "RECON_LASER_DETECT_THRESHOLD_2": LASER_BINARY_THRESHOLD_2
        },
        "roi": {
            "DETECTION_TOP_1":          DETECTION_TOP_LASER_1,
            "DETECTION_BOTTOM_1":       DETECTION_BOTTOM_LASER_1,
            "LASER_LINE_LEFT_BOUND_1":  LASER_LINE_LEFT_BOUND_LASER_1,
            "LASER_LINE_RIGHT_BOUND_1": LASER_LINE_RIGHT_BOUND_LASER_1,

            "DETECTION_TOP_2":          DETECTION_TOP_LASER_2,
            "DETECTION_BOTTOM_2":       DETECTION_BOTTOM_LASER_2,
            "LASER_LINE_LEFT_BOUND_2":  LASER_LINE_LEFT_BOUND_LASER_2,
            "LASER_LINE_RIGHT_BOUND_2": LASER_LINE_RIGHT_BOUND_LASER_2,
        },
        "calib_board": {
            "CALIB_BOARD_SIZE": list(CALIB_BOARD_SIZE),
            "CALIB_BOARD_DIST": CALIB_BOARD_DIST
        },
        "laser_plane_params_1": list(laser_plane_params_1),  # (a_1, b_1, c_1, d_1)
        "laser_plane_params_2": list(laser_plane_params_2)   # (a_2, b_2, c_2, d_2)
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[INFO] 已将参数导出到: {json_path}")



def main():
    ###########################
    ## 1) Camera Calibration ##
    ###########################
    calib_images_path                              = '../data_source/1/*.png'
    camera_matrix, dist_coeffs, rvecs, tvecs, objp = camera_calibration(
        calib_images_path,
        pattern_size    = CALIB_BOARD_SIZE,
        circle_distance = CALIB_BOARD_DIST
    )
    """
        camera_matrix   : 相机内参矩阵 (3x3)
        dist_coeffs     : 畸变系数 (1x5)
        rvecs           : 每张标定图对应的旋转向量列表
        tvecs           : 每张标定图对应的平移向量列表
        objp            : 单块标定板上所有圆心在“物体坐标系”下的 3D 坐标 (pattern_size[0]*pattern_size[1], 3)
    """

    #######################################
    ## 2) 定义统一的世界坐标系为与相机坐标系一致 ##
    #######################################
    """
        1. 使用的坐标系概述
            物体坐标系（Object Coordinate System, OCS）
            相机坐标系（Camera Coordinate System, CCS）
            世界坐标系（World  Coordinate System, WCS）
        2. 各坐标系的定义与用途
            a.物体坐标系（OCS）
                定义：物体坐标系通常与标定板（如棋盘格）相关联。其原点位于标定板的某
                     个定点（如第一个点），x、y轴沿标定板平面方向，z轴垂直于标定板。
                用途：用于定义标定板上圆心或角点的3D位置，是标定过程中的参考坐标系。
            b.相机坐标系（CCS）
                定义：相机坐标系是相机自身的固有坐标系。原点位于相机光心，z轴指向相机
                     前方（与成像方向一致），x轴和y轴定义在相机成像平面内，遵循右手坐标系。
                用途：描述物体在相机视野中的位置和方向。相机坐标系是相机标定的基础，
                     用于将物体坐标系下的点转换到相机坐标系下。
            c.世界坐标系（WCS）
                定义：世界坐标系是一个统一的参考坐标系，用于在多个相机或多个物体之间建
                     立一致的空间参考。在初始定义中，世界坐标系与相机坐标系重合。但在
                     您的代码中，通过标定数据重新定义了世界坐标系，使其 y 轴与标定板
                     的移动方向对齐，z 轴垂直于拟合平面，x 轴通过 y 轴和 z 轴的叉积确定。
                用途：在多个相机系统或多个物体之间建立一致的空间参考，便于后续的三维重
                     建、物体跟踪等任务。
    """

    # 相机坐标系
    R_camera = np.identity(3)
    t_camera = np.zeros(3)

    # 世界坐标系
    R_world, t_world = define_world_coordinate_system(tvecs, rvecs, objp)

    # print("\n[ 相机坐标系 定义 ]")
    # print("旋转矩阵 R_camera (应为单位矩阵):")
    # print(R_camera)
    # print("平移向量 t_camera (应为零向量):")
    # print(t_camera)
    #
    # if not verify_rotation_matrix(R_camera):
    #     print("警告: R_camera 不是一个有效的旋转矩阵！请检查定义方法。")
    #     # 可以选择退出或继续，根据需求
    # else:
    #     print("旋转矩阵 R_camera 已验证为有效。")



    # print("\n[ 新的世界坐标系定义 ]")
    # print("旋转矩阵 R_world:")
    # print(R_world)
    # print("平移向量 t_world:")
    # print(t_world)
    #
    # if not verify_rotation_matrix(R_world):
    #     print("警告: R_world 不是一个有效的旋转矩阵！请检查定义方法。")
    # else:
    #     print("旋转矩阵 R_world 已验证为有效。")

    # 将对象点转换到新的世界坐标系
    # objp 是 (N, 3)，首先减去平移向量，再应用旋转
    # objp_world = (objp - t_world) @ R_world  # 不需要全局转换，点云已经转换在 visualize 函数中

    # 可视化坐标系和标定点
    if WORLD_CAMERA_COORD_DISPLAY is True:
        visualize_coordinate_systems(R_world, t_world, objp, rvecs, tvecs)

    ###########################
    ## 3) 倾斜标定板 + 激光平面标定 ##
    ###########################
    laser_images_path = '../data_source/2/*.png'

    # Laser up
    (all_points_3d_1,
     all_plane_params_1,
     laser_points_1,
     laser_plane_params_1) = compute_plane_from_inclined_board(
        laser_images_path,
        camera_matrix,
        dist_coeffs,
        'red',
        top_bound=DETECTION_TOP_LASER_1,
        bottom_bound=DETECTION_BOTTOM_LASER_1,
        left_bound=LASER_LINE_LEFT_BOUND_LASER_1,
        right_bound=LASER_LINE_RIGHT_BOUND_LASER_1,
        threshold=LASER_BINARY_THRESHOLD_1,
        pattern_size=CALIB_BOARD_SIZE,
        circle_distance=CALIB_BOARD_DIST,
        R_world = R_camera,
        t_world = t_camera
    )
    # Laser down
    (all_points_3d_2,
     all_plane_params_2,
     laser_points_2,
     laser_plane_params_2) = compute_plane_from_inclined_board(
        laser_images_path,
        camera_matrix,
        dist_coeffs,
        'blue',
        top_bound=DETECTION_TOP_LASER_2,
        bottom_bound=DETECTION_BOTTOM_LASER_2,
        left_bound=LASER_LINE_LEFT_BOUND_LASER_2,
        right_bound=LASER_LINE_RIGHT_BOUND_LASER_2,
        threshold=LASER_BINARY_THRESHOLD_2,
        pattern_size=CALIB_BOARD_SIZE,
        circle_distance=CALIB_BOARD_DIST,
        R_world=R_camera,
        t_world=t_camera
    )

    visualize_calib_and_laser_open3d(
        camera_matrix, dist_coeffs, rvecs, tvecs, objp,  # 相机标定结果
        all_points_3d_1,
        all_plane_params_1,
        laser_points_1,
        laser_plane_params_1,
        all_points_3d_2,
        all_plane_params_2,
        laser_points_2,
        laser_plane_params_2,
        R_world, t_world,  # 世界坐标系的旋转和平移
        R_camera, t_camera,
        # reconstructed_lines=all_reconstructed_lines,
    )

    # visualize_calib_and_laser_open3d(
    #     camera_matrix, dist_coeffs, rvecs, tvecs, objp,  # 相机标定结果
    #     all_points_3d_2,
    #     all_plane_params_2,
    #     laser_points_2,
    #     laser_plane_params_2,
    #     R_world, t_world,  # 世界坐标系的旋转和平移
    #     R_camera, t_camera,
    #     # reconstructed_lines=all_reconstructed_lines,
    # )

    # ------------------
    # (6) 导出关键参数
    # ------------------
    # 假设我们想在标定完成后，将这些参数保存成JSON
    json_output_path = "../data_source/system_calibration_result/laser_params_export.json"
    # 若激光平面标定失败返回 None，需要加判定:
    if laser_plane_params_1 is not None and laser_plane_params_2 is not None:
        export_params_to_json(
            json_path=json_output_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            R_world=R_world,
            t_world=t_world,
            laser_plane_params_1=laser_plane_params_1,
            laser_plane_params_2=laser_plane_params_2
        )
    else:
        print("[WARN] 激光平面标定失败，无法导出参数到JSON！")


if __name__ == "__main__":
    main()
