import cv2
import numpy as np
import vmbpy
import threading
import queue

# ------------------ 全局配置 ------------------ #
PATTERN_SIZE = (6, 6)  # 标定板圆点排列
frame_queue = queue.Queue(maxsize=1)  # 仅存储最新帧
stop_event = threading.Event()  # 线程停止事件

processed_data = {
    'original_frame': None,  # 彩色图，用于显示
    'gray_frame': None,      # 灰度图，用于检测/计算
    'sharpness': None,
    'pts_src': None
}
processed_lock = threading.Lock()


# ------------------ 计算锐度（改用 meanStdDev） ------------------ #
def calculate_sharpness(gray_image, polygon):
    """仅在多边形区域计算 RMS 对比度 (std / mean)，使用 meanStdDev 来加速。"""
    x, y, w, h = cv2.boundingRect(polygon)
    roi_gray = gray_image[y:y + h, x:x + w]

    mask = np.zeros((h, w), dtype=np.uint8)
    local_polygon = polygon - [x, y]  # 映射到 ROI 内
    cv2.fillPoly(mask, [local_polygon], 255)

    # 用 OpenCV 内部函数直接计算均值和方差
    mean_val, std_val = cv2.meanStdDev(roi_gray, mask=mask)
    mean_val = float(mean_val[0])
    std_val = float(std_val[0])

    return std_val / mean_val if mean_val > 1e-6 else 0.0


# ------------------ 标定板检测 (只接受灰度图) ------------------ #
def find_calibration_board(gray_image):
    """检测标定板，返回四个角点 (None 表示没找到)"""
    found, centers = cv2.findCirclesGrid(
        gray_image,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID
    )
    if not found:
        return None

    # 获取四个角点
    pts_src = np.array([
        centers[0, 0],
        centers[PATTERN_SIZE[1] - 1, 0],
        centers[-1, 0],
        centers[(PATTERN_SIZE[0] - 1) * PATTERN_SIZE[1], 0]
    ], dtype=np.float32)

    return pts_src


# ------------------ 处理线程 ------------------ #
def processing_thread_func():
    """后台线程，实时处理最新图像"""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.05)  # 获取最新彩色帧
            # 转成灰度图，只做一次
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts_src = find_calibration_board(gray_frame)
            sharpness = None

            if pts_src is not None:
                # 需要 int32 多边形以便填充
                poly_pts = pts_src.reshape(4, 1, 2).astype(np.int32)
                sharpness = calculate_sharpness(gray_frame, poly_pts)

            # 存储处理结果（只存最新的）
            with processed_lock:
                processed_data['original_frame'] = frame
                processed_data['gray_frame'] = gray_frame
                processed_data['sharpness'] = sharpness
                processed_data['pts_src'] = pts_src

            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"图像处理异常: {e}")


# ------------------ 相机回调 ------------------ #
def frame_handler(cam, stream, frame):
    """相机回调"""
    try:
        # 如果相机输出本身是 RGB 或 RGBA，可以考虑直接请求 MONO8 格式以减小带宽 + 避免转换
        img = cv2.cvtColor(frame.as_numpy_ndarray(), cv2.COLOR_RGB2BGR)
        try:
            frame_queue.put_nowait(img)  # 只保留最新一帧
        except queue.Full:
            pass
    finally:
        stream.queue_frame(frame)


# ------------------ 主程序 ------------------ #
def main():
    cv2.namedWindow("Board View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Board View", 800, 600)

    processing_thread = threading.Thread(target=processing_thread_func, daemon=True)
    processing_thread.start()

    with vmbpy.VmbSystem.get_instance() as vmb:
        cameras = vmb.get_all_cameras()
        if not cameras:
            print("未检测到相机设备")
            return

        with cameras[0] as cam:
            print(f"正在使用相机: {cam.get_id()}")

            # 尝试设置相机的像素格式为灰度 (如果相机支持的话), 例如:
            # try:
            #     cam.PixelFormat.set('Mono8')
            # except:
            #     pass

            cam.start_streaming(handler=frame_handler, buffer_count=3)
            print("实时采集已启动（按Q退出）")

            max_val = 0.0

            try:
                while True:
                    with processed_lock:
                        frame = processed_data['original_frame']
                        sharp = processed_data['sharpness']
                        pts_src = processed_data['pts_src']

                    if frame is not None:
                        display_img = frame.copy()

                        if pts_src is not None:
                            poly_pts = pts_src.reshape(4, 1, 2).astype(np.int32)

                            # 在外部叠加灰度的效果(可选，若不需要可以去掉)
                            # 若想避免再次转灰度，可以直接用处理线程的 gray_frame
                            gray_bgr = cv2.cvtColor(processed_data['gray_frame'], cv2.COLOR_GRAY2BGR)
                            mask = np.zeros(display_img.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [poly_pts], 255)

                            display_img = cv2.addWeighted(display_img, 1.0, gray_bgr, 0.3, 0)
                            display_img[mask == 0] = gray_bgr[mask == 0]

                            # 绿色多边形边框
                            cv2.polylines(display_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

                            # 显示锐度
                            if sharp is not None:
                                if sharp > max_val:
                                    max_val = sharp

                                # 归一化显示，让数值更直观
                                cv2.putText(display_img,
                                            f"Sharpness: {sharp / max_val * 100:.1f}",
                                            (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 2)

                        cv2.imshow("Board View", display_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cam.stop_streaming()

    stop_event.set()
    processing_thread.join()
    cv2.destroyAllWindows()
    print("程序已正常退出")


if __name__ == "__main__":
    main()
