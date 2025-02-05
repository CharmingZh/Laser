import os
import time
import cv2
import threading
import queue
import vmbpy

# ------------------ 全局配置 ------------------ #
CONFIG_FILE_PATH = "../data_source/lsr_60_305.xml"

LASER_CONFIG_FILE_PATH = "../data_source/new_laser_recon.xml"
RGB_CONFIG_FILE_PATH = "../data_source/new_rgb_recon.xml"

# ------------------ 全局变量 ------------------ #
latest_frame = None         # 供主线程显示用的最新帧
lock = threading.Lock()      # 保护 latest_frame 的互斥锁

image_index = 0             # 已保存图像的计数
log_file = None             # 用于记录时间戳的文件句柄

# 互斥锁：用于保护 (should_queue_frames, should_save_frames, log_file) 等变量
save_state_lock = threading.Lock()

# 生产者-消费者队列，用于在回调线程和保存线程之间传递图像帧
# 如需更大队列，可修改 maxsize
frame_queue = queue.Queue(maxsize=10240)

# 后台线程退出事件
stop_event = threading.Event()

# ------------------ 新增的两个全局布尔值 ------------------ #
# should_queue_frames: 是否向队列中放入新的图像帧
# should_save_frames : 是否将取出的帧执行保存（写盘 + 写日志）
should_queue_frames = False
should_save_frames  = False


def set_camera_params(cam, exposure=None, gain=None):
    """
    设置相机曝光时间和增益。
    不同机型的 Feature 名称可能不同，请根据实际情况调整。
    """
    # 关闭自动曝光（如果有）
    try:
        expo_auto = cam.get_feature_by_name('ExposureAuto')
        expo_auto.set('Off')
    except vmbpy.VmbFeatureError:
        pass

    # 设置曝光时间（单位通常是微秒）
    if exposure is not None:
        try:
            exp_feature = cam.get_feature_by_name('ExposureTime')
            exp_feature.set(exposure)
            print(f"[INFO] 曝光时间已设置为 {exposure}")
        except Exception as e:
            print(f"[ERROR] 无法设置曝光时间：{e}")

    # 关闭自动增益（如果有）
    try:
        gain_auto = cam.get_feature_by_name('GainAuto')
        gain_auto.set('Off')
    except vmbpy.VmbFeatureError:
        pass

    # 设置增益
    if gain is not None:
        try:
            gain_feature = cam.get_feature_by_name('Gain')
            gain_feature.set(gain)
            print(f"[INFO] 增益已设置为 {gain}")
        except Exception as e:
            print(f"[ERROR] 无法设置增益：{e}")


def saver_thread_func():
    """
    后台线程：不断从 frame_queue 中取出 (frame_data, timestamp)，
    并根据 should_save_frames 写入硬盘和日志文件。
    当 stop_event 被置位或主程序结束时，退出循环。
    """
    global image_index

    while not stop_event.is_set():
        try:
            frame_data, ts = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            with save_state_lock:
                local_should_save = should_save_frames
                local_log_file = log_file

            # 如果还处于“需要保存”的状态，则写盘
            if local_should_save and local_log_file is not None:
                image_index += 1
                filename = f"{image_index}.png"
                save_path = os.path.join("captures", filename)

                cv2.imwrite(save_path, frame_data)
                local_log_file.write(f"{image_index},{ts}\n")
                local_log_file.flush()

            frame_queue.task_done()

        except Exception as e:
            print("[ERROR] 后台保存线程崩溃：", e)
            # 为了不让线程退出，可以继续捕获并打印，但不 break
            pass


def start_saving():
    """
    开始连续保存图像：
    1. 创建 captures 文件夹（如不存在）
    2. 重置计数器
    3. 打开日志文件
    4. 设置 should_queue_frames = True（回调开始放帧）
    5. 设置 should_save_frames  = True（后台线程保存帧）
    """
    global image_index, log_file
    global should_queue_frames, should_save_frames

    with save_state_lock:
        # 如果已经在录制，就不重复开始
        if not should_queue_frames and not should_save_frames:
            if not os.path.exists("captures"):
                os.makedirs("captures")

            image_index = 0
            log_file = open(os.path.join("captures", "timestamps.txt"), "w")

            should_queue_frames = True   # 回调可放入新帧
            should_save_frames  = True   # 后台线程可保存帧

            print("[INFO] 开始保存图像。")


def stop_saving():
    """
    停止连续保存图像：
    1. 先禁止回调向队列里放入新的帧 should_queue_frames = False
    2. 等待队列清空 (frame_queue.join())，把剩余帧都保存完
    3. 再把 should_save_frames = False，关闭日志文件
    """
    global log_file
    global should_queue_frames, should_save_frames

    with save_state_lock:
        # 如果已经是“停止”状态，则不做任何操作
        if not should_queue_frames and not should_save_frames:
            return

        # 让回调停止往队列里放数据
        should_queue_frames = False

    # 等待队列里剩余的数据都被 saver_thread_func() 写盘
    frame_queue.join()

    # 队列清空后，再真正停止保存
    with save_state_lock:
        should_save_frames = False
        if log_file is not None:
            log_file.close()
            log_file = None

    print("[INFO] 停止保存图像。")


def frame_handler(cam, stream, frame):
    """
    相机流回调函数。
    如果 should_queue_frames=True，就把帧放入队列。
    """
    global latest_frame

    try:
        # 提取 numpy 数组
        np_image = frame.as_numpy_ndarray()
        # 如果格式是 RGB8, 转成 BGR 以适配 cv2.imshow 等
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] 转换帧失败: {e}")
    finally:
        # 把 frame 重新放回队列，准备下一帧
        stream.queue_frame(frame)

    # 更新最新帧用于显示（这里不做任何文字叠加，避免影响保存的图像）
    with lock:
        latest_frame = np_image

    # 如果当前在“采集”状态，就尝试把帧放进队列
    with save_state_lock:
        local_should_queue = should_queue_frames

    if local_should_queue:
        timestamp = time.time()
        try:
            # 在这里用 np_image.copy() 取一份拷贝，保证后续不受显示操作影响
            frame_queue.put((np_image.copy(), timestamp), block=False)
        except queue.Full:
            print("[WARNING] 队列已满，丢弃帧。")


def main():
    # 启动后台保存线程（设置为 daemon=True，保证主线程退出时后台线程也能结束）
    saver_thread = threading.Thread(target=saver_thread_func, daemon=True)
    saver_thread.start()

    with vmbpy.VmbSystem.get_instance() as vmb:
        cameras = vmb.get_all_cameras()
        if not cameras:
            print("[ERROR] 未检测到 Allied Vision 相机。")
            return

        with cameras[0] as cam:
            print(f"[INFO] 使用相机: {cam.get_id()}")

            # 加载相机配置
            try:
                cam.load_settings(CONFIG_FILE_PATH, vmbpy.PersistType.All)
                print(f"[INFO] 成功加载相机配置: {CONFIG_FILE_PATH}")
            except vmbpy.VmbError as e:
                print(f"[WARNING] 加载配置文件失败: {e}")
                print("[INFO] 将使用默认相机配置。")

            # 如果需要手动设置曝光和增益，可以在这里调用
            # set_camera_params(cam, exposure=20000, gain=5)

            # 尝试设置像素格式为 Bgr8（如果相机支持）
            try:
                pixel_format_feature = cam.get_feature_by_name('PixelFormat')
                print("[INFO] 相机支持的像素格式：")
                for fmt in pixel_format_feature.get_available_entries():
                    print(f"  - {str(fmt)}")

                pixel_format_feature.set('Bgr8')
                print("[INFO] 成功将相机像素格式设置为 Bgr8。")
            except Exception as e:
                print(f"[ERROR] 无法设置像素格式为 Bgr8：{e}")
                try:
                    current_format = pixel_format_feature.get()
                    print(f"[INFO] 使用当前像素格式：{current_format}")
                except Exception as e2:
                    print(f"[ERROR] 无法获取当前像素格式：{e2}")

            # 启动采集
            cam.start_streaming(handler=frame_handler, buffer_count=5)
            print("[INFO] 开始实时采集图像（异步）。")
            print("  - 按 's' 键开始/停止保存")
            print("  - 按 'ESC' 键退出")

            # ------------------ 新增：实时帧率测量变量 ------------------
            fps_timer_start = time.time()  # 用于计算 FPS 的起始时间
            fps_frame_count = 0           # 记录这一段时间内采集了多少帧
            display_fps = 0.0            # 存储计算出来的 FPS，用于显示

            try:
                while True:
                    # 从全局 latest_frame 取一份拷贝做显示，避免并发问题
                    # （最新帧在回调里已写入 latest_frame）
                    with lock:
                        frame_for_show = None
                        if latest_frame is not None:
                            frame_for_show = latest_frame.copy()

                    # 如果有新帧，用 opencv 显示
                    if frame_for_show is not None:
                        # 帧率统计：累计帧数
                        fps_frame_count += 1
                        elapsed = time.time() - fps_timer_start
                        if elapsed >= 1.0:  # 每秒更新一次 FPS 显示
                            display_fps = fps_frame_count / elapsed
                            fps_frame_count = 0
                            fps_timer_start = time.time()

                        # 在帧上叠加 FPS 信息，不影响原图像的保存
                        cv2.putText(
                            frame_for_show,
                            f"FPS: {display_fps:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2
                        )

                        cv2.imshow("Allied Vision Live", frame_for_show)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        # 切换开始/停止保存
                        if should_queue_frames or should_save_frames:
                            stop_saving()
                        else:
                            start_saving()
                    elif key == 27:  # ESC
                        print("[INFO] 收到退出指令，准备退出程序。")
                        break

            finally:
                cam.stop_streaming()

    # 收尾：退出前把窗口关掉
    cv2.destroyAllWindows()

    # 如果还在录制，先停掉
    if should_queue_frames or should_save_frames:
        stop_saving()

    # 通知后台线程退出
    stop_event.set()
    # 等待后台线程结束
    saver_thread.join()

    print("[INFO] 程序已退出。")


if __name__ == "__main__":
    main()