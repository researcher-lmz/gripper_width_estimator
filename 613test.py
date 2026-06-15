"""
鱼眼相机标定 + 夹爪宽度估计

用法:
    1. 录一段棋盘格视频 (从不同角度拍)，设 CHESSBOARD_VIDEO 路径
    2. CALIB_MODE = True  → 运行标定，生成 fisheye_calib.npz
    3. CALIB_MODE = False → 加载标定结果，跑夹爪宽度估计
"""

import cv2
import numpy as np
import json
from gripper_width_estimator import GripperWidthEstimator

# ============================ 配置区 ============================
MAX_WIDTH_MM = 81          # 夹爪最大物理开口 (mm)
LEFT_TAG_ID = 0
RIGHT_TAG_ID = 1
CALIB_MODE = True           # True=标定相机, False=跑夹爪宽度估计

# 棋盘格参数
CHESSBOARD_SIZE = (9, 6)    # 内角点数 (宽, 高)
SQUARE_SIZE_MM = 25         # 棋盘格每个方块边长 (mm)
CALIB_FILE = "fisheye_calib.npz"

# 输入文件
CHESSBOARD_VIDEO = "chessboard.mp4"   # 棋盘格标定视频
GRIPPER_VIDEO = "data_collection_video1/wrist_left_camera.mp4"  # 夹爪测试视频
GRIPPER_FPS = 60                      # 采集帧率

# 标定帧范围（根据分析结果）
# 全开段 (d ≈ 594 px): 1555~1585
# 闭合段 (d ≈ 323 px): 1952~1982
OPEN_START, OPEN_END = 1555, 1585
CLOSED_START, CLOSED_END = 1952, 1982
# ================================================================


def calibrate_fisheye(video_path, board_size, square_mm):
    """用棋盘格视频标定鱼眼相机，返回 K, D, rvecs, tvecs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open chessboard video: {video_path}")

    # 准备棋盘格角点坐标 (世界坐标系)
    pattern_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    pattern_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    pattern_points *= square_mm

    obj_points = []   # 世界坐标系下的角点 (多帧)
    img_points = []   # 图像坐标系下的角点 (多帧)
    img_size = None
    frame_count = 0
    good_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        # 检测棋盘格角点
        ret_found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret_found:
            continue

        # 亚像素精细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        obj_points.append(pattern_points)
        img_points.append(corners_refined)
        good_count += 1

        # 可视化提示
        vis = cv2.drawChessboardCorners(frame.copy(), board_size, corners_refined, True)
        cv2.putText(vis, f"Good frames: {good_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibration - press Q to stop early", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames, found {good_count} good chessboard frames")

    if good_count < 10:
        raise ValueError(
            f"Only {good_count} good frames. Need at least 10. "
            "Make sure chessboard covers different areas of the image."
        )

    # 鱼眼标定
    print("Running fisheye calibration...")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(good_count)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(good_count)]

    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, img_size, K, D, rvecs, tvecs, flags, criteria
    )

    print(f"Calibration RMS: {retval:.4f}")
    print(f"K (camera matrix):\n{K}")
    print(f"D (distortion coeffs):\n{D}")

    return K, D


# =====================================================================
# 主程序
# =====================================================================

if CALIB_MODE:
    # ---- 模式 A：标定鱼眼相机 ----
    print("=" * 50)
    print("Mode: FISHEYE CALIBRATION")
    print(f"  Chessboard video: {CHESSBOARD_VIDEO}")
    print(f"  Board size: {CHESSBOARD_SIZE}, Square: {SQUARE_SIZE_MM}mm")
    print("=" * 50)

    K, D = calibrate_fisheye(CHESSBOARD_VIDEO, CHESSBOARD_SIZE, SQUARE_SIZE_MM)

    # 保存
    np.savez(CALIB_FILE, K=K, D=D)
    print(f"\n[OK] Calibration saved to: {CALIB_FILE}")

else:
    # ---- 模式 B：夹爪宽度估计 ----
    print("=" * 50)
    print("Mode: GRIPPER WIDTH ESTIMATION")
    print("=" * 50)

    # 加载标定结果
    try:
        data = np.load(CALIB_FILE)
        K = data["K"]
        D = data["D"]
        print(f"Loaded fisheye calibration from {CALIB_FILE}")
        print(f"K:\n{K}")
        print(f"D:\n{D}")
    except FileNotFoundError:
        print(f"[WARN] {CALIB_FILE} not found, running without fisheye correction")
        K, D = None, None

    # 加载夹爪视频
    cap = cv2.VideoCapture(GRIPPER_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open gripper video: {GRIPPER_VIDEO}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {GRIPPER_VIDEO}")

    # 初始化估计器
    estimator = GripperWidthEstimator(
        max_width_mm=MAX_WIDTH_MM,
        left_tag_id=LEFT_TAG_ID,
        right_tag_id=RIGHT_TAG_ID,
        use_fisheye=(K is not None),
        camera_matrix=K,
        distortion_coeffs=D,
    )

    # 标定（预填帧范围，可根据实际情况调整）
    closed_frames = frames[CLOSED_START:CLOSED_END]
    open_frames = frames[OPEN_START:OPEN_END]

    print(f"Calibrating: closed={CLOSED_START}~{CLOSED_END}, open={OPEN_START}~{OPEN_END}")
    estimator.calibrate_closed(closed_frames)
    estimator.calibrate_open(open_frames)
    valid, reason = estimator.finalize_calibration()
    print(f"Calibration: {valid} ({reason})")
    print(f"  d_closed={estimator.d_closed:.2f}  d_open={estimator.d_open:.2f}")

    if not valid:
        print("Calibration failed. Check marker detection and video content.")
        exit(1)

    # 逐帧估计
    estimator.reset_episode()
    for i, frame in enumerate(frames):
        result = estimator.estimate(frame, timestamp=i / GRIPPER_FPS)

        # 可视化
        vis = frame.copy()
        color = (0, 255, 0) if result.valid else (0, 0, 255)
        cv2.putText(vis, f"width: {result.width_smooth_mm:.2f} mm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"valid: {result.valid}  source: {result.source}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"conf: {result.confidence:.1f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 画 marker 中心和连线
        if result.left_center is not None:
            cv2.circle(vis, (int(result.left_center[0]), int(result.left_center[1])),
                       5, (0, 255, 0), -1)
        if result.right_center is not None:
            cv2.circle(vis, (int(result.right_center[0]), int(result.right_center[1])),
                       5, (255, 0, 0), -1)
        if result.left_center is not None and result.right_center is not None:
            cv2.line(vis,
                     (int(result.left_center[0]), int(result.left_center[1])),
                     (int(result.right_center[0]), int(result.right_center[1])),
                     (0, 255, 255), 2)

        cv2.imshow("Gripper Width Estimation", vis)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")
