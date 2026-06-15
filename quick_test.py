"""
快速测试：不标定鱼眼，自动找到最佳标定帧，然后估计夹爪宽度

用法：
  /Users/limingzhe/anaconda3/envs/x/bin/python3 quick_test.py
"""

import cv2
import numpy as np
from gripper_width_estimator import GripperWidthEstimator

# ============================ 配置区 ============================
MAX_WIDTH_MM = 81                  # 夹爪最大物理开口 (mm)
LEFT_TAG_ID = 0
RIGHT_TAG_ID = 1
VIDEO_PATH = "data_collection_video1/wrist_left_camera.mp4"
FPS = 60                           # 采集帧率
CALIB_N_FRAMES = 20                # 标定用多少帧
OUTPUT_VIDEO = "outputs/visualized_result.mp4"   # 输出视频路径
# ================================================================


def find_best_calibration_frames(frames, estimator):
    """从全视频中找到最闭合和最张开的有两个 marker 的帧"""
    all_both = []  # [(frame_idx, distance)]

    for i, frame in enumerate(frames):
        d, *_ = estimator.compute_marker_distance(frame)
        if d is not None:
            all_both.append((i, d))

    all_both.sort(key=lambda x: x[1])  # 按距离从小到大

    n = len(all_both)
    if n < CALIB_N_FRAMES * 2:
        print(f"[ERROR] Only {n} frames with both markers, need at least {CALIB_N_FRAMES*2}")
        return [], []

    closed_list = all_both[:CALIB_N_FRAMES]        # 距离最小的 N 帧
    open_list = all_both[-CALIB_N_FRAMES:]          # 距离最大的 N 帧

    # 统计
    closed_ds = [d for _, d in closed_list]
    open_ds = [d for _, d in open_list]
    closed_median = np.median(closed_ds)
    open_median = np.median(open_ds)
    closed_std = np.std(closed_ds)
    open_std = np.std(open_ds)

    print(f"\n全视频共 {n} 帧同时检测到两个 marker")
    print(f"闭合标定 (最小像素距离 {CALIB_N_FRAMES} 帧):")
    print(f"  d ≈ {closed_median:.1f} ± {closed_std:.2f} px (std={closed_std:.2f})")
    closed_frames_idxs = sorted([f for f, _ in closed_list])
    print(f"  帧范围: {closed_frames_idxs[0]}~{closed_frames_idxs[-1]} (分散度 {closed_frames_idxs[-1]-closed_frames_idxs[0]} 帧)")
    print(f"  前 5 帧: {[(f, round(d,1)) for f,d in closed_list[:5]]}")

    print(f"\n全开标定 (最大像素距离 {CALIB_N_FRAMES} 帧):")
    print(f"  d ≈ {open_median:.1f} ± {open_std:.2f} px (std={open_std:.2f})")
    open_frames_idxs = sorted([f for f, _ in open_list])
    print(f"  帧范围: {open_frames_idxs[0]}~{open_frames_idxs[-1]} (分散度 {open_frames_idxs[-1]-open_frames_idxs[0]} 帧)")
    print(f"  前 5 帧: {[(f, round(d,1)) for f,d in open_list[:5]]}")

    print(f"\n  delta = {open_median - closed_median:.1f} px")

    # 提取实际帧
    closed_frames = [frames[f] for f, _ in closed_list]
    open_frames = [frames[f] for f, _ in open_list]

    return closed_frames, open_frames


def main():
    # 1. 加载视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {VIDEO_PATH}")
    print(f"Frame size: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"Duration: {len(frames)/FPS:.1f}s")

    # 2. 初始化估计器
    estimator = GripperWidthEstimator(
        max_width_mm=MAX_WIDTH_MM,
        left_tag_id=LEFT_TAG_ID,
        right_tag_id=RIGHT_TAG_ID,
        use_fisheye=False,
        detector_backend="apriltag",   # 用 AprilTag 检测器（需重新打印 tag36h11）
        apriltag_family="tag36h11",
        max_std_d=3.0,
        min_valid_frames=12,
    )

    # 3. 自动化标定：从全视频找最佳帧
    print("\n--- 自动搜索标定帧 ---")
    closed_frames, open_frames = find_best_calibration_frames(frames, estimator)

    if not closed_frames:
        print("标定帧不足，退出")
        return

    print(f"\n--- Calibration ---")
    estimator.calibrate_closed(closed_frames)
    estimator.calibrate_open(open_frames)

    valid, reason = estimator.finalize_calibration()
    print(f"  Result: {'✓ PASS' if valid else '✗ FAIL'} ({reason})")
    print(f"  d_closed={estimator.d_closed:.2f} px")
    print(f"  d_open  ={estimator.d_open:.2f} px")
    print(f"  delta   ={estimator.d_open - estimator.d_closed:.2f} px")

    if not valid:
        print("\n即使选最优帧也标定失败。可能的原因：")
        print("  - marker 检测不稳定（鱼眼畸变→需要标定 K,D）")
        print("  - 夹爪在标定帧里没有真正完全闭合/打开")
        print("  - 光线或角度问题导致检测不稳定")
        print("\n继续进行估计（用不完美的标定值）...")

    # 4. 逐帧估计 + 可视化
    print("\n--- Running estimation (press Q to quit) ---")

    # 准备输出视频
    import os
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
    print(f"Saving visualized video to: {OUTPUT_VIDEO}")

    estimator.reset_episode()

    for i, frame in enumerate(frames):
        result = estimator.estimate(frame, timestamp=i / FPS)

        vis = frame.copy()

        # 画 marker 边框和中心
        if result.left_corners is not None:
            pts = result.left_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.circle(vis, (int(result.left_center[0]), int(result.left_center[1])),
                       5, (0, 255, 0), -1)
        if result.right_corners is not None:
            pts = result.right_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
            cv2.circle(vis, (int(result.right_center[0]), int(result.right_center[1])),
                       5, (255, 0, 0), -1)
        if result.left_center is not None and result.right_center is not None:
            cv2.line(vis,
                     (int(result.left_center[0]), int(result.left_center[1])),
                     (int(result.right_center[0]), int(result.right_center[1])),
                     (0, 255, 255), 2)

        # 显示数据
        color = (0, 255, 0) if result.valid else (0, 0, 255)
        width_text = f"{result.width_smooth_mm:.2f} mm" if result.width_smooth_mm is not None else "N/A"
        cv2.putText(vis, f"Frame: {i}/{len(frames)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"Width: {width_text}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, f"Source: {result.source}  Conf: {result.confidence:.1f}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if result.d_current is not None:
            cv2.putText(vis, f"d_current: {result.d_current:.1f} px",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Quick Test - Gripper Width Estimation", vis)
        out_writer.write(vis)                # 写入输出视频

        key = cv2.waitKey(17)     # 60 fps → ~17ms/frame
        if key == ord('q'):
            break

    out_writer.release()
    cv2.destroyAllWindows()
    print(f"Visualized video saved to: {OUTPUT_VIDEO}")
    print("Done.")


if __name__ == "__main__":
    main()
