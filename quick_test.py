"""
快速测试：不标定鱼眼，自动找到最佳标定帧，然后估计夹爪宽度

用法：
  /Users/limingzhe/anaconda3/envs/x/bin/python3 quick_test.py
"""

import cv2
import numpy as np
from gripper_width_estimator import GripperWidthEstimator

# ============================ 配置区 ============================
# 输出"两 marker 中心物理距离"(mm)：两端实测值锚定（两点线性标定）
#   闭合 = 闭合时两 marker 中心实测距离
#   全开 = 机械最大开合时两 marker 中心实测距离（夹爪已硬件限位在此）
CLOSED_DIST_MM = 37.7
OPEN_DIST_MM = 108.71

LEFT_TAG_ID = 0
RIGHT_TAG_ID = 11                 #如果哪天算法要用到方向/正负号（比如判断夹爪在画面里往左还是往右动），那左右就必须分清。
                                  #但当前只算距离，无所谓。
VIDEO_PATH = "/Users/limingzhe/all_github_code/gripper_width_estimator/dataset_06172023/wrist_left_camera.mp4"   # 要估计的任务视频
# 标定专用片段（闭合保持 + 张开到边缘最大保持）。None = 从 VIDEO_PATH 自标定。
# 任务视频里开口会比标定大，所以标定要用单独控制的片段，避免选错 open 锚点。
CALIB_VIDEO = "/Users/limingzhe/all_github_code/gripper_width_estimator/标定片段.mp4"
FPS = 60                           # 采集帧率
CALIB_N_FRAMES = 20                # 标定用多少帧
OUTPUT_VIDEO = "outputs/visualized_result.mp4"   # 输出视频路径

# 鱼眼去畸变：实验证明有 ROI 后去畸变没必要（精度相当甚至略好），关掉更简单
USE_FISHEYE = False
CALIB_FILE = "fisheye_calib_wrist_right.npz"   # 仅 USE_FISHEYE=True 时用

# ROI：相机刚性装在自己夹爪上，自己的 tag 总在画面底部固定区域；
# 用 ROI 排除另一只夹爪的同 ID(0/11) tag 误检。按你的相机视角调整。
# 格式 (x_min, y_min, x_max, y_max) 原始像素，画面 1200x1200
DETECTION_ROI = (0, 1050, 1200, 1200)

MAX_STD_D = 15.0                   # 稳定段标定阈值（原始像素）
STABLE_WIN = 30                    # 稳定段窗口大小（帧）
STABLE_STD_MAX = 10.0              # 窗口内 std 小于此值才算"稳定保持"

# 后处理：对漏检帧做线性插值回填（离线数采推荐开；实时控制不要开）
USE_POST_PROCESS = True
# ================================================================


def find_stable_calibration_frames(frames, estimator):
    """用"稳定保持段"标定，而不是全局最小/最大帧。

    全局最小/最大会选到偶发误检和边缘放大帧（鱼眼边缘亚像素误差被放大）。
    稳定段 = 连续 STABLE_WIN 帧且 std 很小 = 夹爪被特意保持不动的标定姿态，
    天然排除偶发/边缘帧。最闭合稳定段→d_closed，最张开稳定段→d_open。
    """
    n = len(frames)
    dists = np.full(n, np.nan)
    for i, frame in enumerate(frames):
        d, *_ = estimator.compute_marker_distance(frame)
        if d is not None:
            dists[i] = d

    n_valid = int(np.sum(~np.isnan(dists)))
    print(f"\n全视频 {n} 帧, 双tag检测 {n_valid} 帧 ({100*n_valid/n:.1f}%)")
    print(f"距离范围: {np.nanmin(dists):.0f}~{np.nanmax(dists):.0f} px")

    # 滚动窗口找稳定段
    W = STABLE_WIN
    segments = []  # (start, median, std)
    i = 0
    while i <= n - W:
        win = dists[i:i + W]
        good = win[~np.isnan(win)]
        if len(good) >= W * 0.9 and good.std() < STABLE_STD_MAX:
            segments.append((i, float(np.median(good)), float(good.std())))
        i += 5

    if not segments:
        print("[ERROR] 没找到稳定保持段。夹爪标定时要完全闭合/张开各保持 1~2 秒")
        return [], []

    # 最闭合 = median 最小的稳定段；最张开 = median 最大的稳定段
    closed_seg = min(segments, key=lambda s: s[1])
    open_seg = max(segments, key=lambda s: s[1])

    def seg_frames(seg):
        start = seg[0]
        idxs = [j for j in range(start, start + W) if not np.isnan(dists[j])]
        return [frames[j] for j in idxs], idxs

    closed_frames, c_idxs = seg_frames(closed_seg)
    open_frames, o_idxs = seg_frames(open_seg)

    print(f"\n闭合稳定段: 帧 {c_idxs[0]}~{c_idxs[-1]}, d≈{closed_seg[1]:.1f} std={closed_seg[2]:.2f}")
    print(f"全开稳定段: 帧 {o_idxs[0]}~{o_idxs[-1]}, d≈{open_seg[1]:.1f} std={open_seg[2]:.2f}")
    print(f"delta = {open_seg[1]-closed_seg[1]:.1f} px")

    return closed_frames, open_frames


def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    # 1. 加载任务视频
    frames = load_video_frames(VIDEO_PATH)
    print(f"Loaded {len(frames)} frames from {VIDEO_PATH}")
    print(f"Frame size: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"Duration: {len(frames)/FPS:.1f}s")

    # 加载鱼眼标定 K/D
    K, D = None, None
    if USE_FISHEYE:
        import os
        if not os.path.exists(CALIB_FILE):
            raise FileNotFoundError(f"找不到鱼眼标定文件 {CALIB_FILE}，先跑 calibrate_fisheye.py")
        cal = np.load(CALIB_FILE)
        K, D = cal["K"], cal["D"]
        print(f"已加载鱼眼标定 {CALIB_FILE}: fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # 2. 初始化估计器
    estimator = GripperWidthEstimator(
        closed_width_mm=CLOSED_DIST_MM,   # 闭合 marker 距离
        open_width_mm=OPEN_DIST_MM,       # 机械全开 marker 距离
        left_tag_id=LEFT_TAG_ID,
        right_tag_id=RIGHT_TAG_ID,
        use_fisheye=USE_FISHEYE,
        camera_matrix=K,
        distortion_coeffs=D,
        detector_backend="apriltag",   # AprilTag 后端（tag36h11）
        apriltag_family="tag36h11",
        detection_roi=DETECTION_ROI,   # ROI 排除另一夹爪的同 ID tag 误检
        max_std_d=MAX_STD_D,
        min_valid_frames=12,
    )

    # 3. 自动化标定：优先用专用标定片段，否则从任务视频自标定
    print("\n--- 自动搜索标定帧 ---")
    if CALIB_VIDEO:
        calib_frames = load_video_frames(CALIB_VIDEO)
        print(f"从专用标定片段标定: {CALIB_VIDEO} ({len(calib_frames)} 帧)")
    else:
        calib_frames = frames
        print("从任务视频自标定（注意：任务里开口若超过标定姿态会外推）")
    closed_frames, open_frames = find_stable_calibration_frames(calib_frames, estimator)

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

    # 准备输出视频（先写临时文件，结束后按"时间戳+检测率"重命名，避免覆盖）
    import os
    import datetime
    out_dir = os.path.dirname(OUTPUT_VIDEO) or "."
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(out_dir, f"visualized_{timestamp}.tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    out_writer = cv2.VideoWriter(tmp_path, fourcc, FPS, (w, h))
    print(f"Saving visualized video (timestamp={timestamp}) ...")

    # ---- Pass 1: 逐帧估计，收集结果 ----
    estimator.reset_episode()
    results = [estimator.estimate(frame, timestamp=i / FPS)
               for i, frame in enumerate(frames)]
    n_vision = sum(1 for r in results if r.valid)

    # ---- 后处理：对漏检帧线性插值回填 ----
    if USE_POST_PROCESS:
        results = estimator.post_process(results)
        n_interp = sum(1 for r in results if r.source == "interpolated")
        n_hold = sum(1 for r in results if r.source == "last_valid_hold")
        print(f"后处理: 视觉检测 {n_vision} 帧, 插值补回 {n_interp} 帧, 边界保持 {n_hold} 帧")
    else:
        n_interp = n_hold = 0

    # 可用帧 = 有 width 的帧（视觉 + 插值 + 保持）
    n_usable = sum(1 for r in results if r.width_smooth_mm is not None)

    # ---- Pass 2: 画可视化视频 ----
    # 可视化用的"原始坐标"检测器：开鱼眼后 result 里存的是校正后坐标，
    # 画在原始畸变图上会错位，所以单独做一次原始检测来画框（只为显示，不参与计算）
    from pupil_apriltags import Detector as _ATDet
    raw_det = _ATDet(families="tag36h11", nthreads=1, quad_decimate=1.0,
                     quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    def _in_roi(ctr):
        if DETECTION_ROI is None:
            return True
        x0, y0, x1, y1 = DETECTION_ROI
        return x0 <= ctr[0] <= x1 and y0 <= ctr[1] <= y1

    for i, frame in enumerate(frames):
        result = results[i]

        vis = frame.copy()

        # 画 ROI 框（青色虚线区域），让人看清有效检测区
        if DETECTION_ROI is not None:
            x0, y0, x1, y1 = DETECTION_ROI
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)

        # 用原始检测画框，但套上 ROI + 同 ID 取 margin 最高（和估计器一致）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = {}
        for d in raw_det.detect(gray):
            if d.tag_id not in (LEFT_TAG_ID, RIGHT_TAG_ID):
                continue
            ctr = np.asarray(d.corners).mean(axis=0)
            if not _in_roi(ctr):
                continue
            if d.tag_id not in best or d.decision_margin > best[d.tag_id].decision_margin:
                best[d.tag_id] = d
        raw_centers = {}
        for tid, d in best.items():
            c = np.asarray(d.corners, dtype=np.int32).reshape(-1, 1, 2)
            ctr = np.asarray(d.corners).mean(axis=0)
            raw_centers[tid] = ctr
            col = (0, 255, 0) if tid == LEFT_TAG_ID else (255, 0, 0)
            cv2.polylines(vis, [c], True, col, 2)
            cv2.circle(vis, (int(ctr[0]), int(ctr[1])), 5, col, -1)
        if LEFT_TAG_ID in raw_centers and RIGHT_TAG_ID in raw_centers:
            lc, rc = raw_centers[LEFT_TAG_ID], raw_centers[RIGHT_TAG_ID]
            cv2.line(vis, (int(lc[0]), int(lc[1])), (int(rc[0]), int(rc[1])),
                     (0, 255, 255), 2)

        # 显示数据：绿=视觉检测, 黄=插值补回, 红=无数据
        if result.source == "vision_aruco":
            color = (0, 255, 0)
        elif result.source == "interpolated":
            color = (0, 255, 255)
        elif result.source == "last_valid_hold":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
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

    # 按检测率重命名（文件名含时间戳 + 视觉检测率，每次运行都是新文件）
    total = len(frames)
    vis_rate = 100.0 * n_vision / total if total else 0.0
    usable_rate = 100.0 * n_usable / total if total else 0.0
    final_path = os.path.join(out_dir, f"visualized_{timestamp}_det{vis_rate:.1f}pct.mp4")
    os.replace(tmp_path, final_path)
    print(f"视觉检测率: {n_vision}/{total} = {vis_rate:.1f}%")
    if USE_POST_PROCESS:
        print(f"后处理后可用率: {n_usable}/{total} = {usable_rate:.1f}% (插值 {n_interp} + 保持 {n_hold})")
    print(f"Visualized video saved to: {final_path}")
    print("Done.")


if __name__ == "__main__":
    main()
