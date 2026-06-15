"""
分析视频内容：生成缩略图、检测 ArUco marker、画出像素距离曲线
帮你找到闭合 / 全开 / 运动的帧区间
"""

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import Counter

# ============================ 配置 ============================
DATA_DIR = "/Users/limingzhe/all_github_code/gripper_width_estimator/clean_bowl_00000"
CAMERA = "wrist_right"  # wrist_left 或 wrist_right
OUTPUT_DIR = "/Users/limingzhe/all_github_code/gripper_width_estimator/analysis_output"
SAMPLE_STEP = 2  # 每隔几帧采样一次（1=逐帧，2=隔帧，加速用）
# ==============================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(DATA_DIR, f"{CAMERA}_camera.mp4")
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n=== {CAMERA} ===")
    print(f"  {n_frames} frames, {fps} fps, {w}x{h}")
    print(f"  Duration: {n_frames/fps:.1f}s ({n_frames/fps/60:.1f} min)")

    # ---- 1. 缩略图网格 ----
    print("\n[1/3] Generating contact sheet...")
    n_cols, n_rows = 6, 10
    total_thumbs = n_cols * n_rows
    step = max(1, n_frames // total_thumbs)
    thumb_w, thumb_h = 200, 200
    canvas = np.full((thumb_h * n_rows, thumb_w * n_cols, 3), 200, dtype=np.uint8)

    for idx in range(total_thumbs):
        frame_idx = idx * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        cv2.putText(thumb, f"#{frame_idx}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        row, col = idx // n_cols, idx % n_cols
        canvas[row * thumb_h:(row + 1) * thumb_h,
               col * thumb_w:(col + 1) * thumb_w] = thumb

    contact_path = os.path.join(OUTPUT_DIR, f"{CAMERA}_contact_sheet.jpg")
    cv2.imwrite(contact_path, canvas)
    print(f"  Saved: {contact_path}")
    print(f"  Grid: {n_cols}x{n_rows}, step={step} frames")
    print(f"  Frame range: 0 ~ {total_thumbs * step}")

    # ---- 2. ArUco 扫描 ----
    print("\n[2/3] Scanning for ArUco markers...")

    dicts_to_try = [
        ("DICT_4X4_50 (default)", cv2.aruco.DICT_4X4_50),
        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
        ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
        ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ]

    for dname, denum in dicts_to_try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(denum)
        params = cv2.aruco.DetectorParameters()
        try:
            params.minMarkerPerimeterRate = 0.03
            params.maxMarkerPerimeterRate = 4.0
        except:
            pass
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        detections = []

        for i in range(0, n_frames, SAMPLE_STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue

            id_list = ids.flatten().tolist()
            d = None
            if 0 in id_list and 1 in id_list:
                idx0 = id_list.index(0)
                idx1 = id_list.index(1)
                c0 = corners[idx0].reshape(4, 2).mean(axis=0)
                c1 = corners[idx1].reshape(4, 2).mean(axis=0)
                d = float(np.linalg.norm(c0 - c1))
            detections.append((i, id_list, d))

        total_det = len(detections)
        both_det = sum(1 for _, _, d in detections if d is not None)

        if total_det == 0:
            print(f"  {dname:30s}: 0 detections  ✗")
            continue

        idxs = [d[0] for d in detections]
        all_ids = []
        for _, id_list, _ in detections:
            all_ids.extend(id_list)
        common = Counter(all_ids).most_common(10)
        id_str = ", ".join([f"ID={i}({n}x)" for i, n in common])

        print(f"  {dname:30s}: {total_det:4d} detections  "
              f"both 0&1: {both_det:3d}  "
              f"range: {idxs[0]}~{idxs[-1]}")
        print(f"    IDs: {id_str}")

        if both_det > 0:
            both_frames = [d[0] for d in detections if d[2] is not None]
            distances = [d[2] for d in detections if d[2] is not None]
            print(f"    Distance: {min(distances):.1f} ~ {max(distances):.1f} px")
            print(f"    First both: #{both_frames[0]}")

            # 距离曲线
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(both_frames, distances, 'b.-', markersize=3)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Pixel distance")
            ax.set_title(f"{CAMERA} - {dname}: Marker distance (SAMPLE_STEP={SAMPLE_STEP})")
            ax.grid(True, alpha=0.3)
            plot_path = os.path.join(OUTPUT_DIR, f"{CAMERA}_{dname}_distances.png")
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"    Plot: {plot_path}")

    cap.release()

    # ---- 3. 提示 ----
    print(f"\n[3/3] Next steps")
    print(f"  1. Open {contact_path} to see what your video looks like")
    print(f"  2. If markers are visible, look at the distance plots")
    print(f"  3. Find the frame ranges for closed (d≈min) and open (d≈max)")
    print(f"  4. Then modify 613test.py or quick_test.py with those ranges")
    print(f"\n  If NO markers were detected:")
    print(f"  - The markers may not be present in this camera view")
    print(f"  - Try changing CAMERA = 'wrist_left' and re-run")
    print(f"  - Check if markers are clearly visible (≥60px in image)")
    print(f"  - Check marker printing and lighting")


if __name__ == "__main__":
    main()
