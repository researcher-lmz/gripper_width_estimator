"""
鱼眼相机标定 (ChArUco 板)
=========================

用同一个鱼眼相机录一段 ChArUco 板视频，本脚本算出内参 K 和畸变系数 D，
保存到 fisheye_calib.npz，供 gripper_width_estimator 的 use_fisheye=True 使用。

录制要点：
  - 手持 ChArUco 板在相机前慢慢移动 10~20 秒
  - 覆盖画面各区域：中心、四角、四边（鱼眼边缘畸变最强，必须拍到）
  - 变换角度：正对 / 左右倾 / 上下倾 / 远近
  - 动作慢，避免运动模糊

用法：
  /Users/limingzhe/anaconda3/envs/x/bin/python3 calibrate_fisheye.py
"""

import cv2
import numpy as np
import os

# ============================ 配置区 ============================
BOARD_VIDEO = "umi_dataset_20260616_163156/clean_bowl_00000/wrist_right_camera.mp4"  # ChArUco 标定视频
CALIB_FILE = "fisheye_calib_wrist_right.npz"      # 输出标定文件（标的是 wrist_right 相机）

# ChArUco 板参数（按你的实物填）
SQUARES_X = 14            # 横向方格数
SQUARES_Y = 9             # 纵向方格数
SQUARE_MM = 20.0          # 单个方格边长 (mm)
MARKER_MM = 15.0          # 方格内 ArUco marker 边长 (mm)
                          # 注：K/D 与绝对尺寸无关，此值不影响标定结果，填个差不多的即可

MIN_CORNERS_PER_FRAME = 12   # 每帧至少检测到多少 ChArUco 角点才采用
MAX_CALIB_FRAMES = 60        # 最多用多少帧标定（太多冗余且慢）
FRAME_STEP = 3               # 每隔几帧扫一次
# ================================================================


def detect_dictionary(video_path):
    """自动探测 ChArUco 板用的是哪个 5x5 字典。"""
    candidates = ["DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
                  "DICT_4X4_100", "DICT_4X4_250", "DICT_6X6_250"]
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 取中间几帧测试
    test_idxs = [n // 4, n // 2, 3 * n // 4]
    scores = {}
    for name in candidates:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(d, params)
        total = 0
        for idx in test_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                total += len(ids)
        scores[name] = total
    cap.release()
    best = max(scores, key=scores.get)
    print(f"字典探测结果: {scores}")
    print(f"→ 选用 {best} (检测到最多 marker)")
    if scores[best] == 0:
        raise RuntimeError("所有字典都没检测到 marker，检查视频和板是否匹配")
    return getattr(cv2.aruco, best)


def collect_charuco_corners(video_path, board, detector):
    """逐帧检测 ChArUco 角点，返回 (objpoints, imgpoints, img_size)。"""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_size = None
    all_obj, all_img = [], []
    board_corners = board.getChessboardCorners()  # (Ncorner, 3)

    collected = 0
    for i in range(0, n, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS_PER_FRAME:
            continue

        ids = ch_ids.flatten()
        objp = board_corners[ids].reshape(-1, 1, 3).astype(np.float64)
        imgp = ch_corners.reshape(-1, 1, 2).astype(np.float64)
        all_obj.append(objp)
        all_img.append(imgp)
        collected += 1

    cap.release()
    print(f"扫描 {n} 帧（步长{FRAME_STEP}），采集到 {collected} 帧有效角点")

    # 太多则均匀抽样
    if collected > MAX_CALIB_FRAMES:
        idxs = np.linspace(0, collected - 1, MAX_CALIB_FRAMES).astype(int)
        all_obj = [all_obj[k] for k in idxs]
        all_img = [all_img[k] for k in idxs]
        print(f"均匀抽样到 {len(all_obj)} 帧")

    return all_obj, all_img, img_size


def fisheye_calibrate_robust(obj_points, img_points, img_size):
    """鲁棒鱼眼标定：自动剔除病态帧后重试。"""
    import re
    obj = list(obj_points)
    img = list(img_points)
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    while len(obj) >= 10:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj))]
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                obj, img, img_size, K, D, rvecs, tvecs,
                flags | cv2.fisheye.CALIB_CHECK_COND, criteria)
            return rms, K, D, len(obj)
        except cv2.error as e:
            msg = str(e)
            m = re.search(r"input array (\d+)", msg)
            if m:
                bad = int(m.group(1))
                obj.pop(bad)
                img.pop(bad)
                print(f"  剔除病态帧 #{bad}，剩 {len(obj)} 帧重试")
            else:
                # 无法定位坏帧，去掉 CHECK_COND 强行算一次
                print("  CHECK_COND 失败且无法定位坏帧，去掉该约束重算")
                K = np.zeros((3, 3))
                D = np.zeros((4, 1))
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj))]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj))]
                rms, K, D, _, _ = cv2.fisheye.calibrate(
                    obj, img, img_size, K, D, rvecs, tvecs, flags, criteria)
                return rms, K, D, len(obj)

    raise RuntimeError("有效帧不足 10，无法标定。请重录覆盖更全的板视频")


def main():
    if not os.path.exists(BOARD_VIDEO):
        raise FileNotFoundError(f"找不到标定视频: {BOARD_VIDEO}")

    print("=" * 55)
    print("ChArUco 鱼眼标定")
    print(f"  视频: {BOARD_VIDEO}")
    print(f"  板: {SQUARES_X}x{SQUARES_Y} 方格, 方格 {SQUARE_MM}mm, marker {MARKER_MM}mm")
    print("=" * 55)

    # 1. 自动探测字典
    dict_enum = detect_dictionary(BOARD_VIDEO)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    n_corners = board.getChessboardCorners().shape[0]
    print(f"板内角点总数: {n_corners}")

    # 2. 采集角点
    obj_points, img_points, img_size = collect_charuco_corners(BOARD_VIDEO, board, detector)
    if len(obj_points) < 10:
        raise RuntimeError(f"有效帧仅 {len(obj_points)}，太少。重录覆盖更全的视频")
    print(f"图像尺寸: {img_size}")

    # 3. 鲁棒鱼眼标定
    print("\n开始鱼眼标定...")
    rms, K, D, n_used = fisheye_calibrate_robust(obj_points, img_points, img_size)

    print(f"\n{'='*55}")
    print(f"标定完成！用了 {n_used} 帧")
    print(f"重投影 RMS 误差: {rms:.4f} px  ({'优秀' if rms<1 else '可接受' if rms<2 else '偏大，建议重录'})")
    print(f"\nK (内参矩阵):\n{K}")
    print(f"\nD (畸变系数):\n{D.ravel()}")

    # 4. 保存
    np.savez(CALIB_FILE, K=K, D=D, img_size=np.array(img_size), rms=rms)
    print(f"\n[OK] 已保存到 {CALIB_FILE}")

    # 5. 抽一帧做去畸变对比，存图直观检查
    cap = cv2.VideoCapture(BOARD_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        K2 = K.copy()
        und = cv2.fisheye.undistortImage(frame, K, D, Knew=K2, new_size=img_size)
        compare = np.hstack([frame, und])
        cv2.imwrite("fisheye_undistort_check.jpg", compare)
        print("[OK] 去畸变对比图: fisheye_undistort_check.jpg (左=原图 右=校正后，看直线是否变直)")


if __name__ == "__main__":
    main()
