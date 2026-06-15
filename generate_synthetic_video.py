#!/usr/bin/env python3
"""Generate a synthetic gripper open/close video with ArUco markers for testing."""

import argparse
import os
import math

import cv2
import numpy as np
import pandas as pd


def generate_aruco_marker(marker_id: int, size_px: int, dict_name: str = "DICT_4X4_50") -> np.ndarray:
    """Generate an ArUco marker image."""
    dict_enum = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)
    try:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    except AttributeError:
        img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_px)
    return img


def paste_marker(canvas, marker_gray, cx, cy, angle_deg, scale=1.0):
    """Paste a marker image onto canvas at (cx, cy) with rotation and scale."""
    h, w = marker_gray.shape[:2]
    sh, sw = int(h * scale), int(w * scale)
    if sh < 4 or sw < 4:
        return
    marker_resized = cv2.resize(marker_gray, (sw, sh), interpolation=cv2.INTER_AREA)

    marker_bgr = cv2.cvtColor(marker_resized, cv2.COLOR_GRAY2BGR)

    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(sh * sin_a + sw * cos_a)
    new_h = int(sh * cos_a + sw * sin_a)
    M[0, 2] += (new_w - sw) / 2
    M[1, 2] += (new_h - sh) / 2

    rotated = cv2.warpAffine(marker_bgr, M, (new_w, new_h), borderValue=(200, 200, 200))

    x1 = int(cx - new_w / 2)
    y1 = int(cy - new_h / 2)
    x2 = x1 + new_w
    y2 = y1 + new_h

    ch, cw = canvas.shape[:2]
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(cw, x2)
    dst_y2 = min(ch, y2)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    roi = rotated[src_y1:src_y2, src_x1:src_x2]
    mask = np.all(roi != [200, 200, 200], axis=-1).astype(np.uint8)

    # Add white border around marker for better detection
    canvas_roi = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
    for c_idx in range(3):
        canvas_roi[:, :, c_idx] = canvas_roi[:, :, c_idx] * (1 - mask) + roi[:, :, c_idx] * mask


def draw_finger(canvas, cx, cy, finger_w, finger_h, color):
    """Draw a simplified gripper finger rectangle."""
    x1 = int(cx - finger_w // 2)
    y1 = int(cy - finger_h // 2)
    x2 = x1 + finger_w
    y2 = y1 + finger_h
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 2)


def generate_video(
    output_path: str,
    gt_path: str,
    num_frames: int = 180,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    max_width_mm: float = 45.0,
    marker_size_px: int = 80,
    occlude_frames: list = None,
):
    """Generate synthetic gripper video with ground truth CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    marker_left = generate_aruco_marker(0, marker_size_px)
    marker_right = generate_aruco_marker(1, marker_size_px)

    # Add white border around markers for reliable detection
    border = 20
    marker_left = cv2.copyMakeBorder(marker_left, border, border, border, border,
                                     cv2.BORDER_CONSTANT, value=255)
    marker_right = cv2.copyMakeBorder(marker_right, border, border, border, border,
                                      cv2.BORDER_CONSTANT, value=255)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    if occlude_frames is None:
        occlude_frames = list(range(num_frames - 25, num_frames - 18))

    center_y = height // 2
    d_closed_px = 140.0
    d_open_px = 500.0

    # Video phases: closed calibration, open calibration, motion
    n_cal_closed = 25
    n_cal_open = 25
    n_motion = num_frames - n_cal_closed - n_cal_open

    rng = np.random.RandomState(42)
    gt_rows = []

    for i in range(num_frames):
        if i < n_cal_closed:
            ratio = 0.0
        elif i < n_cal_closed + n_cal_open:
            ratio = 1.0
        else:
            mi = i - n_cal_closed - n_cal_open
            t = mi / max(n_motion - 1, 1)
            if t <= 0.5:
                ratio = t * 2
            else:
                ratio = 2 * (1 - t)

        gt_width_mm = ratio * max_width_mm
        d_current = d_closed_px + ratio * (d_open_px - d_closed_px)

        jitter = rng.normal(0, 0.5, 2)
        left_cx = width / 2 - d_current / 2 + jitter[0]
        right_cx = width / 2 + d_current / 2 + jitter[1]

        angle_noise_l = rng.normal(0, 1.5)
        angle_noise_r = rng.normal(0, 1.5)
        brightness_offset = rng.randint(-8, 9)

        bg_val = 200 + brightness_offset
        bg_val = max(150, min(240, bg_val))
        canvas = np.full((height, width, 3), bg_val, dtype=np.uint8)

        # Draw fingers
        finger_w = 50
        finger_h = 250
        draw_finger(canvas, int(left_cx), center_y, finger_w, finger_h, (160, 160, 170))
        draw_finger(canvas, int(right_cx), center_y, finger_w, finger_h, (160, 160, 170))

        # Paste markers
        occluded = i in occlude_frames
        paste_marker(canvas, marker_left, left_cx, center_y, angle_noise_l)
        if not occluded:
            paste_marker(canvas, marker_right, right_cx, center_y, angle_noise_r)
        else:
            # Draw occluder over right marker
            cv2.rectangle(
                canvas,
                (int(right_cx - 60), center_y - 60),
                (int(right_cx + 60), center_y + 60),
                (50, 50, 50),
                -1,
            )

        # Light Gaussian blur for realism
        if rng.random() < 0.3:
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        # HUD info
        cv2.putText(canvas, f"Frame {i}  GT: {gt_width_mm:.1f} mm", (20, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        writer.write(canvas)

        gt_rows.append({
            "frame_idx": i,
            "timestamp": i / fps,
            "width_gt_mm": round(gt_width_mm, 4),
            "left_center_gt_x": round(left_cx, 2),
            "left_center_gt_y": round(center_y, 2),
            "right_center_gt_x": round(right_cx, 2),
            "right_center_gt_y": round(center_y, 2),
            "occluded": int(occluded),
        })

    writer.release()
    df = pd.DataFrame(gt_rows)
    df.to_csv(gt_path, index=False)
    print(f"[OK] Synthetic video saved to: {output_path}")
    print(f"[OK] Ground truth CSV saved to: {gt_path}")
    print(f"     Frames: {num_frames}, FPS: {fps}, Occluded frames: {occlude_frames}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gripper test video")
    parser.add_argument("--output", default="outputs/synthetic_gripper_test.mp4")
    parser.add_argument("--ground_truth", default="outputs/ground_truth.csv")
    parser.add_argument("--num_frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_width_mm", type=float, default=45.0)
    parser.add_argument("--marker_size_px", type=int, default=80)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    generate_video(
        output_path=args.output,
        gt_path=args.ground_truth,
        num_frames=args.num_frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_width_mm=args.max_width_mm,
        marker_size_px=args.marker_size_px,
    )


if __name__ == "__main__":
    main()
