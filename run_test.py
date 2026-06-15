#!/usr/bin/env python3
"""Run gripper width estimation on a synthetic test video and produce results + visualized video + report."""

import argparse
import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from gripper_width_estimator import GripperWidthEstimator


def load_video_frames(video_path: str):
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def draw_visualization(frame, result, gt_width_mm, frame_idx):
    """Draw detection overlay on frame."""
    vis = frame.copy()

    if result.left_corners is not None:
        pts = result.left_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    if result.right_corners is not None:
        pts = result.right_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (255, 0, 0), 2)

    if result.left_center is not None:
        cv2.circle(vis, (int(result.left_center[0]), int(result.left_center[1])), 5, (0, 255, 0), -1)
    if result.right_center is not None:
        cv2.circle(vis, (int(result.right_center[0]), int(result.right_center[1])), 5, (255, 0, 0), -1)

    if result.left_center is not None and result.right_center is not None:
        cv2.line(
            vis,
            (int(result.left_center[0]), int(result.left_center[1])),
            (int(result.right_center[0]), int(result.right_center[1])),
            (0, 255, 255),
            2,
        )

    y0 = 80
    dy = 28
    color_text = (0, 0, 0)

    def put(txt, y):
        cv2.putText(vis, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

    smooth = result.width_smooth_mm
    put(f"width_est: {smooth:.2f} mm" if smooth is not None else "width_est: N/A", y0)
    put(f"width_gt:  {gt_width_mm:.2f} mm", y0 + dy)
    if smooth is not None:
        err = smooth - gt_width_mm
        put(f"error:     {err:+.2f} mm", y0 + 2 * dy)
    else:
        put("error:     N/A", y0 + 2 * dy)
    put(f"valid: {result.valid}", y0 + 3 * dy)
    put(f"confidence: {result.confidence:.2f}", y0 + 4 * dy)
    put(f"source: {result.source}", y0 + 5 * dy)

    if result.d_closed is not None and result.d_open is not None:
        put(f"d_closed: {result.d_closed:.1f}  d_open: {result.d_open:.1f}", y0 + 6 * dy)

    if not result.valid:
        cv2.putText(vis, "INVALID", (vis.shape[1] - 200, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return vis


def generate_report(
    output_dir, df, d_closed, d_open, closed_std, open_std,
    valid_closed, valid_open, cal_valid, cal_reason,
):
    """Generate report.md and error_plot.png."""
    # Valid = detected by vision + interpolated (all have width_smooth_mm)
    has_width = df["width_smooth_mm"].notna()
    vision_mask = has_width & (df["source"] == "vision_aruco")
    interp_mask = has_width & (df["source"] == "interpolated")
    all_usable = has_width & df["source"].isin(["vision_aruco", "interpolated", "last_valid_hold"])

    vision_df = df[vision_mask]
    interp_df = df[interp_mask]
    usable_df = df[all_usable]

    mae_vision = vision_df["error_mm"].abs().mean() if len(vision_df) > 0 else float("nan")
    mae_all = usable_df["error_mm"].abs().mean() if len(usable_df) > 0 else float("nan")
    rmse_all = math.sqrt((usable_df["error_mm"] ** 2).mean()) if len(usable_df) > 0 else float("nan")
    max_err = usable_df["error_mm"].abs().max() if len(usable_df) > 0 else float("nan")
    mae_interp = interp_df["error_mm"].abs().mean() if len(interp_df) > 0 else float("nan")

    total = len(df)
    n_vision = len(vision_df)
    n_interp = len(interp_df)
    n_hold = int((df["source"] == "last_valid_hold").sum())
    n_invalid = int((df["source"] == "invalid").sum())

    passed = cal_valid and mae_all < 2.0

    report = f"""# Gripper Width Estimation — Test Report

## 1. Calibration

| Metric | Value |
|---|---|
| d_closed (px) | {d_closed:.2f} |
| d_open (px) | {d_open:.2f} |
| valid_closed_frames | {valid_closed} |
| valid_open_frames | {valid_open} |
| closed_std (px) | {closed_std:.3f} |
| open_std (px) | {open_std:.3f} |
| calibration_valid | {cal_valid} |
| calibration_reason | {cal_reason} |

## 2. Accuracy (all usable frames: vision + interpolated + hold)

| Metric | Value |
|---|---|
| MAE — all usable (mm) | {mae_all:.4f} |
| MAE — vision only (mm) | {mae_vision:.4f} |
| MAE — interpolated (mm) | {mae_interp:.4f} |
| RMSE (mm) | {rmse_all:.4f} |
| max_abs_error (mm) | {max_err:.4f} |
| usable_frame_ratio | {len(usable_df)}/{total} ({100*len(usable_df)/total:.1f}%) |

## 3. Detection Statistics

| Metric | Value |
|---|---|
| total_frames | {total} |
| vision_aruco | {n_vision} |
| interpolated (post-process) | {n_interp} |
| last_valid_hold | {n_hold} |
| invalid (no data) | {n_invalid} |

## 4. Conclusion

- **Test passed**: {'YES' if passed else 'NO'}
- **MAE**: {mae_all:.4f} mm {'(< 2 mm threshold)' if mae_all < 2 else '(>= 2 mm threshold)'}
- **Post-process interpolation**: {n_interp} occluded frames recovered, MAE {mae_interp:.4f} mm
- **Main error source**: Pixel jitter noise in synthetic data and EMA smoothing lag at turning points.
- **Recommendations for real UMI integration**:
  1. Use `cv2.fisheye.undistortPoints` with calibrated K/D for the fisheye lens.
  2. Run two-point calibration (`close -> open`) at each episode start.
  3. Call `post_process()` after each episode for offline data collection.
  4. Tune `smoothing_alpha` and `max_lost_frames_hold` for real-time performance.
"""
    mae = mae_all
    rmse = rmse_all

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[OK] Report saved to: {report_path}")

    # Error plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["frame_idx"], df["width_gt_mm"], label="Ground Truth", linewidth=1.5)
    smooth_vals = df["width_smooth_mm"].where(df["width_smooth_mm"].notna())
    axes[0].plot(df["frame_idx"], smooth_vals, label="Estimated (smooth)", linewidth=1.5, alpha=0.9)
    axes[0].set_ylabel("Width (mm)")
    axes[0].legend()
    axes[0].set_title("Gripper Width: GT vs Estimated")
    axes[0].grid(True, alpha=0.3)

    err_vals = df["error_mm"].where(df["error_mm"].notna())
    axes[1].plot(df["frame_idx"], err_vals, color="red", linewidth=1.0, label="Error")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Error (mm)")
    axes[1].set_xlabel("Frame")
    axes[1].legend()
    axes[1].set_title(f"Error Curve — MAE={mae:.3f} mm, RMSE={rmse:.3f} mm")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Error plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run gripper width estimation test")
    parser.add_argument("--video", default="outputs/synthetic_gripper_test.mp4")
    parser.add_argument("--ground_truth", default="outputs/ground_truth.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_width_mm", type=float, default=45.0)
    parser.add_argument("--left_tag_id", type=int, default=0)
    parser.add_argument("--right_tag_id", type=int, default=1)
    parser.add_argument("--calib_closed_frames", type=int, default=20)
    parser.add_argument("--smoothing_alpha", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading video: {args.video}")
    frames, fps = load_video_frames(args.video)
    print(f"  Loaded {len(frames)} frames, FPS={fps}")

    gt_df = pd.read_csv(args.ground_truth)
    print(f"Loading ground truth: {args.ground_truth} ({len(gt_df)} rows)")

    estimator = GripperWidthEstimator(
        max_width_mm=args.max_width_mm,
        left_tag_id=args.left_tag_id,
        right_tag_id=args.right_tag_id,
        smoothing_alpha=args.smoothing_alpha,
    )

    # --- Calibration ---
    n_cal = args.calib_closed_frames

    # Select closed frames: gt width < 1 mm
    closed_indices = gt_df.index[gt_df["width_gt_mm"] < 1.0].tolist()
    if len(closed_indices) < n_cal:
        closed_indices = gt_df.nsmallest(n_cal, "width_gt_mm").index.tolist()
    closed_frame_indices = sorted(closed_indices[:n_cal])
    closed_frames = [frames[i] for i in closed_frame_indices]
    print(f"Calibrating closed with {len(closed_frames)} frames (indices {closed_frame_indices[0]}..{closed_frame_indices[-1]}) ...")
    estimator.calibrate_closed(closed_frames)

    # Select open frames: gt width > max - 1 mm
    open_indices = gt_df.index[gt_df["width_gt_mm"] > args.max_width_mm - 1.0].tolist()
    if len(open_indices) < n_cal:
        open_indices = gt_df.nlargest(n_cal, "width_gt_mm").index.tolist()
    open_frame_indices = sorted(open_indices[:n_cal])
    open_frames = [frames[i] for i in open_frame_indices]
    print(f"Calibrating open with {len(open_frames)} frames (indices {open_frame_indices[0]}..{open_frame_indices[-1]}) ...")
    estimator.calibrate_open(open_frames)

    cal_valid, cal_reason = estimator.finalize_calibration()
    print(f"Calibration valid: {cal_valid}  reason: {cal_reason}")
    print(f"  d_closed={estimator.d_closed:.2f}  d_open={estimator.d_open:.2f}")

    # --- Per-frame estimation ---
    estimator.reset_episode()

    all_results = []
    print("Running per-frame estimation ...")
    for i, frame in enumerate(tqdm(frames)):
        gt_row = gt_df.iloc[i]
        ts = gt_row["timestamp"]
        result = estimator.estimate(frame, timestamp=ts)
        all_results.append(result)

    # --- Post-process: back-fill occluded frames via linear interpolation ---
    all_results = estimator.post_process(all_results)
    n_interpolated = sum(1 for r in all_results if r.source == "interpolated")
    print(f"Post-process: {n_interpolated} frames back-filled by interpolation")

    # --- Build CSV rows and visualized video ---
    h, w = frames[0].shape[:2]
    vis_path = os.path.join(args.output_dir, "visualized_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vis_writer = cv2.VideoWriter(vis_path, fourcc, fps, (w, h))

    rows = []
    for i, (frame, result) in enumerate(zip(frames, all_results)):
        gt_w = gt_df.iloc[i]["width_gt_mm"]
        err = (result.width_smooth_mm - gt_w) if result.width_smooth_mm is not None else None

        rows.append({
            "frame_idx": i,
            "timestamp": result.timestamp,
            "width_gt_mm": gt_w,
            "width_raw_mm": result.width_raw_mm,
            "width_smooth_mm": result.width_smooth_mm,
            "valid": result.valid,
            "confidence": result.confidence,
            "source": result.source,
            "d_current": result.d_current,
            "d_closed": result.d_closed,
            "d_open": result.d_open,
            "error_mm": err,
            "lost_frame_count": result.lost_frame_count,
        })

        vis_frame = draw_visualization(frame, result, gt_w, i)
        vis_writer.write(vis_frame)

    vis_writer.release()
    print(f"[OK] Visualized video saved to: {vis_path}")

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Results CSV saved to: {csv_path}")

    # --- Report ---
    generate_report(
        args.output_dir,
        results_df,
        d_closed=estimator.d_closed,
        d_open=estimator.d_open,
        closed_std=estimator.closed_std,
        open_std=estimator.open_std,
        valid_closed=len(estimator._closed_distances),
        valid_open=len(estimator._open_distances),
        cal_valid=cal_valid,
        cal_reason=cal_reason,
    )

    # Summary
    usable_mask = results_df["width_smooth_mm"].notna() & results_df["source"].isin(
        ["vision_aruco", "interpolated", "last_valid_hold"]
    )
    usable_df = results_df[usable_mask]
    n_interp_total = int((results_df["source"] == "interpolated").sum())
    if len(usable_df) > 0:
        mae = usable_df["error_mm"].abs().mean()
        rmse = math.sqrt((usable_df["error_mm"] ** 2).mean())
        print(f"\n===== RESULTS =====")
        print(f"  Usable frames: {len(usable_df)}/{len(results_df)} (interpolated: {n_interp_total})")
        print(f"  MAE:  {mae:.4f} mm")
        print(f"  RMSE: {rmse:.4f} mm")
        print(f"  Test {'PASSED' if mae < 2.0 else 'FAILED'} (threshold: MAE < 2.0 mm)")
    else:
        print("\nNo usable frames — test FAILED")


if __name__ == "__main__":
    main()
