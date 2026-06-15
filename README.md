# UMI Gripper Width Estimator

Vision-based gripper opening width estimation for UMI data collection. Detects ArUco markers on left/right gripper fingers from a (fisheye) camera and computes opening width in millimeters via two-point calibration.

## Install

```bash
pip install -r requirements.txt
```

Requires `opencv-contrib-python` (ArUco module).

## Quick Start

```bash
# 1. Generate synthetic test video
python generate_synthetic_video.py \
  --output outputs/synthetic_gripper_test.mp4 \
  --ground_truth outputs/ground_truth.csv \
  --num_frames 180 --fps 30 --max_width_mm 45

# 2. Run estimation + visualization + report
python run_test.py \
  --video outputs/synthetic_gripper_test.mp4 \
  --ground_truth outputs/ground_truth.csv \
  --output_dir outputs --max_width_mm 45
```

## Outputs

| File | Description |
|---|---|
| `outputs/synthetic_gripper_test.mp4` | Synthetic test video |
| `outputs/ground_truth.csv` | Per-frame ground truth |
| `outputs/results.csv` | Per-frame estimation results |
| `outputs/visualized_result.mp4` | Video with detection overlay |
| `outputs/error_plot.png` | GT vs estimated width + error curve |
| `outputs/report.md` | Full test report with MAE/RMSE |

## Real UMI Integration

Replace synthetic video with live camera frames. Before each episode:

```python
from gripper_width_estimator import GripperWidthEstimator

estimator = GripperWidthEstimator(
    max_width_mm=45.0,
    left_tag_id=0,
    right_tag_id=1,
    aruco_dict_name="DICT_4X4_50",
    camera_matrix=K,           # 3x3 numpy array (optional)
    distortion_coeffs=D,       # 4x1 numpy array (optional)
    use_fisheye=True,          # set True if fisheye lens
    smoothing_alpha=0.6,
)

# 1. Calibrate
robot.close_gripper()
closed_frames = [cam.read() for _ in range(20)]
estimator.calibrate_closed(closed_frames)

robot.open_gripper()
open_frames = [cam.read() for _ in range(20)]
estimator.calibrate_open(open_frames)

valid, reason = estimator.finalize_calibration()
assert valid, f"Calibration failed: {reason}"

# 2. Collect data
estimator.reset_episode()
while collecting:
    frame = cam.read()
    result = estimator.estimate(frame)
    # result.width_smooth_mm  — filtered width in mm
    # result.valid            — True if markers detected this frame
    # result.confidence       — 0.0 / 0.3 / 1.0
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_width_mm` | 45.0 | Physical max gripper opening (mm) |
| `left_tag_id` | 0 | ArUco ID on left finger |
| `right_tag_id` | 1 | ArUco ID on right finger |
| `aruco_dict_name` | DICT_4X4_50 | ArUco dictionary |
| `camera_matrix` | None | 3x3 intrinsic matrix K |
| `distortion_coeffs` | None | Fisheye distortion coeffs D |
| `use_fisheye` | False | Enable fisheye undistortion on marker corners only |
| `smoothing_alpha` | 0.6 | EMA filter alpha (higher = less smoothing) |
| `min_valid_frames` | 12 | Min frames needed per calibration phase |
| `min_delta_d` | 20.0 | Min pixel distance between open/closed |
| `max_std_d` | 3.0 | Max std allowed in calibration distances |
| `max_lost_frames_hold` | 3 | Frames to hold last value on detection loss |

## Notes

- Markers should be at least 60-80 px in the image for reliable detection.
- The fisheye path only undistorts marker corner points (not the full image) for real-time performance.
- Frames with `valid=False` should be filtered from training datasets.
- Calibration is per-episode to account for camera/mount drift.
