"""
UMI Gripper Width Estimator
#input: fisheye camera image
output: Detects ArUco markers
1.Detects ArUco markers on left/right gripper fingers from a fisheye camera image
2.and estimates gripper opening width in millimeters via two-point calibration.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np


@dataclass
class GripperWidthResult:
    """Per-frame estimation result."""
    width_mm: Optional[float] = None
    width_raw_mm: Optional[float] = None
    width_smooth_mm: Optional[float] = None
    valid: bool = False
    confidence: float = 0.0
    source: str = "invalid"

    tag_left_detected: bool = False
    tag_right_detected: bool = False

    d_current: Optional[float] = None
    d_closed: Optional[float] = None
    d_open: Optional[float] = None

    left_center: Optional[Tuple[float, float]] = None
    right_center: Optional[Tuple[float, float]] = None

    left_corners: Optional[np.ndarray] = None
    right_corners: Optional[np.ndarray] = None

    lost_frame_count: int = 0
    timestamp: float = 0.0
    error_reason: Optional[str] = None


def _get_aruco_dict(name: str):
    """Get ArUco dictionary by name, compatible with multiple OpenCV versions."""
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    if callable(attr):
        return attr()
    return cv2.aruco.getPredefinedDictionary(attr)


def _detect_markers_compat(gray, aruco_dict, params):
    """Detect ArUco markers with API compatible across OpenCV versions."""
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


class GripperWidthEstimator:
    """Estimates gripper opening width from ArUco markers on finger tips.

    Workflow:
        1. Collect closed-gripper frames  -> calibrate_closed(frames)
        2. Collect open-gripper frames     -> calibrate_open(frames)
        3. finalize_calibration()
        4. Per-frame: estimate(frame)
    """

    def __init__(
        self,
        max_width_mm: float = 45.0,
        left_tag_id: int = 0,
        right_tag_id: int = 1,
        aruco_dict_name: str = "DICT_4X4_50",
        camera_matrix: Optional[np.ndarray] = None,
        distortion_coeffs: Optional[np.ndarray] = None,
        use_fisheye: bool = False,
        smoothing_alpha: float = 0.6,
        min_valid_frames: int = 12,
        min_delta_d: float = 20.0,
        max_std_d: float = 3.0,
        max_lost_frames_hold: int = 3,
        max_width_jump_per_frame_mm: float = 10.0,
        width_out_of_range_tolerance_mm: float = 5.0,
    ):
        self.max_width_mm = max_width_mm
        self.left_tag_id = left_tag_id
        self.right_tag_id = right_tag_id
        self.smoothing_alpha = smoothing_alpha
        self.min_valid_frames = min_valid_frames
        self.min_delta_d = min_delta_d
        self.max_std_d = max_std_d
        self.max_lost_frames_hold = max_lost_frames_hold
        self.max_width_jump_per_frame_mm = max_width_jump_per_frame_mm
        self.width_out_of_range_tolerance_mm = width_out_of_range_tolerance_mm

        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.use_fisheye = use_fisheye

        self.aruco_dict = _get_aruco_dict(aruco_dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Relax adaptive threshold for synthetic images
        try:
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 30
            self.aruco_params.adaptiveThreshWinSizeStep = 5
            self.aruco_params.minMarkerPerimeterRate = 0.02
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.05
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            pass

        # Calibration state
        self._closed_distances: List[float] = []
        self._open_distances: List[float] = []
        self.d_closed: Optional[float] = None
        self.d_open: Optional[float] = None
        self.closed_std: Optional[float] = None
        self.open_std: Optional[float] = None
        self.calibration_valid: bool = False
        self.calibration_reason: str = "not calibrated"

        # Runtime state
        self._last_smooth_width: Optional[float] = None
        self._last_valid_width: Optional[float] = None
        self._lost_frame_count: int = 0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Undistort points using fisheye model if configured."""
        if not self.use_fisheye or self.camera_matrix is None or self.distortion_coeffs is None:
            return pts
        pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
        K = self.camera_matrix.astype(np.float64)
        D = self.distortion_coeffs.astype(np.float64)
        undist = cv2.fisheye.undistortPoints(pts_in, K, D, P=K)
        return undist.reshape(-1, 2)

    def detect_markers(self, frame: np.ndarray) -> Dict:
        """Detect ArUco markers and return per-id corners & center.

        Returns:
            dict mapping tag_id -> {"corners": ndarray(4,2), "center": ndarray(2)}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids = _detect_markers_compat(gray, self.aruco_dict, self.aruco_params)

        result: Dict = {}
        if ids is None or len(ids) == 0:
            return result

        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i].reshape(4, 2)
            c = self._undistort_points(c)
            result[int(tag_id)] = {
                "corners": c,
                "center": c.mean(axis=0),
            }
        return result

    def compute_marker_distance(self, frame: np.ndarray):
        """Detect left/right markers and compute pixel distance.

        Returns:
            (d_current, left_center, right_center, left_corners, right_corners, debug_info)
        """
        markers = self.detect_markers(frame)
        left = markers.get(self.left_tag_id)
        right = markers.get(self.right_tag_id)

        left_det = left is not None
        right_det = right is not None

        if left_det and right_det:
            lc = left["center"]
            rc = right["center"]
            d = float(np.linalg.norm(lc - rc))
            return d, tuple(lc), tuple(rc), left["corners"], right["corners"], {
                "tag_left_detected": True,
                "tag_right_detected": True,
            }

        return None, None, None, None, None, {
            "tag_left_detected": left_det,
            "tag_right_detected": right_det,
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_closed(self, frames: List[np.ndarray]):
        """Collect closed-gripper marker distances from multiple frames."""
        self._closed_distances = []
        for f in frames:
            d, *_ = self.compute_marker_distance(f)
            if d is not None:
                self._closed_distances.append(d)

        if len(self._closed_distances) >= 1:
            self.d_closed = float(np.median(self._closed_distances))
            self.closed_std = float(np.std(self._closed_distances))
        return self.d_closed

    def calibrate_open(self, frames: List[np.ndarray]):
        """Collect open-gripper marker distances from multiple frames."""
        self._open_distances = []
        for f in frames:
            d, *_ = self.compute_marker_distance(f)
            if d is not None:
                self._open_distances.append(d)

        if len(self._open_distances) >= 1:
            self.d_open = float(np.median(self._open_distances))
            self.open_std = float(np.std(self._open_distances))
        return self.d_open

    def finalize_calibration(self) -> Tuple[bool, str]:
        """Validate two-point calibration."""
        reasons = []

        if len(self._closed_distances) < self.min_valid_frames:
            reasons.append(
                f"closed valid frames {len(self._closed_distances)} < {self.min_valid_frames}"
            )
        if len(self._open_distances) < self.min_valid_frames:
            reasons.append(
                f"open valid frames {len(self._open_distances)} < {self.min_valid_frames}"
            )

        if self.d_closed is None or self.d_open is None:
            reasons.append("d_closed or d_open is None")
        else:
            delta = self.d_open - self.d_closed
            if delta < self.min_delta_d:
                reasons.append(f"delta_d {delta:.1f} < {self.min_delta_d}")
            if self.d_open <= self.d_closed:
                reasons.append("d_open <= d_closed")

        if self.closed_std is not None and self.closed_std > self.max_std_d:
            reasons.append(f"closed_std {self.closed_std:.2f} > {self.max_std_d}")
        if self.open_std is not None and self.open_std > self.max_std_d:
            reasons.append(f"open_std {self.open_std:.2f} > {self.max_std_d}")

        if reasons:
            self.calibration_valid = False
            self.calibration_reason = "; ".join(reasons)
        else:
            self.calibration_valid = True
            self.calibration_reason = "OK"

        return self.calibration_valid, self.calibration_reason

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, timestamp: Optional[float] = None) -> GripperWidthResult:
        """Estimate gripper width from a single frame."""
        ts = timestamp if timestamp is not None else time.time()

        if not self.calibration_valid:
            return GripperWidthResult(
                timestamp=ts,
                error_reason="calibration invalid",
                d_closed=self.d_closed,
                d_open=self.d_open,
            )

        d, lc, rc, l_corners, r_corners, dbg = self.compute_marker_distance(frame)

        if d is not None:
            self._lost_frame_count = 0
            raw_mm = self.max_width_mm * (d - self.d_closed) / (self.d_open - self.d_closed)
            raw_mm_clipped = float(np.clip(raw_mm, 0, self.max_width_mm))

            if self._last_smooth_width is None:
                smooth = raw_mm_clipped
            else:
                smooth = self.smoothing_alpha * raw_mm_clipped + (1 - self.smoothing_alpha) * self._last_smooth_width

            self._last_smooth_width = smooth
            self._last_valid_width = smooth

            return GripperWidthResult(
                width_mm=raw_mm_clipped,
                width_raw_mm=raw_mm_clipped,
                width_smooth_mm=round(smooth, 4),
                valid=True,
                confidence=1.0,
                source="vision_aruco",
                tag_left_detected=True,
                tag_right_detected=True,
                d_current=d,
                d_closed=self.d_closed,
                d_open=self.d_open,
                left_center=lc,
                right_center=rc,
                left_corners=l_corners,
                right_corners=r_corners,
                lost_frame_count=0,
                timestamp=ts,
            )
        else:
            self._lost_frame_count += 1
            if self._lost_frame_count <= self.max_lost_frames_hold and self._last_valid_width is not None:
                return GripperWidthResult(
                    width_mm=self._last_valid_width,
                    width_raw_mm=None,
                    width_smooth_mm=self._last_valid_width,
                    valid=False,
                    confidence=0.3,
                    source="last_valid_hold",
                    tag_left_detected=dbg.get("tag_left_detected", False),
                    tag_right_detected=dbg.get("tag_right_detected", False),
                    d_current=None,
                    d_closed=self.d_closed,
                    d_open=self.d_open,
                    lost_frame_count=self._lost_frame_count,
                    timestamp=ts,
                    error_reason="marker lost, using last valid",
                )
            else:
                return GripperWidthResult(
                    valid=False,
                    confidence=0.0,
                    source="invalid",
                    tag_left_detected=dbg.get("tag_left_detected", False),
                    tag_right_detected=dbg.get("tag_right_detected", False),
                    d_closed=self.d_closed,
                    d_open=self.d_open,
                    lost_frame_count=self._lost_frame_count,
                    timestamp=ts,
                    error_reason="marker lost too long",
                )

    @staticmethod
    def post_process(results: List[GripperWidthResult]) -> List[GripperWidthResult]:
        """Back-fill occluded frames with linear interpolation after an episode.

        For each contiguous block of invalid frames (valid=False), find the
        last valid frame before and the first valid frame after. Linearly
        interpolate width_smooth_mm between them and mark source as
        "interpolated". If there is no valid frame on either side the block
        is left unchanged.

        This is intended for offline / data-collection post-processing.
        The original estimate() real-time behavior is unaffected.
        """
        n = len(results)
        i = 0
        while i < n:
            if not results[i].valid:
                block_start = i
                while i < n and not results[i].valid:
                    i += 1
                block_end = i  # exclusive, first valid after block

                prev_idx = block_start - 1
                next_idx = block_end if block_end < n else None

                has_prev = prev_idx >= 0 and results[prev_idx].valid
                has_next = next_idx is not None and results[next_idx].valid

                if has_prev and has_next:
                    v0 = results[prev_idx].width_smooth_mm
                    v1 = results[next_idx].width_smooth_mm
                    span = block_end - prev_idx
                    for j in range(block_start, block_end):
                        alpha = (j - prev_idx) / span
                        interp = v0 + alpha * (v1 - v0)
                        results[j].width_smooth_mm = round(interp, 4)
                        results[j].width_mm = round(interp, 4)
                        results[j].confidence = 0.5
                        results[j].source = "interpolated"
                        results[j].error_reason = "back-fill interpolated"
                elif has_prev:
                    v0 = results[prev_idx].width_smooth_mm
                    for j in range(block_start, block_end):
                        results[j].width_smooth_mm = v0
                        results[j].width_mm = v0
                        results[j].confidence = 0.3
                        results[j].source = "last_valid_hold"
            else:
                i += 1
        return results

    def reset_episode(self):
        """Reset per-episode runtime state (keeps calibration)."""
        self._last_smooth_width = None
        self._last_valid_width = None
        self._lost_frame_count = 0
