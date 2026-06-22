"""
夹爪宽度估计 —— MCAP 输入/输出版本（左右双相机）

处理逻辑与 quick_test.py 完全相同（AprilTag + ROI + 两点物理距离标定 + 后处理插值），
只把 I/O 换成 mcap：
  输入：从 MCAP 的 /camera/<cam>/compressed 解码 JPEG 帧（左右两路都处理）
  输出：每路相机写一个 /umi/gripper_width/<cam> topic
        COPY_ORIGINAL=True 时 = 原始所有消息 + 两个 width topic（完整增广数据集）

运行：/Users/limingzhe/anaconda3/envs/x/bin/python3 mcap_gripper_width.py
"""

import os
import json
import base64
import datetime
import numpy as np
import cv2
from mcap.reader import make_reader
from mcap.writer import Writer
from gripper_width_estimator import GripperWidthEstimator

# ============================ 配置区 ============================
MCAP_IN = "/Users/limingzhe/all_github_code/gripper_width_estimator/1_00001.mcap"   # 任务数据(MCAP)
CAMERAS = ["wrist_left", "wrist_right"]   # 要处理的相机（左右都要）

# 每路相机的标定来源（mp4 或 mcap 路径）。像素距离因相机而异，理应各自标定。
# 注：wrist_right 暂复用左相机标定片段——两相机几何近似对称(右任务闭合337≈左标定335)，
#     等有 wrist_right 专用标定片段后替换更准。
CALIB_SOURCES = {
    "wrist_left":  "/Users/limingzhe/all_github_code/gripper_width_estimator/标定片段.mp4",
    "wrist_right": "/Users/limingzhe/all_github_code/gripper_width_estimator/标定片段.mp4",
}

COPY_ORIGINAL = True               # True=原始所有消息 + width topics(完整增广数据集)

# 物理距离两点标定（同一套夹爪硬件，左右通用）
CLOSED_DIST_MM = 37.7
OPEN_DIST_MM = 108.71

# 左右夹爪共用同一套 tag(0/11)，靠 ROI 区分（两相机自身夹爪都在画面底部）
LEFT_TAG_ID = 0
RIGHT_TAG_ID = 11
DETECTION_ROI = (0, 1050, 1200, 1200)

# 检测加速：quad_decimate 降采样(大 tag 下更快且更稳)，nthreads 多线程
APRILTAG_QUAD_DECIMATE = 2.0       # 1.0=不降采样(慢)，2.0≈3.5x，3.0≈8x
APRILTAG_NTHREADS = 4             # 多线程；纯 CPU 上 ≈2~3x

MAX_STD_D = 15.0
STABLE_WIN = 30
STABLE_STD_MAX = 10.0
USE_POST_PROCESS = True
# ================================================================

WIDTH_SCHEMA = {
    "type": "object",
    "properties": {
        "recorded_ns": {"type": "integer"},
        "width_mm": {"type": ["number", "null"]},
        "valid": {"type": "boolean"},
        "confidence": {"type": "number"},
        "source": {"type": "string"},
        "frame_id": {"type": "string"},
    },
}


_CALIB_CACHE = {}   # (calib_src, ltag, rtag, roi, decimate) -> (d_closed, d_open)


def read_camera_frames(mcap_path, cam_topic):
    """从 MCAP 解码某路相机所有帧，返回 [(recorded_ns, frame_bgr), ...]（按时间序）。"""
    out = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, _, message in reader.iter_messages(topics=[cam_topic]):
            d = json.loads(message.data)
            jpg = base64.b64decode(d["data_b64"])
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            out.append((d.get("recorded_ns", message.log_time), frame))
    out.sort(key=lambda x: x[0])
    return out


def read_all_camera_frames(mcap_path, cameras):
    """单次遍历 MCAP，解码所有指定相机的帧，返回 {cam: [(ns, frame), ...]}。"""
    topic_to_cam = {f"/camera/{c}/compressed": c for c in cameras}
    out = {c: [] for c in cameras}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=list(topic_to_cam)):
            cam = topic_to_cam.get(channel.topic)
            if cam is None:
                continue
            d = json.loads(message.data)
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(d["data_b64"]), np.uint8), cv2.IMREAD_COLOR)
            out[cam].append((d.get("recorded_ns", message.log_time), frame))
    for c in out:
        out[c].sort(key=lambda x: x[0])
    return out


def load_mp4_frames(path):
    """从 mp4 读取所有帧（用于 mp4 标定片段）。"""
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


def load_calib_frames(path):
    """标定片段：按扩展名选 mp4 或 mcap 读取（mcap 取同名相机topic不确定时取首个相机topic）。"""
    if path.lower().endswith(".mp4"):
        return load_mp4_frames(path)
    # mcap：取该文件里第一个 camera topic
    frames = []
    with open(path, "rb") as f:
        reader = make_reader(f)
        cam_topics = [c.topic for c in reader.get_summary().channels.values()
                      if c.topic.startswith("/camera/")]
        topic = cam_topics[0] if cam_topics else None
    return [fr for _, fr in read_camera_frames(path, topic)] if topic else frames


def find_stable_calibration_frames(frames, estimator):
    """稳定保持段标定（与 quick_test.py 相同）。"""
    n = len(frames)
    dists = np.full(n, np.nan)
    for i, frame in enumerate(frames):
        d, *_ = estimator.compute_marker_distance(frame)
        if d is not None:
            dists[i] = d
    n_valid = int(np.sum(~np.isnan(dists)))
    print(f"  标定片段 {n} 帧, 双tag检测 {n_valid} ({100*n_valid/max(n,1):.0f}%), "
          f"距离 {np.nanmin(dists):.0f}~{np.nanmax(dists):.0f} px" if n_valid else "  无检测")

    W = STABLE_WIN
    segments = []
    i = 0
    while i <= n - W:
        good = dists[i:i + W][~np.isnan(dists[i:i + W])]
        if len(good) >= W * 0.9 and good.std() < STABLE_STD_MAX:
            segments.append((i, float(np.median(good)), float(good.std())))
        i += 5
    if not segments:
        return [], []
    closed_seg = min(segments, key=lambda s: s[1])
    open_seg = max(segments, key=lambda s: s[1])

    def seg_frames(seg):
        s = seg[0]
        return [frames[j] for j in range(s, s + W) if not np.isnan(dists[j])]

    return seg_frames(closed_seg), seg_frames(open_seg)


def process_camera(cam, task_frames_ns):
    """对一路相机：标定 + 逐帧估计 + 后处理 → 返回 (width_records, stats)。"""
    print(f"\n===== 处理相机 {cam} =====")
    timestamps = [ns for ns, _ in task_frames_ns]
    frames = [fr for _, fr in task_frames_ns]

    est = GripperWidthEstimator(
        closed_width_mm=CLOSED_DIST_MM, open_width_mm=OPEN_DIST_MM,
        left_tag_id=LEFT_TAG_ID, right_tag_id=RIGHT_TAG_ID,
        use_fisheye=False, detector_backend="apriltag", apriltag_family="tag36h11",
        detection_roi=DETECTION_ROI, max_std_d=MAX_STD_D, min_valid_frames=12,
        apriltag_quad_decimate=APRILTAG_QUAD_DECIMATE,
        apriltag_nthreads=APRILTAG_NTHREADS,
    )

    # 标定（同一标定源 + 同 tag + 同 ROI → 复用，避免重复检测标定片段）
    calib_src = CALIB_SOURCES.get(cam)
    if not calib_src:
        print(f"  [WARN] {cam} 无标定来源，跳过")
        return None, None
    key = (calib_src, LEFT_TAG_ID, RIGHT_TAG_ID, DETECTION_ROI,
           APRILTAG_QUAD_DECIMATE)
    if key in _CALIB_CACHE:
        est.d_closed, est.d_open = _CALIB_CACHE[key]
        est.calibration_valid = True
        print(f"  标定: 复用缓存  d_closed={est.d_closed:.1f} d_open={est.d_open:.1f}")
    else:
        print(f"  标定来源: {os.path.basename(calib_src)}")
        cf, of = find_stable_calibration_frames(load_calib_frames(calib_src), est)
        if not cf:
            print(f"  [ERROR] {cam} 找不到稳定标定段，跳过")
            return None, None
        est.calibrate_closed(cf)
        est.calibrate_open(of)
        valid, reason = est.finalize_calibration()
        print(f"  标定: {'✓' if valid else '✗'} ({reason})  d_closed={est.d_closed:.1f} d_open={est.d_open:.1f}")
        if valid:
            _CALIB_CACHE[key] = (est.d_closed, est.d_open)

    # 估计 + 后处理
    est.reset_episode()
    results = [est.estimate(fr, timestamp=ns / 1e9) for ns, fr in zip(timestamps, frames)]
    n_vision = sum(1 for r in results if r.valid)
    if USE_POST_PROCESS:
        results = est.post_process(results)
    n_usable = sum(1 for r in results if r.width_smooth_mm is not None)

    records = [{
        "recorded_ns": int(ns),
        "width_mm": (None if r.width_smooth_mm is None else round(float(r.width_smooth_mm), 4)),
        "valid": bool(r.valid),
        "confidence": float(r.confidence),
        "source": r.source,
        "frame_id": cam,
    } for ns, r in zip(timestamps, results)]

    total = len(frames)
    print(f"  视觉检测率 {n_vision}/{total} ({100*n_vision/total:.1f}%), "
          f"后处理可用 {n_usable}/{total} ({100*n_usable/total:.1f}%)")
    wvals = [r["width_mm"] for r in records if r["width_mm"] is not None]
    if wvals:
        print(f"  width_mm 范围 {min(wvals):.2f}~{max(wvals):.2f} mm")
    return records, {"vision": n_vision, "usable": n_usable, "total": total}


def write_output_mcap(out_path, width_topics, copy_from=None):
    """写输出 MCAP：多个 width topic；可选拷入原始所有消息。
    width_topics: {topic_name: [records]}"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        if copy_from:
            with open(copy_from, "rb") as fin:
                reader = make_reader(fin)
                schema_map, channel_map = {}, {}
                for schema, channel, message in reader.iter_messages():
                    if channel.id not in channel_map:
                        if schema and schema.id not in schema_map:
                            schema_map[schema.id] = writer.register_schema(
                                schema.name, schema.encoding, schema.data)
                        sid = schema_map.get(schema.id, 0) if schema else 0
                        channel_map[channel.id] = writer.register_channel(
                            channel.topic, channel.message_encoding, sid)
                    writer.add_message(
                        channel_map[channel.id], log_time=message.log_time,
                        data=message.data, publish_time=message.publish_time,
                        sequence=message.sequence)

        sid = writer.register_schema(
            name="deepleap.gripper_width.v1", encoding="jsonschema",
            data=json.dumps(WIDTH_SCHEMA).encode())
        for topic, records in width_topics.items():
            cid = writer.register_channel(topic=topic, message_encoding="json", schema_id=sid)
            for seq, rec in enumerate(records):
                writer.add_message(
                    cid, log_time=rec["recorded_ns"], publish_time=rec["recorded_ns"],
                    data=json.dumps(rec).encode(), sequence=seq)
        writer.finish()


def main():
    print(f"任务 MCAP: {MCAP_IN}")
    print(f"处理相机: {CAMERAS}, tags={LEFT_TAG_ID}/{RIGHT_TAG_ID}, ROI={DETECTION_ROI}")

    # 单次遍历读取所有相机帧（避免对大文件多次扫描）
    frames_by_cam = read_all_camera_frames(MCAP_IN, CAMERAS)

    width_topics = {}
    for cam in CAMERAS:
        task = frames_by_cam.get(cam) or []
        if not task:
            print(f"\n[WARN] {cam}: 任务 MCAP 里没有该相机帧，跳过")
            continue
        records, _ = process_camera(cam, task)
        if records:
            width_topics[f"/umi/gripper_width/{cam}"] = records

    if not width_topics:
        print("\n没有任何相机成功处理，退出")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(MCAP_IN))[0]
    if COPY_ORIGINAL:
        out_path = f"outputs/{stem}_with_width_{ts}.mcap"
    else:
        out_path = f"outputs/gripper_width_{stem}_{ts}.mcap"
    write_output_mcap(out_path, width_topics, copy_from=(MCAP_IN if COPY_ORIGINAL else None))

    print(f"\n写出 width topics: {list(width_topics.keys())}")
    print(f"{'（含原始全部消息拷贝）' if COPY_ORIGINAL else ''}")
    print(f"输出 MCAP: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
