# UMI 夹爪宽度检测 — 移植集成指南

> 面向将本模块并入 UMI 数据采集代码的工程师。

---

## 1. 需要移植的文件

只需要 **一个文件**：

```
gripper_width_estimator.py
```

该文件包含两个公开接口：
- `GripperWidthResult`（dataclass）— 每帧估计结果
- `GripperWidthEstimator`（class）— 核心估计器

依赖：`opencv-contrib-python`、`numpy`。无其他第三方依赖。

其余文件（`generate_synthetic_video.py`、`run_test.py`）只用于测试验证，无需移植。

---

## 2. 本模块解决什么问题

UMI 夹爪没有编码器，无法直接读取开合宽度。夹爪左右手指上各贴一个 ArUco marker（ID 0 和 1），通过夹爪上的鱼眼相机拍摄图像，检测两个 marker 的像素距离，再经过两点标定换算为毫米宽度。

```
           ┌─ marker 0 ─┐         ┌─ marker 1 ─┐
           │  左手指      │ ◄─ d ─► │  右手指      │
           └─────────────┘         └─────────────┘
                         ▲
                    鱼眼相机
```

---

## 3. 核心算法原理

### 3.1 两点标定

每个 episode 开始前采集两组图像：

| 阶段 | 操作 | 采集量 | 得到 |
|---|---|---|---|
| 闭合标定 | 夹爪完全闭合，保持静止 | 20 帧 | `d_closed`（像素距离中位数） |
| 全开标定 | 夹爪完全张开，保持静止 | 20 帧 | `d_open`（像素距离中位数） |

标定建立了像素距离到毫米宽度的线性映射。

### 3.2 宽度计算

```
width_mm = max_width_mm × (d_current - d_closed) / (d_open - d_closed) 只用计算比例，0到1之间
width_mm = clip(width_mm, 0, max_width_mm)
width_mm = max_width_mm × (d_current ) / (d_open ) 只用计算比例，0到1之间
```

- `d_current`：当前帧两个 marker 中心的像素距离
- `max_width_mm`：夹爪物理最大开口，默认 45.0 mm

### 3.3 EMA 平滑

```
width_smooth = alpha × width_current + (1 - alpha) × width_last
```

`alpha=0.6`，平衡响应速度和抗抖动。

### 3.4 检测失败处理（实时）

```
marker 丢失 ≤ 3 帧  →  保持上一帧值，source="last_valid_hold"，valid=False
marker 丢失 > 3 帧  →  输出 None，source="invalid"，valid=False
```

### 3.5 后处理插值（离线）

Episode 结束后调用 `post_process()`，对连续 invalid 帧段做线性插值回填：

```
遮挡前 valid 帧  ──── 线性插值 ────  遮挡后 valid 帧
       v0           →  interp  →          v1
```

插值帧标记为 `source="interpolated"`，`valid=False`（便于训练时区分）。

---

## 4. 鱼眼相机处理

为了实时性，**不做整图去畸变**。只在检测到 marker 角点后，对角点坐标调用 `cv2.fisheye.undistortPoints` 校正，再计算中心和距离。

启用方式：构造时传入 `camera_matrix`、`distortion_coeffs`、`use_fisheye=True`。

不传或 `use_fisheye=False` 则直接使用原图像素坐标。

---

## 5. 集成到 UMI 数采的调用流程

### 5.1 初始化（程序启动时一次）

```python
from gripper_width_estimator import GripperWidthEstimator

estimator = GripperWidthEstimator(
    max_width_mm=45.0,            # 根据实际夹爪量程修改
    left_tag_id=0,                # 左手指 marker ID
    right_tag_id=1,               # 右手指 marker ID
    aruco_dict_name="DICT_4X4_50",
    camera_matrix=K,              # np.ndarray (3,3)，可选
    distortion_coeffs=D,          # np.ndarray (4,1)，可选
    use_fisheye=True,             # 鱼眼镜头设 True
    smoothing_alpha=0.6,
)
```

### 5.2 Episode 标定（每个 episode 开始前）

```python
# ---- 步骤 1：闭合标定 ----
robot.close_gripper()
time.sleep(1.0)                     # 等夹爪稳定
closed_frames = []
for _ in range(20):
    frame = camera.read()           # BGR numpy array
    closed_frames.append(frame)
estimator.calibrate_closed(closed_frames)

# ---- 步骤 2：全开标定 ----
robot.open_gripper()
time.sleep(1.0)
open_frames = []
for _ in range(20):
    frame = camera.read()
    open_frames.append(frame)
estimator.calibrate_open(open_frames)

# ---- 步骤 3：检查标定 ----
valid, reason = estimator.finalize_calibration()
if not valid:
    raise RuntimeError(f"夹爪标定失败: {reason}")
    # 常见失败原因：
    #   - marker 被遮挡/反光检测不到 → 检查 marker 粘贴和光照
    #   - d_open ≈ d_closed → marker 间距变化太小，检查是否真的开合了
    #   - std 太大 → 夹爪标定时不够稳定，等更久再采

# ---- 步骤 4：重置运行状态 ----
estimator.reset_episode()
```

### 5.3 实时采集（每帧）

```python
while collecting:
    frame = camera.read()
    ts = time.time()

    result = estimator.estimate(frame, timestamp=ts)

    # 写入数据集的字段
    record["gripper_width_mm"]    = result.width_smooth_mm   # 平滑后宽度
    record["gripper_width_valid"] = result.valid             # 是否由视觉直接检测
    record["gripper_confidence"]  = result.confidence        # 1.0 / 0.5 / 0.3 / 0.0
    record["gripper_source"]      = result.source            # 来源标识
```

### 5.4 Episode 结束后处理（可选，离线数采推荐）

```python
# episode_results: List[GripperWidthResult]，整个 episode 的逐帧结果
episode_results = estimator.post_process(episode_results)
# 遮挡帧被线性插值回填，source 变为 "interpolated"
```

---

## 6. GripperWidthResult 字段说明

| 字段 | 类型 | 说明 | 训练时建议 |
|---|---|---|---|
| `width_smooth_mm` | float / None | 平滑后宽度（mm），主要使用字段 | 作为 label |
| `width_raw_mm` | float / None | 原始未平滑宽度 | 一般不用 |
| `valid` | bool | True = 当前帧视觉直接检测到两个 marker | **valid=False 的帧建议降权或过滤** |
| `confidence` | float | **1.0=视觉检测, 0.5=插值, 0.3=保持, 0.0=无数据** | 可作为 sample weight |
| `source` | str | `"vision_aruco"` / `"interpolated"` / `"last_valid_hold"` / `"invalid"` | 用于统计和过滤 |
| `d_current` | float / None | 当前帧像素距离 | debug 用 |
| `d_closed` / `d_open` | float | 标定参考值 | 记录到 episode metadata |
| `left_center` / `right_center` | tuple / None | marker 中心像素坐标 | 可视化用 |
| `lost_frame_count` | int | 连续丢失帧数 | 监控用 |
| `timestamp` | float | 时间戳 | 对齐用 |

---

## 7. 关键参数调节指南

| 参数 | 默认值 | 什么时候需要改 |
|---|---|---|
| `max_width_mm` | 45.0 | 换了不同量程的夹爪 |
| `left_tag_id` / `right_tag_id` | 0 / 1 | 换了不同 ID 的 marker |
| `aruco_dict_name` | `DICT_4X4_50` | 换了不同字典的 marker |
| `smoothing_alpha` | 0.6 | 0.8=更灵敏少平滑，0.3=更平滑有延迟 |
| `min_valid_frames` | 12 | 标定帧数少于 20 帧时可调低，但不建议 < 8 |
| `max_lost_frames_hold` | 3 | 遮挡频繁可调高到 5，但太大会让错误值持续太久 |
| `min_delta_d` | 20.0 | 分辨率很低时 marker 间距变化小，可调低 |
| `max_std_d` | 3.0 | 机械振动大的场景可放宽到 5.0 |

---

## 8. Marker 粘贴要求

```
  ┌──────────────────┐
  │    白色留边 ≥5mm   │
  │  ┌────────────┐  │
  │  │            │  │
  │  │  ArUco     │  │
  │  │  marker    │  │
  │  │            │  │
  │  └────────────┘  │
  │                  │
  └──────────────────┘
```

- Marker 打印尺寸建议 15-25 mm，在图像中对应 ≥60 px
- 白色留边不少于 marker 边长的 25%，否则检测率下降
- 使用哑光材质打印，避免反光
- 贴平，不要有气泡或翘边
- 左右 marker ID 不能相同
- 使用 `DICT_4X4_50` 字典（bit 少，小尺寸也能检测）

---

## 9. 标定失败排查

`finalize_calibration()` 返回 `(False, reason)` 时，按 reason 排查：

| reason 关键词 | 原因 | 解决 |
|---|---|---|
| `closed valid frames < 12` | 闭合状态 marker 检测不到 | 检查 marker 是否在画面内、是否被手指挡住、光照是否充足 |
| `open valid frames < 12` | 全开状态 marker 检测不到 | 同上，另外检查全开时 marker 是否超出画面 |
| `delta_d < 20` | 开合前后 marker 像素距离变化太小 | 相机太远或分辨率太低，marker 间距变化不明显 |
| `closed_std > 3` | 闭合标定时数据不稳定 | 夹爪没夹紧就开始采了，或有振动 |
| `open_std > 3` | 全开标定时数据不稳定 | 同上 |
| `d_open <= d_closed` | 开比闭的距离还小 | left/right tag ID 搞反了，或标定顺序弄反了 |

---

## 10. OpenCV 版本兼容

代码兼容 OpenCV 4.7+ 和旧版本：

| 功能 | 新接口 (≥4.7) | 旧接口 fallback |
|---|---|---|
| 检测 | `ArucoDetector.detectMarkers()` | `cv2.aruco.detectMarkers()` |
| 字典 | `getPredefinedDictionary()` | 同 |
| 参数 | `DetectorParameters()` | 同 |

必须安装 `opencv-contrib-python`，不能用 `opencv-python`（缺少 aruco 模块）。

---

## 11. 数据流示意

```
每个 Episode:

  ┌──────────┐     ┌──────────┐     ┌──────────────────┐
  │ 闭合标定  │────►│ 全开标定  │────►│ finalize_calib   │
  │ 20帧     │     │ 20帧     │     │ d_closed, d_open │
  └──────────┘     └──────────┘     └────────┬─────────┘
                                             │
                                             ▼
  ┌────────────────────────────────────────────────────┐
  │  采集循环 (每帧)                                     │
  │                                                    │
  │  frame ──► detect_markers()                        │
  │                │                                   │
  │                ├─ 双 marker 检测成功                 │
  │                │   └─► 计算 d_current               │
  │                │       └─► 线性映射 → raw_mm        │
  │                │           └─► EMA 平滑 → smooth_mm │
  │                │               └─► valid=True       │
  │                │                                   │
  │                └─ 检测失败                           │
  │                    ├─ 丢失≤3帧 → 保持上帧值          │
  │                    └─ 丢失>3帧 → invalid            │
  └────────────────────────────────────────────────────┘
                                             │
                                             ▼
  ┌────────────────────────────────────────────────────┐
  │  后处理 post_process()  (episode 结束后, 可选)       │
  │                                                    │
  │  连续 invalid 帧段 → 线性插值回填                    │
  │  source="interpolated", confidence=0.5             │
  └────────────────────────────────────────────────────┘
```

---

## 12. 性能参考

在合成测试视频上（180 帧, 1280x720, 含 7 帧遮挡）：

| 指标 | 值 |
|---|---|
| 检测速度 | ~690 FPS (i7 CPU, 无 GPU) |
| 视觉帧 MAE | 0.67 mm |
| 稳态 MAE (排除标定过渡) | 0.48 mm |
| 插值帧 MAE | 1.49 mm |
| 可用帧率 | 100%（含插值后） |

实际设备上检测速度取决于图像分辨率和 marker 大小，1280x720 预计 200-400 FPS。

---

## 13. 最小集成示例

```python
"""最小可运行的集成示例，复制即用。"""

import time
import numpy as np
from gripper_width_estimator import GripperWidthEstimator

# -- 初始化 --
estimator = GripperWidthEstimator(max_width_mm=45.0)

# -- 标定（伪代码，替换为真实 robot/camera 调用）--
# closed_frames = [camera.read() for _ in range(20)]   # 闭合状态 20 帧
# open_frames   = [camera.read() for _ in range(20)]   # 全开状态 20 帧
# estimator.calibrate_closed(closed_frames)
# estimator.calibrate_open(open_frames)
# valid, reason = estimator.finalize_calibration()

# -- 逐帧估计 --
# estimator.reset_episode()
# while collecting:
#     frame = camera.read()
#     result = estimator.estimate(frame, timestamp=time.time())
#     print(f"width={result.width_smooth_mm:.1f}mm  valid={result.valid}")

# -- 后处理（可选）--
# all_results = estimator.post_process(all_results)
```

---

## 14. FAQ

**Q: 可以同时跑多个夹爪吗？**
A: 可以。每个夹爪创建独立的 `GripperWidthEstimator` 实例，使用不同的 tag ID。

**Q: 标定必须每个 episode 都做吗？**
A: 建议每次都做。相机安装位置微调、温度变化都会影响像素距离。标定只需 2 秒。

**Q: post_process 是必须的吗？**
A: 不是。实时控制场景不需要。**离线数据采集建议使用**，可以回填遮挡帧数据。

**Q: marker 检测不到怎么排查？**
A: 用 `detect_markers(frame)` 单独调用，看返回了哪些 ID。常见问题：marker 太小（< 40px）、模糊、反光、白色留边不够。

**Q: 精度不够怎么办？**
A: 先检查鱼眼参数是否正确传入。其次检查 marker 打印质量。最后可以降低 `smoothing_alpha`（更平滑但有延迟）。



#  6.15

15.10: 目前使用6.15早上调整完相机重新采集的视频，跑quick_test.py，输出如下：

```
(x) MacBookAir:gripper_width_estimator limingzhe$ python quick_test.py 
Loaded 3613 frames from data_collection_video1/wrist_left_camera.mp4
Frame size: 800x800
Duration: 60.2s

--- 自动搜索标定帧 ---

全视频共 1334 帧同时检测到两个 marker
闭合标定 (最小像素距离 20 帧):
  d ≈ 277.4 ± 12.36 px (std=12.36)
  帧范围: 1345~2631 (分散度 1286 帧)
  前 5 帧: [(2452, 221.0), (1345, 270.5), (2441, 275.4), (2437, 276.4), (2440, 276.5)]

全开标定 (最大像素距离 20 帧):
  d ≈ 586.3 ± 1.68 px (std=1.68)
  帧范围: 1560~2495 (分散度 935 帧)
  前 5 帧: [(2244, 585.0), (2379, 585.3), (2494, 585.4), (2495, 585.4), (2478, 585.8)]

  delta = 308.9 px

--- Calibration ---
  Result: ✗ FAIL (closed_std 12.36 > 3.0)
  d_closed=277.43 px
  d_open  =586.30 px
  delta   =308.87 px

即使选最优帧也标定失败。可能的原因：
  - marker 检测不稳定（鱼眼畸变→需要标定 K,D）
  - 夹爪在标定帧里没有真正完全闭合/打开
  - 光线或角度问题导致检测不稳定

继续进行估计（用不完美的标定值）...

--- Running estimation (press Q to quit) ---
2026-06-15 10:58:34.128 python[6479:76356] +[IMKClient subclass]: chose IMKClient_Modern
2026-06-15 10:58:34.128 python[6479:76356] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Done.
```

放宽条件后，输出成功，输出如下

```
(x) MacBookAir:gripper_width_estimator limingzhe$ python quick_test.py 
Loaded 3613 frames from data_collection_video1/wrist_left_camera.mp4
Frame size: 800x800
Duration: 60.2s

--- 自动搜索标定帧 ---

全视频共 1334 帧同时检测到两个 marker
闭合标定 (最小像素距离 20 帧):
  d ≈ 277.4 ± 12.36 px (std=12.36)
  帧范围: 1345~2631 (分散度 1286 帧)
  前 5 帧: [(2452, 221.0), (1345, 270.5), (2441, 275.4), (2437, 276.4), (2440, 276.5)]

全开标定 (最大像素距离 20 帧):
  d ≈ 586.3 ± 1.68 px (std=1.68)
  帧范围: 1560~2495 (分散度 935 帧)
  前 5 帧: [(2244, 585.0), (2379, 585.3), (2494, 585.4), (2495, 585.4), (2478, 585.8)]

  delta = 308.9 px

--- Calibration ---
  Result: ✓ PASS (OK)
  d_closed=277.43 px
  d_open  =586.30 px
  delta   =308.87 px

--- Running estimation (press Q to quit) ---
Saving visualized video to: outputs/visualized_result.mp4
2026-06-15 15:42:30.675 python[18110:263226] +[IMKClient subclass]: chose IMKClient_Modern
2026-06-15 15:42:30.675 python[18110:263226] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Visualized video saved to: outputs/visualized_result.mp4
Done.
```

分析原因：

1.仍需要鱼眼相机标注得到K、D，进而减小std，marker的抖动

2.改进算法



![image-20260615160454091](/Users/limingzhe/Library/Application Support/typora-user-images/image-20260615160454091.png)

![image-20260615213403067](/Users/limingzhe/Library/Application Support/typora-user-images/image-20260615213403067.png)
