"""
生成用于打印的 AprilTag 标记 (tag36h11)
贴到夹爪左右手指上：ID=0 贴左手指，ID=1 贴右手指

用法:
  /Users/limingzhe/anaconda3/envs/x/bin/python3 generate_apriltag_markers.py
"""

import cv2
import numpy as np
import os

# ============================ 配置 ============================
OUTPUT_DIR = "apriltag_markers"
MARKER_SIZE_PX = 600          # 输出图像分辨率（越大打印越清晰）
WHITE_BORDER_RATIO = 0.25     # 白边占 marker 边长的比例（≥25%，检测必需）
TAG_IDS = [0, 1]              # 左=0, 右=1
# ==============================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # OpenCV 内置 AprilTag 字典，用于生成图像（检测时用 pupil-apriltags）
    apriltag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    border = int(MARKER_SIZE_PX * WHITE_BORDER_RATIO)

    for tag_id in TAG_IDS:
        # 生成 marker 本体
        marker = cv2.aruco.generateImageMarker(apriltag_dict, tag_id, MARKER_SIZE_PX)

        # 加白边（AprilTag 检测必须有足够白边）
        marker_bordered = cv2.copyMakeBorder(
            marker, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255
        )

        # 加文字标注（打印后方便区分左右，注意贴的时候文字朝外不影响检测）
        canvas = cv2.cvtColor(marker_bordered, cv2.COLOR_GRAY2BGR)
        label = f"tag36h11  ID={tag_id}  ({'LEFT' if tag_id == 0 else 'RIGHT'})"
        cv2.putText(canvas, label, (10, border - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, f"apriltag_36h11_id{tag_id}.png")
        cv2.imwrite(out_path, canvas)
        print(f"[OK] Saved: {out_path}  ({canvas.shape[1]}x{canvas.shape[0]} px)")

    print(f"\n生成完成，保存在 {OUTPUT_DIR}/")
    print("打印要求：")
    print("  - 哑光纸打印，避免反光")
    print("  - 打印尺寸建议 15-25 mm（图像中需 ≥60 px）")
    print("  - 白边保留完整，不要裁掉")
    print("  - ID=0 贴左手指，ID=1 贴右手指")
    print("  - 贴平整，无气泡翘边")


if __name__ == "__main__":
    main()
