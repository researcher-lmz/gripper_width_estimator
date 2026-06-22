import os
import re
import datetime
from PIL import Image, ImageDraw


def tag_id_from_path(path):
    """从文件名里提取 tag id，如 tag36_11_00001.png -> 1。
    会先去掉 ' (1)' 这类重复后缀，再取最后一个数字组，避免误取到副本编号。"""
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"\(.*?\)", "", name)   # 去掉括号后缀，如 "(1)"
    nums = re.findall(r"\d+", name)
    return str(int(nums[-1])) if nums else name


# 1. 基础设置 (按 300 DPI 计算像素)
DPI = 300
MM_TO_PX = DPI / 25.4

# A4 纸张竖向尺寸: 210mm x 297mm
A4_WIDTH = int(210 * MM_TO_PX)
A4_HEIGHT = int(297 * MM_TO_PX)

# 目标尺寸
TAG_SIZE = int(12.5 * MM_TO_PX)   # 整张 tag 图边长 12.5mm（原图最外圈自带白边）
MARGIN = int(8 * MM_TO_PX)        # 页面四周留边 8mm（打印机一般无法打到纸边）

# 直接按原图放入单元格（原图本身已带白边 quiet zone），沿单元格边界画裁剪线
CELL = TAG_SIZE

# 裁剪引导线：浅灰细线
LINE_COLOR = (170, 170, 170)
LINE_WIDTH = 1

# 两个 tag 图片：奇数列=id0，偶数列=id1
IMG_ID0 = "/Users/limingzhe/Downloads/tag36_11_00002.png"   # id=0
IMG_ID1 = "/Users/limingzhe/Downloads/tag36_11_00003.png"   # id=1

# 创建白色 A4 画布
canvas = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), 'white')
draw = ImageDraw.Draw(canvas)

try:
    # 2. 读取图片并精确缩放 (使用 NEAREST 保持黑白像素边缘锐利)
    img_id0 = Image.open(IMG_ID0).resize((TAG_SIZE, TAG_SIZE), Image.Resampling.NEAREST)
    img_id1 = Image.open(IMG_ID1).resize((TAG_SIZE, TAG_SIZE), Image.Resampling.NEAREST)

    # 3. 自动计算能填满 A4 的列数 / 行数（按单元格大小，在留边范围内）
    usable_w = A4_WIDTH - 2 * MARGIN
    usable_h = A4_HEIGHT - 2 * MARGIN
    COLS = usable_w // CELL
    ROWS = usable_h // CELL

    # 4. 整个网格在页面居中
    grid_w = COLS * CELL
    grid_h = ROWS * CELL
    start_x = (A4_WIDTH - grid_w) // 2
    start_y = (A4_HEIGHT - grid_h) // 2

    # 5. 逐格粘贴：直接按原图放入单元格
    #    奇数列(第1,3,5...列)=id0，偶数列(第2,4,6...列)=id1
    for row in range(ROWS):
        for col in range(COLS):
            cell_x = start_x + col * CELL
            cell_y = start_y + row * CELL
            img = img_id0 if col % 2 == 0 else img_id1   # 0-indexed 偶下标=奇数列=id0
            canvas.paste(img, (cell_x, cell_y))

    # 6. 画浅色裁剪网格线（单元格边界）——剪下来每张都带白边
    for c in range(COLS + 1):
        x = start_x + c * CELL
        draw.line([(x, start_y), (x, start_y + grid_h)], fill=LINE_COLOR, width=LINE_WIDTH)
    for r in range(ROWS + 1):
        y = start_y + r * CELL
        draw.line([(start_x, y), (start_x + grid_w, y)], fill=LINE_COLOR, width=LINE_WIDTH)

    # 7. 保存为高质量 PDF（文件名含 id 信息 + 时间戳，多次运行不覆盖）
    id0 = tag_id_from_path(IMG_ID0)
    id1 = tag_id_from_path(IMG_ID1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"tags_12_5mm_id{id0}-id{id1}_{COLS}x{ROWS}_{timestamp}.pdf"
    canvas.save(out_name, "PDF", resolution=300.0)
    n0 = sum(1 for c in range(COLS) if c % 2 == 0) * ROWS
    n1 = COLS * ROWS - n0
    print(f"生成成功！{COLS} 列 x {ROWS} 行 = {COLS*ROWS} 个 tag "
          f"(id{id0}={n0}, id{id1}={n1})，每张 {TAG_SIZE/MM_TO_PX:.1f}mm 原图 + 浅灰裁剪线，"
          f"已输出 {out_name}")

except FileNotFoundError as e:
    print(f"找不到图片文件，请检查文件名：{e}")
