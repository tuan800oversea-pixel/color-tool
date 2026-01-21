import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import math
import os
import io

# --- 配置参数 ---
EXCEL_FILE = 'RGB_CMYK.xlsx'
DPI = 300
CM_TO_INCH = 1 / 2.54
BLOCK_SIZE_CM = 4       # 色块 4cm
TEXT_HEIGHT_CM = 1.0    # 预留文字高度
MARGIN_CM = 0.5         # 边距
COLUMNS = 4             # 1行4个

# 计算像素尺寸
BLOCK_PX = int(BLOCK_SIZE_CM * CM_TO_INCH * DPI)      # 472 px
TEXT_H_PX = int(TEXT_HEIGHT_CM * CM_TO_INCH * DPI)    # 约 118 px
MARGIN_PX = int(MARGIN_CM * CM_TO_INCH * DPI)         # 约 59 px

# 优化后的字体大小：55pt 在 472px 宽度内显示 "R:255 G:255 B:255" 最为清晰且不换行
FONT_SIZE = 55 

def load_data_from_excel(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return []
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        data_list = []
        for index, row in df.iterrows():
            if pd.isna(row.get('工厂')): continue
            try:
                item = {
                    "factory": str(row['工厂']).strip(),
                    "rgb": (int(row['RGB_R']), int(row['RGB_G']), int(row['RGB_B'])),
                    "cmyk": (int(row['CMYK_C']), int(row['CMYK_M']), int(row['CMYK_Y']), int(row['CMYK_K']))
                }
                data_list.append(item)
            except (ValueError, KeyError):
                continue
        return data_list
    except Exception as e:
        print(f"读取失败: {e}")
        return []

def get_font(size):
    # 优先使用粗体以增加可读性
    font_names = ["arialbd.ttf", "simhei.ttf", "arial.ttf"]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except:
            continue
    return ImageFont.load_default()

def generate_chart(factory_name, items, mode="RGB"):
    num_items = len(items)
    rows = math.ceil(num_items / COLUMNS)

    # 画布宽度 = (色块宽 * 4) + (间距 * 5)
    canvas_width = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    # 画布高度 = ((色块+文字) * 行数) + (间距 * 行数) + 顶部标题空间
    canvas_height = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 120

    if mode == 'CMYK':
        img_mode = 'CMYK'
        bg_color = (0, 0, 0, 0)      # CMYK 纸张白
        text_color = (0, 0, 0, 255)  # 单黑 K100
    else:
        img_mode = 'RGB'
        bg_color = (255, 255, 255)   # RGB 纯白
        text_color = (0, 0, 0)       # RGB 纯黑

    img = Image.new(img_mode, (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(img)
    font = get_font(FONT_SIZE)
    title_font = get_font(int(FONT_SIZE * 0.8))

    # 绘制标题
    title_text = f"工厂: {factory_name} | 模式: {mode} (300 DPI)"
    draw.text((MARGIN_PX, 40), title_text, fill=text_color, font=title_font)

    for i, item in enumerate(items):
        r_idx = i // COLUMNS
        c_idx = i % COLUMNS

        x = MARGIN_PX + c_idx * (BLOCK_PX + MARGIN_PX)
        y = 120 + r_idx * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)

        # 1. 设置色块填充色
        if mode == "RGB":
            fill_color = item['rgb']
        else:
            c_val, m_val, y_val, k_val = item['cmyk']
            fill_color = (
                int(c_val * 2.55),
                int(m_val * 2.55),
                int(y_val * 2.55),
                int(k_val * 2.55)
            )

        # 绘制色块 (带极细边框防止浅色色块看不见)
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill_color, outline=text_color, width=1)

        # 2. 绘制文字 (单行显示)
        if mode == "RGB":
            label = f"R:{item['rgb'][0]} G:{item['rgb'][1]} B:{item['rgb'][2]}"
        else:
            label = f"C:{item['cmyk'][0]} M:{item['cmyk'][1]} Y:{item['cmyk'][2]} K:{item['cmyk'][3]}"

        # 文字居中处理
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        # 如果文字宽度超过色块，稍微缩小字号
        current_font = font
        if text_w > BLOCK_PX:
            current_font = get_font(int(FONT_SIZE * 0.85))
            bbox = draw.textbbox((0, 0), label, font=current_font)
            text_w = bbox[2] - bbox[0]

        text_x = x + (BLOCK_PX - text_w) / 2
        text_y = y + BLOCK_PX + 15  # 色块下方 15 像素开始写字

        draw.text((text_x, text_y), label, fill=text_color, font=current_font)

    return img

def main():
    print(f"正在处理数据: {EXCEL_FILE}...")
    raw_data = load_data_from_excel(EXCEL_FILE)

    if not raw_data:
        print("未发现有效数据，请检查 Excel 文件内容。")
        return

    factories = {}
    for item in raw_data:
        f_name = item['factory']
        if f_name not in factories: factories[f_name] = []
        factories[f_name].append(item)

    output_dir = "color_charts_output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for fac, items in factories.items():
        print(f"正在生成工厂图稿: {fac}...")
        safe_fac = "".join([c for c in fac if c.isalnum() or c in (' ', '_', '-')]).strip()

        # 生成并保存 RGB 版本
        img_rgb = generate_chart(fac, items, mode="RGB")
        img_rgb.save(f"{output_dir}/{safe_fac}_RGB.tif", compression='tiff_lzw', dpi=(300, 300))

        # 生成并保存 CMYK 版本
        img_cmyk = generate_chart(fac, items, mode="CMYK")
        img_cmyk.save(f"{output_dir}/{safe_fac}_CMYK.tif", compression='tiff_lzw', dpi=(300, 300))

    print(f"\n成功! 所有图片已保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
