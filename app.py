import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import math
import io

# --- 页面配置 ---
st.set_page_config(page_title="🎨 色彩对照提取与色卡生成工具", layout="wide")

# --- 参数配置 (根据你的脚本优化) ---
DPI = 300
CM_TO_INCH = 1 / 2.54
BLOCK_SIZE_PX = int(4 * CM_TO_INCH * DPI)  # 4cm = 472px
TEXT_H_PX = int(1.0 * CM_TO_INCH * DPI)    # 文字区域
MARGIN_PX = int(0.5 * CM_TO_INCH * DPI)    # 边距
COLUMNS = 4                                # 1行4个
FONT_SIZE = 55                             # 优化后的大字号

def get_font(size):
    for name in ["arialbd.ttf", "simhei.ttf", "arial.ttf"]:
        try: return ImageFont.truetype(name, size)
        except: continue
    return ImageFont.load_default()

def create_chart(items, mode="RGB"):
    rows = math.ceil(len(items) / COLUMNS)
    canvas_w = (BLOCK_SIZE_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_SIZE_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 100
    
    img_mode = "RGB" if mode == "RGB" else "CMYK"
    bg_color = (255, 255, 255) if mode == "RGB" else (0, 0, 0, 0)
    text_color = (0, 0, 0) if mode == "RGB" else (0, 0, 0, 255)

    img = Image.new(img_mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    font = get_font(FONT_SIZE)

    for i, item in enumerate(items):
        r_idx, c_idx = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_idx * (BLOCK_SIZE_PX + MARGIN_PX)
        y = 100 + r_idx * (BLOCK_SIZE_PX + TEXT_H_PX + MARGIN_PX)
        
        # 色块填充
        if mode == "RGB":
            fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B']))
            label = f"R:{item['RGB_R']} G:{item['RGB_G']} B:{item['RGB_B']}"
        else:
            fill = (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
            label = f"C:{item['CMYK_C']} M:{item['CMYK_M']} Y:{item['CMYK_Y']} K:{item['CMYK_K']}"
        
        # 绘制色块
        draw.rectangle([x, y, x + BLOCK_SIZE_PX, y + BLOCK_SIZE_PX], fill=fill, outline=text_color, width=2)
        
        # 文字居中绘制
        bbox = draw.textbbox((0, 0), label, font=font)
        text_x = x + (BLOCK_SIZE_PX - (bbox[2] - bbox[0])) / 2
        draw.text((text_x, y + BLOCK_SIZE_PX + 15), label, fill=text_color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()

# --- Streamlit 界面 ---
st.title("🎨 色彩对照提取与色卡生成工具")
uploaded_file = st.file_uploader("上传 Excel 文件 (需包含 RGB/CMYK 数据)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("数据预览：", df.head())
    
    if st.button("生成色卡预览"):
        data_list = df.to_dict('records')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 下载 RGB 校色", create_chart(data_list, "RGB"), "RGB_Chart.tif", "image/tiff")
        with col2:
            st.download_button("📥 下载 CMYK 打样", create_chart(data_list, "CMYK"), "CMYK_Chart.tif", "image/tiff")
