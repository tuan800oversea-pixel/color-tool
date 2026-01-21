import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import io
import math

# --- 页面配置 ---
st.set_page_config(page_title="色彩管理工具 Pro (300DPI 标准版)", layout="wide")
st.title("🎨 色彩对照提取与色卡生成工具")

# --- 颜色提取函数 ---
def process_images(rgb_file, cmyk_file):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    if img_rgb.size != img_cmyk.size:
        img_cmyk = img_cmyk.resize(img_rgb.size, Image.Resampling.NEAREST)
    
    # 采样
    small_size = (int(img_rgb.width * 0.2), int(img_rgb.height * 0.2))
    img_rgb_s = img_rgb.resize(small_size, Image.Resampling.NEAREST)
    img_cmyk_s = img_cmyk.resize(small_size, Image.Resampling.NEAREST)
    
    arr_rgb = np.array(img_rgb_s).reshape(-1, 3)
    arr_cmyk = np.array(img_cmyk_s).reshape(-1, 4)
    
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=5)
    labels = kmeans.fit_predict(arr_rgb)
    
    results = []
    label_counts = Counter(labels)
    for label_idx, count in label_counts.most_common(12):
        if (count / len(arr_rgb)) < 0.01: continue
        mask = (labels == label_idx)
        r, g, b = Counter([tuple(x) for x in arr_rgb[mask]]).most_common(1)[0][0]
        c, m, y, k = Counter([tuple(x) for x in arr_cmyk[mask]]).most_common(1)[0][0]
        
        hex_design = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        r_p = round(255 * (1-c/255) * (1-k/255))
        g_p = round(255 * (1-m/255) * (1-k/255))
        b_p = round(255 * (1-y/255) * (1-k/255))
        hex_factory = '#{:02x}{:02x}{:02x}'.format(r_p, g_p, b_p)
        
        results.append({
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "设计图色块": hex_design,
            "工厂稿色块": hex_factory,
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100)
        })
    return results

# --- 核心绘图函数 ---
def create_tif_chart(selected_items, mode="RGB"):
    # 300DPI, 4cm = 472像素
    DPI = 300
    CM_TO_INCH = 1 / 2.54
    BLOCK_PX = int(4 * CM_TO_INCH * DPI) # 472 px
    TEXT_H_PX = int(0.5 * CM_TO_INCH * DPI) + 40 # 约 60px 基础高度 + 40px 缓冲
    MARGIN_PX = int(0.5 * CM_TO_INCH * DPI) # 约 60px 边距
    COLUMNS = 4
    
    num_items = len(selected_items)
    rows = math.ceil(num_items / COLUMNS)
    
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    if mode == "RGB":
        bg_color, text_color = (255, 255, 255), (0, 0, 0)
    else:
        bg_color, text_color = (0, 0, 0, 0), (0, 0, 0, 255) # CMYK 单黑

    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 字号设定：75px 在 472px 宽度的色块下非常巨大且不会换行
    try:
        font = ImageFont.truetype("arialbd.ttf", 75) 
    except:
        font = ImageFont.load_default()
    
    for i, item in enumerate(selected_items):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        if mode == "RGB":
            fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B']))
        else:
            fill = (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
        
        # 1. 绘制 4cm 色块 (无边框)
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=None)
        
        # 2. 绘制标注 (RGB 数值)
        label = f"R:{int(item['RGB_R'])} G:{int(item['RGB_G'])} B:{int(item['RGB_B'])}"
        # 放置在色块下方 10px 处
        draw.text((x, y + BLOCK_PX + 10), label, fill=text_color, font=font)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()

# --- 界面 ---
c1, c2 = st.columns(2)
with c1: design_img = st.file_uploader("1. 上传设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: factory_img = st.file_uploader("2. 上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始提取颜色并对比"):
        with st.spinner("处理中..."):
            st.session_state['data_list'] = process_images(design_img, factory_img)

if 'data_list' in st.session_state:
    st.subheader("🔍 色块选择")
    if 'checks' not in st.session_state:
        st.session_state['checks'] = [True] * len(st.session_state['data_list'])

    selected_indices = []
    for i, row in enumerate(st.session_state['data_list']):
        col_chk, col_txt, col_pre1, col_pre2 = st.columns([1, 2, 4, 4])
        with col_chk:
            st.session_state['checks'][i] = st.checkbox(f"生成", value=st.session_state['checks'][i], key=f"chk_{i}")
        with col_txt:
            st.write(f"颜色 {i+1} ({row['占比']})")
        with col_pre1:
            st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:50px; border:1px solid #000;"></div>', unsafe_allow_html=True)
        with col_pre2:
            st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:50px; border:1px solid #000;"></div>', unsafe_allow_html=True)
        
        if st.session_state['checks'][i]:
            selected_indices.append(row)

    st.divider()
    if selected_indices:
        ca, cb = st.columns(2)
        with ca:
            st.download_button("📥 设计师核对校色用 (RGB)", create_tif_chart(selected_indices, "RGB"), "校色_RGB.tif", "image/tiff", use_container_width=True)
        with cb:
            st.download_button("📥 工厂打样用 (CMYK)", create_tif_chart(selected_indices, "CMYK"), "打样_CMYK.tif", "image/tiff", use_container_width=True)
