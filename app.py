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
        # 简易转换预览色
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

# --- 核心绘图函数 (1行8个+大字体优化) ---
def create_tif_chart(selected_items, mode="RGB"):
    DPI = 300
    CM_TO_INCH = 1 / 2.54
    BLOCK_PX = int(4 * CM_TO_INCH * DPI) # 472 px
    
    # 调整布局为 1 行 8 个
    COLUMNS = 8 
    # 增加文字区域高度以适应大字体
    TEXT_H_PX = int(1.2 * CM_TO_INCH * DPI) 
    MARGIN_PX = int(0.5 * CM_TO_INCH * DPI) 
    
    num_items = len(selected_items)
    rows = math.ceil(num_items / COLUMNS)
    
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    if mode == "RGB":
        bg_color, text_color = (255, 255, 255), (0, 0, 0)
    else:
        bg_color, text_color = (255, 255, 255, 255), (0, 0, 0, 255) # 改为白底黑字便于打印

    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 增大字号至 90 (如果 arialbd 不存在则用默认)
    try:
        font = ImageFont.truetype("arialbd.ttf", 90) 
    except:
        font = ImageFont.load_default()
    
    for i, item in enumerate(selected_items):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        if mode == "RGB":
            fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B']))
            label = f"R{int(item['RGB_R'])}\nG{int(item['RGB_G'])}\nB{int(item['RGB_B'])}"
        else:
            fill = (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
            label = f"C{item['CMYK_C']}\nM{item['CMYK_M']}\nY{item['CMYK_Y']}\nK{item['CMYK_K']}"
        
        # 1. 绘制色块
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill)
        
        # 2. 绘制标注 (换行显示，字号大更清晰)
        draw.text((x, y + BLOCK_PX + 15), label, fill=text_color, font=font, spacing=10)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()

# --- 界面逻辑 ---
c1, c2 = st.columns(2)
with c1: design_img = st.file_uploader("1. 设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: factory_img = st.file_uploader("2. 工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始对比提取"):
        with st.spinner("处理中..."):
            st.session_state['data_list'] = process_images(design_img, factory_img)

if 'data_list' in st.session_state:
    st.subheader("🔍 选择要导出的色块")
    
    selected_indices = []
    for i, row in enumerate(st.session_state['data_list']):
        col_chk, col_txt, col_pre1, col_pre2 = st.columns([1, 2, 4, 4])
        with col_chk:
            checked = st.checkbox(f"选择", value=True, key=f"chk_{i}")
        with col_txt:
            st.write(f"颜色 {i+1}")
        with col_pre1:
            st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:40px; border:1px solid #ddd;"></div>', unsafe_allow_html=True)
        with col_pre2:
            st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:40px; border:1px solid #ddd;"></div>', unsafe_allow_html=True)
        
        if checked:
            selected_indices.append(row)

    st.divider()
    if selected_indices:
        ca, cb = st.columns(2)
        with ca:
            st.download_button("📥 下载 RGB 校色条 (1x8)", create_tif_chart(selected_indices, "RGB"), "RGB_Strip.tif", "image/tiff")
        with cb:
            st.download_button("📥 下载 CMYK 打样条 (1x8)", create_tif_chart(selected_indices, "CMYK"), "CMYK_Strip.tif", "image/tiff")
