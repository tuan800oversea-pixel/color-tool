import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import io
import math

# --- 页面配置 ---
st.set_page_config(page_title="色彩管理工具 Pro", layout="wide")
st.title("🎨 色彩对照提取与色卡生成工具")

# --- 核心提取函数 ---
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

# --- 核心绘图函数（重点修复字体与标注） ---
def create_tif_chart(selected_items, mode="RGB"):
    # 极大化参数：BLOCK 400, TEXT区域增加到 450 (翻倍以上)
    BLOCK_PX, TEXT_H_PX, MARGIN_PX = 400, 450, 100
    COLUMNS = 4
    num_items = len(selected_items)
    rows = math.ceil(num_items / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    # 颜色设置：CMYK模式下，单黑文字需要 (0,0,0,255)
    if mode == "RGB":
        bg_color = (255, 255, 255)
        text_color = (0, 0, 0)
    else:
        bg_color = (0, 0, 0, 0)
        text_color = (0, 0, 0, 255) # K=100，确保工厂能看到
    
    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 字体大小直接拉满到 200 (比之前又大了近一倍)
    try:
        font = ImageFont.truetype("arialbd.ttf", 200) 
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 200)
        except:
            font = ImageFont.load_default()
    
    for i, item in enumerate(selected_items):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        # 色块填充逻辑
        if mode == "RGB":
            fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B']))
        else:
            fill = (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
        
        # 1. 绘制色块
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=text_color, width=10)
        
        # 2. 绘制多行标注（让数值更醒目）
        line1 = f"R:{int(item['RGB_R'])}"
        line2 = f"G:{int(item['RGB_G'])}"
        line3 = f"B:{int(item['RGB_B'])}"
        
        # 依次向下排列，每行间隔 120 像素
        draw.text((x, y + BLOCK_PX + 40), line1, fill=text_color, font=font)
        draw.text((x, y + BLOCK_PX + 160), line2, fill=text_color, font=font)
        draw.text((x, y + BLOCK_PX + 280), line3, fill=text_color, font=font)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
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
    st.subheader("🔍 色块校对与选择")
    
    if 'checks' not in st.session_state or len(st.session_state['checks']) != len(st.session_state['data_list']):
        st.session_state['checks'] = [True] * len(st.session_state['data_list'])

    selected_indices = []
    
    for i, row in enumerate(st.session_state['data_list']):
        col_chk, col_txt, col_pre1, col_pre2 = st.columns([1, 2, 4, 4])
        with col_chk:
            st.session_state['checks'][i] = st.checkbox(f"生成", value=st.session_state['checks'][i], key=f"chk_{i}")
        with col_txt:
            st.write(f"颜色 {i+1}\n({row['占比']})")
        with col_pre1:
            st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:60px; border:2px solid #000; text-align:center; line-height:60px; color:white; font-weight:bold; text-shadow:1px 1px 2px #000;">设计图色</div>', unsafe_allow_html=True)
        with col_pre2:
            st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:60px; border:2px solid #000; text-align:center; line-height:60px; color:white; font-weight:bold; text-shadow:1px 1px 2px #000;">工厂预览</div>', unsafe_allow_html=True)
        
        if st.session_state['checks'][i]:
            selected_indices.append(row)

    st.divider()
    
    if selected_indices:
        ca, cb = st.columns(2)
        with ca:
            st.download_button(
                "📥 设计师核对校色用 (RGB 模式 - 巨大字体版)", 
                create_tif_chart(selected_indices, "RGB"), 
                "设计师校色_RGB_巨大字.tif", 
                "image/tiff", 
                use_container_width=True
            )
        with cb:
            st.download_button(
                "📥 工厂打样用 (CMYK 模式 - 包含RGB标注)", 
                create_tif_chart(selected_indices, "CMYK"), 
                "工厂打样_CMYK_巨大字.tif", 
                "image/tiff", 
                use_container_width=True
            )
