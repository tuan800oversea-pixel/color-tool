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

# --- 核心函数 ---
def process_images(rgb_file, cmyk_file):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    
    if img_rgb.size != img_cmyk.size:
        img_cmyk = img_cmyk.resize(img_rgb.size, Image.Resampling.NEAREST)
    
    # 缩小尺寸以确保稳定性
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
        
        # 近似 CMYK 预览色
        r_p = round(255 * (1-c/255) * (1-k/255))
        g_p = round(255 * (1-m/255) * (1-k/255))
        b_p = round(255 * (1-y/255) * (1-k/255))
        hex_factory = '#{:02x}{:02x}{:02x}'.format(r_p, g_p, b_p)
        
        results.append({
            "打样": True,
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "设计图色块": hex_design,
            "工厂稿色块": hex_factory,
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100)
        })
    return results

def create_tif_chart(df, mode="RGB"):
    # 参数设置
    BLOCK_PX, TEXT_H_PX, MARGIN_PX = 400, 150, 60
    COLUMNS = 4
    rows = math.ceil(len(df) / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    img = Image.new(mode, (canvas_w, canvas_h), (255,255,255) if mode=="RGB" else (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体，如果失败使用默认
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    for i, (_, row) in enumerate(df.iterrows()):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (int(row['RGB_R']), int(row['RGB_G']), int(row['RGB_B'])) if mode=="RGB" else \
               (int(row['CMYK_C']*2.55), int(row['CMYK_M']*2.55), int(row['CMYK_Y']*2.55), int(row['CMYK_K']*2.55))
        
        # 绘制色块
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=0, width=4)
        
        # 强制标注 RGB 值文本
        label = f"R:{int(row['RGB_R'])} G:{int(row['RGB_G'])} B:{int(row['RGB_B'])}"
        draw.text((x + 10, y + BLOCK_PX + 20), label, fill=0, font=font)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
    return buf.getvalue()

# --- 界面 ---
c1, c2 = st.columns(2)
with c1: design_img = st.file_uploader("1. 上传设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: factory_img = st.file_uploader("2. 上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始提取颜色并对比"):
        with st.spinner("正在抓取核心颜色..."):
            st.session_state['data_list'] = process_images(design_img, factory_img)

if 'data_list' in st.session_state:
    st.subheader("💡 颜色校对表 (勾选需要打样的颜色)")
    
    df = pd.DataFrame(st.session_state['data_list'])
    
    # 定义预览颜色样式
    def color_preview(val):
        return f'background-color: {val}; color: {val};'

    # 使用 data_editor 配合样式预览
    edited_df = st.data_editor(
        df,
        column_config={
            "打样": st.column_config.CheckboxColumn("生成?", default=True),
            "设计图色块": None, # 隐藏原始 Hex 列，用样式展示
            "工厂稿色块": None,
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 在表格下方展示带颜色背景的预览区，解决之前预览不显示的问题
    st.markdown("### 🔍 色块校对预览 (左侧为设计图，右侧为工厂图)")
    for i, row in edited_df.iterrows():
        if row["打样"]:
            col_pre1, col_pre2, col_pre3 = st.columns([1, 2, 2])
            with col_pre1: st.write(f"颜色 {i+1} ({row['占比']})")
            with col_pre2: st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:40px; border:1px solid #000; text-align:center; line-height:40px; color:white; text-shadow:1px 1px 2px #000;">设计图色</div>', unsafe_allow_html=True)
            with col_pre3: st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:40px; border:1px solid #000; text-align:center; line-height:40px; color:white; text-shadow:1px 1px 2px #000;">工厂预览</div>', unsafe_allow_html=True)

    final_df = edited_df[edited_df["打样"] == True]

    st.divider()
    if not final_df.empty:
        ca, cb = st.columns(2)
        with ca:
            st.download_button("📥 设计师核对校色用 (RGB 模式)", create_tif_chart(final_df, "RGB"), "校色_RGB.tif", "image/tiff", use_container_width=True)
        with cb:
            st.download_button("📥 工厂打样用 (CMYK 模式)", create_tif_chart(final_df, "CMYK"), "打样_CMYK.tif", "image/tiff", use_container_width=True)
