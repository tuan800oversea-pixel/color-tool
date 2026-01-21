import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
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
    
    # 进一步缩小尺寸以确保稳定性
    small_size = (int(img_rgb.width * 0.2), int(img_rgb.height * 0.2))
    img_rgb_s = img_rgb.resize(small_size, Image.Resampling.NEAREST)
    img_cmyk_s = img_cmyk.resize(small_size, Image.Resampling.NEAREST)
    
    arr_rgb = np.array(img_rgb_s).reshape(-1, 3)
    arr_cmyk = np.array(img_cmyk_s).reshape(-1, 4)
    
    # 增加聚类数量以获取更多候选色
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=5)
    labels = kmeans.fit_predict(arr_rgb)
    
    results = []
    label_counts = Counter(labels)
    for label_idx, count in label_counts.most_common(15):
        if (count / len(arr_rgb)) < 0.01: continue
        mask = (labels == label_idx)
        r, g, b = Counter([tuple(x) for x in arr_rgb[mask]]).most_common(1)[0][0]
        c, m, y, k = Counter([tuple(x) for x in arr_cmyk[mask]]).most_common(1)[0][0]
        
        # 将 RGB 转换为十六进制用于表格背景色预览
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        # 将 CMYK 转换为近似 RGB 十六进制用于预览
        c_p, m_p, y_p, k_p = c/255, m/255, y/255, k/255
        r_p = round(255 * (1-c_p) * (1-k_p))
        g_p = round(255 * (1-m_p) * (1-k_p))
        b_p = round(255 * (1-y_p) * (1-k_p))
        hex_cmyk = '#{:02x}{:02x}{:02x}'.format(r_p, g_p, b_p)
        
        results.append({
            "打样": True,
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "设计图预览": hex_color,
            "工厂图预览": hex_cmyk,
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100)
        })
    return results

def create_tif_chart(df, mode="RGB"):
    BLOCK_PX, TEXT_H_PX, MARGIN_PX = 400, 160, 50
    COLUMNS = 4
    rows = math.ceil(len(df) / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 50
    
    img = Image.new(mode, (canvas_w, canvas_h), (255,255,255) if mode=="RGB" else (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    for i, (_, row) in enumerate(df.iterrows()):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = 50 + MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (int(row['RGB_R']), int(row['RGB_G']), int(row['RGB_B'])) if mode=="RGB" else \
               (int(row['CMYK_C']*2.55), int(row['CMYK_M']*2.55), int(row['CMYK_Y']*2.55), int(row['CMYK_K']*2.55))
        
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=0, width=3)
        # 标注 RGB 值
        txt = f"R:{row['RGB_R']} G:{row['RGB_G']} B:{row['RGB_B']}"
        draw.text((x + 20, y + BLOCK_PX + 30), txt, fill=0)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
    return buf.getvalue()

# --- 界面 ---
col1, col2 = st.columns(2)
with col1: design_img = st.file_uploader("1. 上传设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with col2: factory_img = st.file_uploader("2. 上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始提取颜色并对比"):
        with st.spinner("正在抓取核心颜色..."):
            st.session_state['raw_data'] = process_images(design_img, factory_img)

if 'raw_data' in st.session_state:
    st.subheader("💡 颜色校对表 (取消勾选不需要的浅色/杂色)")
    
    df_display = pd.DataFrame(st.session_state['raw_data'])
    
    # 兼容低版本的表格编辑方案
    edited_df = st.data_editor(
        df_display,
        column_config={
            "打样": st.column_config.CheckboxColumn("生成?", default=True),
            "设计图预览": st.column_config.TextColumn("设计色预览 (Hex)"),
            "工厂图预览": st.column_config.TextColumn("工厂色预览 (Hex)"),
        },
        hide_index=True,
    )

    # 过滤勾选行
    final_df = edited_df[edited_df["打样"] == True]

    st.divider()
    if not final_df.empty:
        st.success(f"已选择 {len(final_df)} 个色块")
        c_a, c_b = st.columns(2)
        with c_a:
            st.download_button(
                label="📥 设计师核对校色用 (RGB 模式)",
                data=create_tif_chart(final_df, "RGB"),
                file_name="设计师校色_RGB.tif",
                mime="image/tiff",
                use_container_width=True
            )
        with c_b:
            st.download_button(
                label="📥 工厂打样用 (CMYK 模式)",
                data=create_tif_chart(final_df, "CMYK"),
                file_name="工厂打样_CMYK.tif",
                mime="image/tiff",
                use_container_width=True
            )
    else:
        st.warning("请勾选至少一个颜色")
