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
def process_images(rgb_file, cmyk_file, factory_name):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    
    if img_rgb.size != img_cmyk.size:
        img_cmyk = img_cmyk.resize(img_rgb.size, Image.Resampling.NEAREST)
    
    # 缩小尺寸加快处理速度
    small_size = (int(img_rgb.width * 0.3), int(img_rgb.height * 0.3))
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
        
        results.append({
            "打样": True, # 默认为勾选状态
            "工厂": factory_name,
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "RGB预览": f"rgb({r},{g},{b})",
            "CMYK预览": f"rgb({round(255*(1-c/255)*(1-k/255))},{round(255*(1-m/255)*(1-k/255))},{round(255*(1-y/255)*(1-k/255))})",
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100)
        })
    return results

def create_tif_chart(df, mode="RGB"):
    BLOCK_PX, TEXT_H_PX, MARGIN_PX = 400, 150, 50
    COLUMNS = 4
    rows = math.ceil(len(df) / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 100
    
    img = Image.new(mode, (canvas_w, canvas_h), (255,255,255) if mode=="RGB" else (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    for i, (_, row) in enumerate(df.iterrows()):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = 100 + MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (int(row['RGB_R']), int(row['RGB_G']), int(row['RGB_B'])) if mode=="RGB" else \
               (int(row['CMYK_C']*2.55), int(row['CMYK_M']*2.55), int(row['CMYK_Y']*2.55), int(row['CMYK_K']*2.55))
        
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=0, width=3)
        # 加上 RGB 文字约定
        txt = f"R:{row['RGB_R']} G:{row['RGB_G']} B:{row['RGB_B']}"
        draw.text((x + 20, y + BLOCK_PX + 20), txt, fill=0)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
    return buf.getvalue()

# --- 界面 ---
col1, col2 = st.columns(2)
with col1: design_img = st.file_uploader("1. 上传设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with col2: factory_img = st.file_uploader("2. 上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    fac_name = st.text_input("工厂名称", "MyFactory")
    if st.button("🚀 开始提取颜色并对比"):
        with st.spinner("正在抓取核心颜色..."):
            st.session_state['raw_data'] = process_images(design_img, factory_img, fac_name)

if 'raw_data' in st.session_state:
    st.subheader("💡 颜色校对表 (勾选需要生成的颜色)")
    # 使用 data_editor 实现可勾选的表格
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state['raw_data']),
        column_config={
            "打样": st.column_config.CheckboxColumn("生成?", default=True),
            "RGB预览": st.column_config.ColorColumn("设计图色块"),
            "CMYK预览": st.column_config.ColorColumn("工厂图预览"),
        },
        disabled=["工厂", "占比", "RGB_R", "RGB_G", "RGB_B", "CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"],
        hide_index=True,
    )

    # 过滤出用户勾选要“打样”的行
    final_df = edited_df[edited_df["打样"] == True]

    st.divider()
    if not final_df.empty:
        st.success(f"已选择 {len(final_df)} 个颜色，准备生成文件：")
        c_a, c_b = st.columns(2)
        with c_a:
            st.download_button(
                label="📥 下载：设计师核对校色用 (RGB.tif)",
                data=create_tif_chart(final_df, "RGB"),
                file_name=f"{fac_name}_设计师校色_RGB.tif",
                mime="image/tiff",
                use_container_width=True
            )
        with c_b:
            st.download_button(
                label="📥 下载：工厂打样用 (CMYK.tif)",
                data=create_tif_chart(final_df, "CMYK"),
                file_name=f"{fac_name}_工厂打样_CMYK.tif",
                mime="image/tiff",
                use_container_width=True
            )
    else:
        st.warning("请至少勾选一个颜色用于生成文件。")
