import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from collections import Counter
import io
import math

# --- 页面配置 ---
st.set_page_config(page_title="色彩管理工具", layout="wide")
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
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
    labels = kmeans.fit_predict(arr_rgb)
    
    results = []
    label_counts = Counter(labels)
    for label_idx, count in label_counts.most_common(8):
        if (count / len(arr_rgb)) < 0.02: continue
        mask = (labels == label_idx)
        r, g, b = Counter([tuple(x) for x in arr_rgb[mask]]).most_common(1)[0][0]
        c, m, y, k = Counter([tuple(x) for x in arr_cmyk[mask]]).most_common(1)[0][0]
        
        results.append({
            "工厂": factory_name,
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100)
        })
    return results

def create_tif_chart(df, mode="RGB"):
    BLOCK_PX, TEXT_H_PX, MARGIN_PX = 400, 120, 50
    COLUMNS = 4
    rows = math.ceil(len(df) / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 100
    
    img = Image.new(mode, (canvas_w, canvas_h), (255,255,255) if mode=="RGB" else (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    for i, (_, row) in enumerate(df.iterrows()):
        r, c = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c * (BLOCK_PX + MARGIN_PX)
        y = 100 + MARGIN_PX + r * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (int(row['RGB_R']), int(row['RGB_G']), int(row['RGB_B'])) if mode=="RGB" else \
               (int(row['CMYK_C']*2.55), int(row['CMYK_M']*2.55), int(row['CMYK_Y']*2.55), int(row['CMYK_K']*2.55))
        
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=0, width=3)
        txt = f"R:{row['RGB_R']} G:{row['RGB_G']} B:{row['RGB_B']}"
        draw.text((x + 20, y + BLOCK_PX + 20), txt, fill=0)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
    return buf.getvalue()

# --- 界面 ---
c1, c2 = st.columns(2)
with c1: design_img = st.file_uploader("上传设计稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: factory_img = st.file_uploader("上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    fac_name = st.text_input("输入工厂名称", "MyFactory")
    if st.button("开始提取颜色"):
        with st.spinner("处理中..."):
            data = process_images(design_img, factory_img, fac_name)
            st.session_state['color_df'] = pd.DataFrame(data)

if 'color_df' in st.session_state:
    df = st.session_state['color_df']
    st.success("颜色提取成功！")
    st.table(df) # 使用最基础的表格显示，绝对不会报错
    
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("下载 RGB 模式 TIF", create_tif_chart(df, "RGB"), "RGB_Output.tif", "image/tiff")
    with col_b:
        st.download_button("下载 CMYK 模式 TIF", create_tif_chart(df, "CMYK"), "CMYK_Output.tif", "image/tiff")
