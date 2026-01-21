import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import io
import math

# --- 页面配置 ---
st.set_page_config(page_title="色彩管理与色卡生成工具", layout="wide")
st.title("🎨 色彩对照提取与色卡生成工具")
st.markdown("上传设计稿与工厂稿，自动提取主色并生成印刷色卡。")

# --- 核心逻辑函数 ---
def cmyk_to_rgb_hex(c, m, y, k):
    c_norm, m_norm, y_norm, k_norm = c/255.0, m/255.0, y/255.0, k/255.0
    r = max(0, min(255, round(255 * (1 - c_norm) * (1 - k_norm))))
    g = max(0, min(255, round(255 * (1 - m_norm) * (1 - k_norm))))
    b = max(0, min(255, round(255 * (1 - y_norm) * (1 - k_norm))))
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def process_images(rgb_file, cmyk_file, factory_name):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    
    if img_rgb.size != img_cmyk.size:
        img_cmyk = img_cmyk.resize(img_rgb.size, Image.Resampling.NEAREST)
    
    # 缩小尺寸加快处理速度
    small_size = (int(img_rgb.width * 0.4), int(img_rgb.height * 0.4))
    img_rgb_s = img_rgb.resize(small_size, Image.Resampling.NEAREST)
    img_cmyk_s = img_cmyk.resize(small_size, Image.Resampling.NEAREST)
    
    arr_rgb = np.array(img_rgb_s).reshape(-1, 3)
    arr_cmyk = np.array(img_cmyk_s).reshape(-1, 4)
    
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=5)
    labels = kmeans.fit_predict(arr_rgb)
    
    results = []
    label_counts = Counter(labels)
    for label_idx, count in label_counts.most_common(10):
        if (count / len(arr_rgb)) < 0.02: continue
        mask = (labels == label_idx)
        r, g, b = Counter([tuple(x) for x in arr_rgb[mask]]).most_common(1)[0][0]
        c, m, y, k = Counter([tuple(x) for x in arr_cmyk[mask]]).most_common(1)[0][0]
        
        results.append({
            "工厂": factory_name,
            "占比": f"{(count / len(arr_rgb)):.1%}",
            "RGB_R": r, "RGB_G": g, "RGB_B": b,
            "CMYK_C": round(c/255*100), "CMYK_M": round(m/255*100), 
            "CMYK_Y": round(y/255*100), "CMYK_K": round(k/255*100),
            "hex": '#{:02x}{:02x}{:02x}'.format(r, g, b)
        })
    return results

def create_tif_chart(df, mode="RGB"):
    # 简化版色卡生成逻辑
    BLOCK_PX = 400
    TEXT_H_PX = 120
    MARGIN_PX = 50
    COLUMNS = 4
    rows = math.ceil(len(df) / COLUMNS)
    
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1)) + 100
    
    img = Image.new(mode, (canvas_w, canvas_h), (255,255,255) if mode=="RGB" else (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    for i, row in df.iterrows():
        r, c = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c * (BLOCK_PX + MARGIN_PX)
        y = 100 + MARGIN_PX + r * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (row['RGB_R'], row['RGB_G'], row['RGB_B']) if mode=="RGB" else \
               (int(row['CMYK_C']*2.55), int(row['CMYK_M']*2.55), int(row['CMYK_Y']*2.55), int(row['CMYK_K']*2.55))
        
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=0, width=3)
        txt = f"R:{row['RGB_R']} G:{row['RGB_G']} B:{row['RGB_B']}"
        draw.text((x + 20, y + BLOCK_PX + 20), txt, fill=0)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw')
    return buf.getvalue()

# --- 界面交互 ---
col1, col2 = st.columns(2)
with col1:
    design_img = st.file_uploader("上传设计稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with col2:
    factory_img = st.file_uploader("上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    fac_name = st.text_input("输入工厂名称", "某某工厂")
    if st.button("开始提取颜色"):
        with st.spinner("正在分析颜色，请稍候..."):
            data = process_images(design_img, factory_img, fac_name)
            st.session_state['color_data'] = pd.DataFrame(data)

if 'color_data' in st.session_state:
    df = st.session_state['color_data']
    st.success("颜色提取完成！")
    
    # 展示带颜色的表格
    st.dataframe(df.style.background_gradient(subset=['hex'], gprop='background-color'))
    
    st.divider()
    st.subheader("生成并下载色卡")
    c1, c2 = st.columns(2)
    with c1:
        rgb_tif = create_tif_chart(df, "RGB")
        st.download_button("下载 RGB 模式 TIF", rgb_tif, "RGB_Chart.tif", "image/tiff")
    with c2:
        cmyk_tif = create_tif_chart(df, "CMYK")
        st.download_button("下载 CMYK 模式 TIF", cmyk_tif, "CMYK_Chart.tif", "image/tiff")