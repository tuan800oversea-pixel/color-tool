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

# --- 1. 置顶前置提醒 ---
with st.container():
    st.warning("### 📌 上传前置提醒 (必读)")
    c_tip1, c_tip2 = st.columns(2)
    with c_tip1:
        st.markdown("""
        * **内容要求**：上传图稿必须为**同一花型**。
        * **对位要求**：设计稿(RGB)与工厂稿(CMYK)的**尺寸**、**位置**必须完全一致。
        """)
    with c_tip2:
        st.markdown("""
        * **操作建议**：人工核对上传图稿的对位情况，防止位移导致数据失效。
        * **无效判定**：若两边图稿尺寸或花位不符，色彩提取结果将不具参考价值。
        """)

st.divider()

# --- 2. 核心逻辑函数 ---
def process_images(rgb_file, cmyk_file):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    
    # 强制检查或调整尺寸（如果尺寸不一，强制对位）
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
        # 简易转换以便前端展示
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

def create_tif_chart(selected_items, mode="RGB"):
    DPI = 300
    CM_TO_INCH = 1 / 2.54
    BLOCK_PX = int(4 * CM_TO_INCH * DPI) 
    TEXT_H_PX = int(1.3 * CM_TO_INCH * DPI) 
    MARGIN_PX = int(0.5 * CM_TO_INCH * DPI) 
    COLUMNS = 4
    
    rows = math.ceil(len(selected_items) / COLUMNS)
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    bg_color, text_color = ((255, 255, 255), (0, 0, 0)) if mode == "RGB" else ((0, 0, 0, 0), (0, 0, 0, 255))
    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    try: font = ImageFont.truetype("arialbd.ttf", 100)
    except: font = ImageFont.load_default()
    
    for i, item in enumerate(selected_items):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B'])) if mode == "RGB" else \
               (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
        
        # 统一输出 RGB 标注
        label = f"R:{int(item['RGB_R'])} G:{int(item['RGB_G'])} B:{int(item['RGB_B'])}"
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=text_color, width=1)
        
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        cur_font = font
        if text_w > BLOCK_PX:
            cur_font = ImageFont.truetype("arialbd.ttf", 85)
            bbox = draw.textbbox((0, 0), label, font=cur_font)
            text_w = bbox[2] - bbox[0]
        
        draw.text((x + (BLOCK_PX - text_w) // 2, y + BLOCK_PX + 20), label, fill=text_color, font=cur_font)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()

# --- 3. 界面交互 ---
c1, c2 = st.columns(2)
with c1: design_img = st.file_uploader("1. 设计师稿 (RGB格式)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: factory_img = st.file_uploader("2. 工厂打样稿 (CMYK格式)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始提取颜色并核验对位"):
        with st.spinner("正在对比花型色彩..."):
            st.session_state['data_list'] = process_images(design_img, factory_img)

if 'data_list' in st.session_state:
    st.divider()
    
    # --- 您要求的核心警告提示放置在勾选区域正上方 ---
    st.error("""
    ⚠️ **核心核验提醒：**
    * **异常判定**：若提取出的双侧色块差距巨大（如：一侧为纯黑，另一侧为彩色），说明图稿**花型位置未对齐**。
    * **人工核验**：请务必人工核对上传图稿的对位情况。**位置不一致会导致色彩提取数据完全无效**。
    """)
    
    st.subheader("🔍 色块勾选确认")
    st.info("💡 取消勾选左侧按钮可排除不需要打印的色块。")
    
    selected_indices = []
    for i, row in enumerate(st.session_state['data_list']):
        col_chk, col_txt, col_pre1, col_pre2 = st.columns([1, 2, 4, 4])
        with col_chk:
            checked = st.checkbox(f"导出", value=True, key=f"chk_{i}")
        with col_txt:
            st.write(f"颜色 {i+1} ({row['占比']})")
        with col_pre1:
            st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:50px; border:1px solid #000; text-align:center; color:white; font-size:12px; line-height:50px; text-shadow: 1px 1px 2px #000;">设计(RGB)</div>', unsafe_allow_html=True)
        with col_pre2:
            st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:50px; border:1px solid #000; text-align:center; color:white; font-size:12px; line-height:50px; text-shadow: 1px 1px 2px #000;">工厂(CMYK)</div>', unsafe_allow_html=True)
        if checked: selected_indices.append(row)

    st.divider()
    if selected_indices:
        ca, cb = st.columns(2)
        with ca:
            st.download_button("📥 下载设计师校色色块", create_tif_chart(selected_indices, "RGB"), "设计师校色色块.tif", "image/tiff", use_container_width=True)
        with cb:
            st.download_button("📥 下载工厂打样色色块", create_tif_chart(selected_indices, "CMYK"), "工厂打样色色块.tif", "image/tiff", use_container_width=True)
