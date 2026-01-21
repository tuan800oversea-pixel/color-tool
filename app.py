import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import io
import math

# --- 页面配置 ---
st.set_page_config(page_title="色彩管理工具 Pro (专业版)", layout="wide")
st.title("🎨 色彩对照提取与色卡生成工具")

# --- 颜色提取函数 ---
def process_images(rgb_file, cmyk_file):
    img_rgb = Image.open(rgb_file).convert('RGB')
    img_cmyk = Image.open(cmyk_file).convert('CMYK')
    if img_rgb.size != img_cmyk.size:
        img_cmyk = img_cmyk.resize(img_rgb.size, Image.Resampling.NEAREST)
    
    # 采样以提升计算速度
    small_size = (int(img_rgb.width * 0.2), int(img_rgb.height * 0.2))
    img_rgb_s = img_rgb.resize(small_size, Image.Resampling.NEAREST)
    img_cmyk_s = img_cmyk.resize(small_size, Image.Resampling.NEAREST)
    
    arr_rgb = np.array(img_rgb_s).reshape(-1, 3)
    arr_cmyk = np.array(img_cmyk_s).reshape(-1, 4)
    
    # 提取主要色群
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
        # 简易预览转换
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

# --- 核心绘图函数 (1行4个 + 超大字号 + 统一RGB标注 + 白底) ---
def create_tif_chart(selected_items, mode="RGB"):
    DPI = 300
    CM_TO_INCH = 1 / 2.54
    BLOCK_PX = int(4 * CM_TO_INCH * DPI)  # 4cm = 472像素
    
    # 预留文字高度与间距
    TEXT_H_PX = int(1.3 * CM_TO_INCH * DPI) 
    MARGIN_PX = int(0.5 * CM_TO_INCH * DPI) 
    COLUMNS = 4
    
    num_items = len(selected_items)
    rows = math.ceil(num_items / COLUMNS)
    
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    # 强制设置背景与文字颜色
    if mode == "RGB":
        bg_color, text_color = (255, 255, 255), (0, 0, 0)
    else:
        # CMYK模式：(0,0,0,0)为纯白底，(0,0,0,255)为单黑字
        bg_color, text_color = (0, 0, 0, 0), (0, 0, 0, 255) 

    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 字号：100pt 粗体
    try:
        font = ImageFont.truetype("arialbd.ttf", 100) 
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
        
        # 统一标注文字显示为 RGB 数值
        label = f"R:{int(item['RGB_R'])} G:{int(item['RGB_G'])} B:{int(item['RGB_B'])}"
        
        # 1. 绘制色块 (带1px细边框防止浅色看不清)
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=text_color, width=1)
        
        # 2. 计算居中位置
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        
        # 如果三位数值太满，自动降号
        current_font = font
        if text_w > BLOCK_PX:
            current_font = ImageFont.truetype("arialbd.ttf", 85)
            bbox = draw.textbbox((0, 0), label, font=current_font)
            text_w = bbox[2] - bbox[0]
            
        text_x = x + (BLOCK_PX - text_w) // 2
        
        # 3. 绘制文字标注
        draw.text((text_x, y + BLOCK_PX + 20), label, fill=text_color, font=current_font)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()

# --- 界面 ---
c1, c2 = st.columns(2)
with c1: 
    design_img = st.file_uploader("1. 上传设计师稿 (RGB)", type=['tif', 'tiff', 'jpg', 'png'])
with c2: 
    factory_img = st.file_uploader("2. 上传工厂稿 (CMYK)", type=['tif', 'tiff', 'jpg', 'png'])

if design_img and factory_img:
    if st.button("🚀 开始提取颜色并生成预览"):
        with st.spinner("正在提取主色彩..."):
            st.session_state['data_list'] = process_images(design_img, factory_img)

if 'data_list' in st.session_state:
    st.divider()
    # --- 新增的引导提示框 ---
    st.info("💡 **操作提示**：下方展示了从图稿中提取的主要颜色。若有不想打印的色块，请**取消勾选**左侧按钮即可排除。")
    
    st.subheader("🔍 色块勾选确认")
    
    selected_indices = []
    for i, row in enumerate(st.session_state['data_list']):
        col_chk, col_txt, col_pre1, col_pre2 = st.columns([1, 2, 4, 4])
        with col_chk:
            # 默认勾选，用户可手动去掉
            checked = st.checkbox(f"导出", value=True, key=f"chk_{i}")
        with col_txt:
            st.write(f"颜色 {i+1} ({row['占比']})")
        with col_pre1:
            st.markdown(f'<div style="background-color:{row["设计图色块"]}; height:50px; border:1px solid #000; text-align:center; color:white; font-size:12px; line-height:50px; text-shadow: 1px 1px 2px #000;">设计师稿</div>', unsafe_allow_html=True)
        with col_pre2:
            st.markdown(f'<div style="background-color:{row["工厂稿色块"]}; height:50px; border:1px solid #000; text-align:center; color:white; font-size:12px; line-height:50px; text-shadow: 1px 1px 2px #000;">工厂稿</div>', unsafe_allow_html=True)
        
        if checked:
            selected_indices.append(row)

    st.divider()
    if selected_indices:
        st.write(f"当前已选 **{len(selected_indices)}** 个色块准备生成图稿")
        ca, cb = st.columns(2)
        with ca:
            st.download_button(
                label="📥 下载设计师校色条 (RGB模式)",
                data=create_tif_chart(selected_indices, "RGB"),
                file_name="Check_RGB_Strip.tif",
                mime="image/tiff",
                use_container_width=True
            )
        with cb:
            st.download_button(
                label="📥 下载工厂打样色条 (CMYK模式)",
                data=create_tif_chart(selected_indices, "CMYK"),
                file_name="Print_CMYK_Strip.tif",
                mime="image/tiff",
                use_container_width=True
            )
