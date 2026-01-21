def create_tif_chart(selected_items, mode="RGB"):
    # 300DPI 标准参数
    DPI = 300
    CM_TO_INCH = 1 / 2.54
    BLOCK_PX = int(4 * CM_TO_INCH * DPI)  # 4cm 色块 = 472 px
    
    # --- 修改点 1: 一行 8 个 ---
    COLUMNS = 8 
    
    # --- 修改点 2: 增大文字高度和字号 ---
    # 将基础高度增加，以容纳更大的字体 (原本是 0.5cm，现在改为约 1cm 的空间)
    TEXT_H_PX = int(1.0 * CM_TO_INCH * DPI) + 60 
    MARGIN_PX = int(0.5 * CM_TO_INCH * DPI) 
    
    num_items = len(selected_items)
    rows = math.ceil(num_items / COLUMNS)
    
    canvas_w = (BLOCK_PX * COLUMNS) + (MARGIN_PX * (COLUMNS + 1))
    canvas_h = ((BLOCK_PX + TEXT_H_PX) * rows) + (MARGIN_PX * (rows + 1))
    
    if mode == "RGB":
        bg_color, text_color = (255, 255, 255), (0, 0, 0)
    else:
        bg_color, text_color = (0, 0, 0, 0), (0, 0, 0, 255) 

    img = Image.new(mode, (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # --- 修改点 3: 显著增大字号 ---
    # 从 75 增加到 100-120 左右，视需求而定
    try:
        font = ImageFont.truetype("arialbd.ttf", 110)  
    except:
        font = ImageFont.load_default()
    
    for i, item in enumerate(selected_items):
        r_pos, c_pos = i // COLUMNS, i % COLUMNS
        x = MARGIN_PX + c_pos * (BLOCK_PX + MARGIN_PX)
        y = MARGIN_PX + r_pos * (BLOCK_PX + TEXT_H_PX + MARGIN_PX)
        
        if mode == "RGB":
            fill = (int(item['RGB_R']), int(item['RGB_G']), int(item['RGB_B']))
        else:
            # 修正：CMYK 填充需要直接使用 0-255 的值，或者根据 item 中的 CMYK_C 等百分比转换
            fill = (int(item['CMYK_C']*2.55), int(item['CMYK_M']*2.55), int(item['CMYK_Y']*2.55), int(item['CMYK_K']*2.55))
        
        # 1. 绘制色块
        draw.rectangle([x, y, x + BLOCK_PX, y + BLOCK_PX], fill=fill, outline=None)
        
        # 2. 绘制标注
        if mode == "RGB":
            label = f"R:{int(item['RGB_R'])}\nG:{int(item['RGB_G'])}\nB:{int(item['RGB_B'])}"
        else:
            label = f"C:{item['CMYK_C']}\nM:{item['CMYK_M']}\nY:{item['CMYK_Y']}\nK:{item['CMYK_K']}"
            
        # 放置在色块下方，增加间距
        draw.text((x, y + BLOCK_PX + 20), label, fill=text_color, font=font, spacing=10)
        
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression='tiff_lzw', dpi=(300, 300))
    return buf.getvalue()
