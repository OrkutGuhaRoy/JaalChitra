import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import math
import io

# ==========================================
# 1. UI/UX CONFIGURATION (JaalChitra)
# ==========================================

st.set_page_config(
    page_title="JaalChitra",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme & Spider Web CSS
st.markdown("""
<style>
    :root {
        --primary-color: #009688;
        --background-color: #0e1117;
        --text-color: #fafafa;
    }
    
    .stApp {
        background-color: var(--background-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 150, 136, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(0, 150, 136, 0.1) 0%, transparent 20%);
    }

    /* Title Styling */
    .jaal-title {
        font-family: 'Courier New', Courier, monospace;
        text-align: center;
        color: var(--primary-color);
        text-shadow: 0px 0px 15px rgba(0, 150, 136, 0.8);
        font-size: 4rem;
        font-weight: bold;
        letter-spacing: -2px;
        margin-bottom: 0px;
        animation: glow 2s infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px #009688, 0 0 20px #009688; }
        to { text-shadow: 0 0 20px #26a69a, 0 0 30px #26a69a; }
    }

    .jaal-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-family: monospace;
    }

    /* Spider Web Animation Overlay */
    .web-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        opacity: 0.05;
        pointer-events: none;
        z-index: 0;
    }

    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(45deg, #009688, #00796b);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 1px;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 3rem;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 150, 136, 0.6);
    }
    
    /* Loading Spinner Color */
    .stSpinner > div {
        border-top-color: #009688 !important;
    }
</style>
<div class="web-overlay"></div>
<h1 class="jaal-title">🕸️ JaalChitra</h1>
<div class="jaal-subtitle">Digital String Art Generator</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE ALGORITHM
# ==========================================

def get_peg_coordinates(width, height, n_pegs, shape):
    """Calculates peg coordinates."""
    pegs = []
    if shape == "Ellipsis (Circle)":
        cx, cy = width / 2, height / 2
        rx, ry = (width - 1) / 2, (height - 1) / 2
        for i in range(n_pegs):
            angle = 2 * math.pi * i / n_pegs
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
            pegs.append((x, y))
    else: # Rectangle
        perimeter = 2 * (width + height)
        step = perimeter / n_pegs
        for i in range(n_pegs):
            d = i * step
            if d < width:
                pegs.append((d, 0))
            elif d < width + height:
                pegs.append((width - 1, d - width))
            elif d < 2 * width + height:
                pegs.append((width - 1 - (d - (width + height)), height - 1))
            else:
                pegs.append((0, height - 1 - (d - (2 * width + height))))
    return pegs

def compute_line_score_and_indices(img, r0, c0, r1, c1):
    """Computes the sum of pixel values along a line."""
    dist = np.hypot(r1-r0, c1-c0)
    if dist == 0: return 0, None, None
    
    num = int(dist) 
    
    rs = np.linspace(r0, r1, num)
    cs = np.linspace(c0, c1, num)
    
    r_idx = rs.astype(np.int32)
    c_idx = cs.astype(np.int32)
    
    valid = (r_idx >= 0) & (r_idx < img.shape[0]) & (c_idx >= 0) & (c_idx < img.shape[1])
    r_idx = r_idx[valid]
    c_idx = c_idx[valid]
    
    if len(r_idx) == 0: return 0, None, None
    
    score = np.sum(img[r_idx, c_idx])
    
    return score, r_idx, c_idx

def generate_string_art(image_input, n_pegs, max_lines, opacity_val, shape, invert, 
                        status_placeholder=None, canvas_preview=None, canvas_color=None):
    """
    Core Algorithm Implementation with Live Rendering.
    """
    # 1. Resize for Calculation 
    calc_size = 300
    w_orig, h_orig = image_input.size
    scale = min(calc_size/w_orig, calc_size/h_orig)
    w = int(w_orig * scale)
    h = int(h_orig * scale)
    
    img_calc = image_input.resize((w, h), Image.Resampling.LANCZOS).convert('L')
    img_arr = np.array(img_calc, dtype=np.float32)
    
    if not invert: 
        img_arr = 255.0 - img_arr
    
    # 2. Pegs
    pegs = get_peg_coordinates(w, h, n_pegs, shape)
    
    # 3. Greedy Algorithm
    lines = []
    current_peg = 0
    prev_peg = 0
    
    # Algorithm weight (simulation opacity)
    line_weight = (opacity_val / 5.0) * 40 
    
    # Setup live preview drawing tool if canvas provided
    draw_preview = None
    if canvas_preview and canvas_color:
        draw_preview = ImageDraw.Draw(canvas_preview, "RGBA")
        # Convert Hex to RGB + Alpha
        c_r = int(canvas_color[1:3], 16)
        c_g = int(canvas_color[3:5], 16)
        c_b = int(canvas_color[5:7], 16)
        # Use a lower alpha for preview so we see buildup
        live_alpha = int((opacity_val / 5.0) * 100)
        fill_color = (c_r, c_g, c_b, live_alpha)
        
        # Scaling factor for preview (preview is usually high res, calc is low res)
        p_w, p_h = canvas_preview.size
        scale_x = p_w / w
        scale_y = p_h / h
    
    progress_bar = st.progress(0)
    
    for l in range(max_lines):
        best_peg = -1
        best_score = -1
        best_indices = None
        
        min_separation = int(n_pegs / 20)
        
        # Random sampling optimization for speed vs quality
        # Check a subset of pegs to speed up the loop for live rendering
        # or check all. We check all minus neighbors.
        for i in range(min_separation, n_pegs - min_separation):
            target = (current_peg + i) % n_pegs
            if target == prev_peg: continue 
            
            p1 = pegs[current_peg]
            p2 = pegs[target]
            
            score, r_idx, c_idx = compute_line_score_and_indices(img_arr, p1[1], p1[0], p2[1], p2[0])
            
            if score > best_score:
                best_score = score
                best_peg = target
                best_indices = (r_idx, c_idx)
        
        if best_peg == -1: break
            
        # Add line to result
        p1_norm = (pegs[current_peg][0] / w, pegs[current_peg][1] / h)
        p2_norm = (pegs[best_peg][0] / w, pegs[best_peg][1] / h)
        lines.append((p1_norm, p2_norm))
        
        # Error Diffusion
        if best_indices:
            r_idx, c_idx = best_indices
            img_arr[r_idx, c_idx] = np.maximum(0, img_arr[r_idx, c_idx] - line_weight)
        
        # Real-time Render
        if draw_preview:
            x0, y0 = p1_norm[0] * p_w, p1_norm[1] * p_h
            x1, y1 = p2_norm[0] * p_w, p2_norm[1] * p_h
            draw_preview.line([x0, y0, x1, y1], fill=fill_color, width=1)
            
            # Update UI every 50 lines to prevent lag
            if l % 50 == 0 and status_placeholder:
                status_placeholder.image(canvas_preview, caption=f"Weaving Thread {l}/{max_lines}", use_container_width=True)
                progress_bar.progress(min(l / max_lines, 1.0))

        prev_peg = current_peg
        current_peg = best_peg
        
            
    progress_bar.empty()
    return lines

def render_lines(lines, width, height, color, bg_color, thickness, opacity_val):
    """Renders high-quality output using PIL."""
    scale = 2 
    img = Image.new("RGB", (width * scale, height * scale), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")
    
    c_r = int(color[1:3], 16)
    c_g = int(color[3:5], 16)
    c_b = int(color[5:7], 16)
    
    alpha = int((opacity_val / 5.0) * 100)
    fill_col = (c_r, c_g, c_b, alpha)
    
    for p1, p2 in lines:
        x0, y0 = p1[0] * width * scale, p1[1] * height * scale
        x1, y1 = p2[0] * width * scale, p2[1] * height * scale
        draw.line([x0, y0, x1, y1], fill=fill_col, width=int(thickness * scale))
        
    return img.resize((width, height), Image.Resampling.LANCZOS)

def get_svg(lines, width, height, color, bg_color, thickness, opacity_val):
    op = (opacity_val / 5.0) * 0.5
    svg = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background-color:{bg_color}">']
    svg.append(f'<g stroke="{color}" stroke-width="{thickness}" stroke-opacity="{op}" fill="none">')
    for p1, p2 in lines:
        x1, y1 = p1[0] * width, p1[1] * height
        x2, y2 = p2[0] * width, p2[1] * height
        svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />')
    svg.append('</g></svg>')
    return "".join(svg)

# ==========================================
# 3. SIDEBAR & EXECUTION
# ==========================================

# Initialize Session State
if 'generated_lines' not in st.session_state:
    st.session_state['generated_lines'] = None
if 'generated_lines_rgb' not in st.session_state:
    st.session_state['generated_lines_rgb'] = None
if 'last_mode' not in st.session_state:
    st.session_state['last_mode'] = None

with st.sidebar:
    st.markdown("## 🎛️ Settings")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "webp"])
    
    st.divider()
    
    # SECTION: GENERATION (Requires Re-run)
    st.markdown("### 1. Generation Parameters")
    st.caption("Changing these requires re-weaving.")
    shape = st.selectbox("Frame Shape", ["Rectangle", "Ellipsis (Circle)"])
    n_pegs = st.slider("Pegs Count", 100, 5000, 250)
    max_lines = st.slider("Max Lines (Threads)", 1000, 50000, 2500, step=100)
    sim_opacity = st.slider("Simulation Weight", 1, 5, 2, help="Algorithm sensitivity. Higher = bolder threads assumed during calculation.")
    
    mode = st.radio("Style", ["Monochrome", "RGB (Color)"])
    dark_mode = st.toggle("Dark Mode (Invert)", value=True)
    
    generate_btn = st.button("🕸️ Weave Art (Generate)", type="primary", use_container_width=True)

    st.divider()
    
    # SECTION: RENDERING (Instant)
    st.markdown("### 2. Live Rendering Options")
    st.caption("Change these instantly without waiting.")
    thickness = st.slider("Line Thickness", 0.1, 3.0, 0.4, step=0.1)
    render_opacity = st.slider("Display Opacity", 1, 5, 2, help="Visual transparency of the threads.")

col1, col2 = st.columns([1, 1.5])

if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Source")
        st.image(input_img, use_container_width=True)
        
        # Determine Colors based on Dark Mode
        if dark_mode:
            bg = "#000000"
            fg_mono = "#FFFFFF"
        else:
            bg = "#FFFFFF"
            fg_mono = "#000000"
            
        with col2:
            st.subheader("Result")
            res_box = st.empty()
            
            # LOGIC: GENERATION VS RENDERING
            # If Button Clicked -> Run Algorithm (Slow, with Live View)
            # If Not Clicked but Data Exists -> Run Renderer (Fast)
            
            if generate_btn:
                st.session_state['last_mode'] = mode
                
                # --- MONOCHROME GENERATION ---
                if mode == "Monochrome":
                    # Prepare Canvas for Live View
                    w, h = input_img.size
                    # Use a reasonable preview size to keep UI responsive
                    scale_preview = min(800/w, 800/h)
                    pw, ph = int(w*scale_preview), int(h*scale_preview)
                    live_canvas = Image.new("RGB", (pw, ph), bg)
                    
                    # Run
                    lines = generate_string_art(
                        input_img, n_pegs, max_lines, sim_opacity, shape, dark_mode,
                        status_placeholder=res_box,
                        canvas_preview=live_canvas,
                        canvas_color=fg_mono
                    )
                    
                    # Store
                    st.session_state['generated_lines'] = lines
                    st.session_state['generated_lines_rgb'] = None
                
                # --- RGB GENERATION ---
                else:
                    st.info("Weaving 3 layers (R, G, B)...")
                    img_r, img_g, img_b = input_img.split()
                    
                    # Prepare Canvas (Composite)
                    w, h = input_img.size
                    scale_preview = min(800/w, 800/h)
                    pw, ph = int(w*scale_preview), int(h*scale_preview)
                    live_canvas = Image.new("RGB", (pw, ph), bg)
                    
                    # We pass the SAME live_canvas to all 3 calls so they add up visually!
                    
                    lines_r = generate_string_art(
                        img_r.convert("RGB"), n_pegs, int(max_lines/3), sim_opacity, shape, dark_mode,
                        status_placeholder=res_box, canvas_preview=live_canvas, canvas_color="#FF0000"
                    )
                    lines_g = generate_string_art(
                        img_g.convert("RGB"), n_pegs, int(max_lines/3), sim_opacity, shape, dark_mode,
                        status_placeholder=res_box, canvas_preview=live_canvas, canvas_color="#00FF00"
                    )
                    lines_b = generate_string_art(
                        img_b.convert("RGB"), n_pegs, int(max_lines/3), sim_opacity, shape, dark_mode,
                        status_placeholder=res_box, canvas_preview=live_canvas, canvas_color="#0000FF"
                    )
                    
                    st.session_state['generated_lines_rgb'] = (lines_r, lines_g, lines_b)
                    st.session_state['generated_lines'] = None

            # --- FINAL DISPLAY / CACHED RENDER ---
            # This block runs if we just generated OR if we are tweaking sliders
            
            if st.session_state['last_mode'] == "Monochrome" and st.session_state['generated_lines']:
                lines = st.session_state['generated_lines']
                # Render using CURRENT visual settings (thickness, render_opacity)
                out_img = render_lines(lines, input_img.width, input_img.height, fg_mono, bg, thickness, render_opacity)
                svg_out = get_svg(lines, input_img.width, input_img.height, fg_mono, bg, thickness, render_opacity)
                
                res_box.image(out_img, use_container_width=True)
                
                # Download Buttons
                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                b1, b2 = st.columns(2)
                b1.download_button("💾 Save PNG", buf.getvalue(), "jaalchitra.png", "image/png")
                b2.download_button("📐 Save SVG", svg_out, "jaalchitra.svg", "image/svg+xml")

            elif st.session_state['last_mode'] == "RGB (Color)" and st.session_state['generated_lines_rgb']:
                lines_r, lines_g, lines_b = st.session_state['generated_lines_rgb']
                
                # Composite Render
                w, h = input_img.size
                scale = 2
                comp_img = Image.new("RGB", (w*scale, h*scale), bg)
                draw = ImageDraw.Draw(comp_img, "RGBA")
                alpha = int((render_opacity / 5.0) * 100)
                
                def draw_layer(lns, color_hex):
                    c = (int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16), alpha)
                    for p1, p2 in lns:
                        draw.line([p1[0]*w*scale, p1[1]*h*scale, p2[0]*w*scale, p2[1]*h*scale], fill=c, width=int(thickness*scale))
                
                draw_layer(lines_r, "#FF0000")
                draw_layer(lines_g, "#00FF00")
                draw_layer(lines_b, "#0000FF")
                
                out_img = comp_img.resize((w, h), Image.Resampling.LANCZOS)
                res_box.image(out_img, use_container_width=True)
                
                # SVG
                svg_out = f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:{bg}">'
                def get_g(lns, col):
                    op = (render_opacity/5.0)*0.5
                    g = [f'<g stroke="{col}" stroke-width="{thickness}" stroke-opacity="{op}">']
                    g.extend([f'<line x1="{p1[0]*w:.1f}" y1="{p1[1]*h:.1f}" x2="{p2[0]*w:.1f}" y2="{p2[1]*h:.1f}" />' for p1, p2 in lns])
                    g.append('</g>')
                    return "".join(g)
                svg_out += get_g(lines_r, "#FF0000") + get_g(lines_g, "#00FF00") + get_g(lines_b, "#0000FF") + "</svg>"
                
                # Downloads
                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                b1, b2 = st.columns(2)
                b1.download_button("💾 Save PNG", buf.getvalue(), "jaalchitra.png", "image/png")
                b2.download_button("📐 Save SVG", svg_out, "jaalchitra.svg", "image/svg+xml")

else:
    with col2:
        st.markdown("""
        <div style="
            border: 2px dashed #009688; 
            border-radius: 10px; 
            height: 400px; 
            display: flex; 
            align-items: center; 
            justify-content: center;
            opacity: 0.3;">
            <h2>Waiting for Image...</h2>
        </div>
        """, unsafe_allow_html=True)