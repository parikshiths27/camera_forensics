import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import numpy as np

# Ensure src modules can be imported
import sys
sys.path.append(os.path.dirname(__file__))

from src.preprocessing import preprocess_image
from src.residual_extraction import extract_residual
from src.predict import predict_camera

st.set_page_config(page_title="Camera Model Identification", layout="wide")

# Model paths
MODEL_PATH = r"c:\Projects\camera_forensics\models\forensics_cnn.pth"
CLASS_MAPPING_PATH = r"c:\Projects\camera_forensics\models\class_mapping.txt"

def run_app():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Gradient Header */
        .main-header {
            background: linear-gradient(135deg, #FF6B6B 0%, #556270 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0px;
            text-align: center;
        }
        
        .sub-header {
            text-align: center;
            color: #bdc3c7;
            font-weight: 300;
            margin-bottom: 30px;
        }
        
        /* Glassmorphism Container */
        .glass-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        /* Metric Box */
        .metric-box {
            background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 5px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📷 Camera Forensics AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify the intrinsic sensor noise fingerprint of modern smartphones.</p>', unsafe_allow_html=True)

    # Input option: Upload image
    with st.sidebar:
        st.markdown("## ⚙️ Pipeline Input")
        st.info("💡 **Pro-Tip**: If you open this URL on your phone you can tap 'Browse files' to take a live photo straight from your camera!")
        image_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("### 🔍 Model Information")
        st.markdown("""
        - **Architecture**: PyTorch ResNet-18
        - **Feature**: Deep PRNU Extraction
        - **Patches**: 16 instances (32x32px)
        - **Aggregation**: Extracted Majority Vote
        """)

    if image_file is not None:
        # We need to save it to a temporary file to run the cv2 processing logic
        image = Image.open(image_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(tmp_file.name)
            temp_path = tmp_file.name
            
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Original Capture")
            st.image(image, use_container_width=True)
            
        # Process to show Residual
        try:
            img_prep = preprocess_image(temp_path)
            residual = extract_residual(img_prep)
            
            with col2:
                st.markdown("### 🔬 Sensor Fingerprint")
                res_view = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
                res_view = res_view.astype(np.uint8)
                st.image(res_view, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)
            
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAPPING_PATH):
            st.warning(f"Model or class mapping not found at {MODEL_PATH}. Your ongoing training run will place it there once finished!")
        else:
            with st.spinner("🧠 Analyzing high-frequency noise patterns..."):
                try:
                    pred_class, conf = predict_camera(temp_path, MODEL_PATH, CLASS_MAPPING_PATH)
                    
                    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                    st.markdown("### 📊 AI Forensic Report")
                    
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f"""
                        <div class="metric-box" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);">
                            <div class="metric-title">Predicted Camera</div>
                            <div class="metric-value">{pred_class.replace('_', ' ').title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with mc2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-title">Fingerprint Confidence</div>
                            <div class="metric-value">{conf * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    
        with st.expander("📖 Uncover the Science (How it works)"):
            st.markdown("""
            1. **Scene Removal**: We mathematically denoise the image to separate the physical object (a tree, a face) from the high-frequency electronic noise.
            2. **PRNU Extraction**: Subtracting the denoised image from the original isolates the sensor's unique pixel-level noise pattern (Photo Response Non-Uniformity).
            3. **Deep Classification**: The AI looks at 16 tiny patches of this isolated static and uses deep convolutions (ResNet-18) to match the noise syntax to known hardware architectures.
            """)
            
        os.remove(temp_path) # cleanup

if __name__ == "__main__":
    run_app()
