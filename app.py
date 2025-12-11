import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av

# ============================
# CUSTOM PAGE CONFIG
# ============================
st.set_page_config(
    page_title="SWAYO – Smart Waste YOLO",
    layout="wide",
    page_icon="🗑️"
)

# ============================
# CUSTOM CSS FOR MODERN UI
# ============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
}

.main-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    margin-top: 30px;
}

h1 {
    font-weight: 700 !important;
    color: #1c3d27 !important;
}

.start-btn {
    background: #00a884;
    padding: 12px 25px;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: 0.2s;
}

.start-btn:hover {
    background: #008b6f;
    transform: scale(1.02);
}

.footer {
    margin-top: 50px;
    text-align: center;
    opacity: 0.7;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ============================
# LOAD MODEL
# ============================
model = YOLO("best.torchscript")

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, imgsz=640)
        annotated = results[0].plot()
        return annotated


# ============================
# PAGE LAYOUT
# ============================
st.markdown("<h1>🗑️ Smart Waste Classifier with YOLOv8</h1>", unsafe_allow_html=True)
st.write("Deteksi sampah secara real-time menggunakan model YOLOv8 yang sudah Anda latih.")

st.markdown('<div class="main-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🎥 Live Detection (Webcam)")
    st.write("Tekan tombol di bawah untuk memulai live detection:")

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write("""
Smart Waste Classifier (SWAYO) mendeteksi jenis sampah secara otomatis 
menggunakan model YOLOv8 yang telah dilatih.
    
**Fitur:**
- Real-time detection via webcam  
- Model cepat & akurat  
- Hasil annotasi langsung tampil  
""")

    st.markdown('<button class="start-btn">Mulai Deteksi Sekarang</button>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="footer">© 2025 SWAYO – Smart Waste Classifier with YOLO</p>', unsafe_allow_html=True)
