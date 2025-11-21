import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2
import numpy as np

st.title("YOLOv8 Trash Detection – Webcam (WebRTC)")

# Load model (gunakan torchscript)
model = YOLO("best.torchscript")


# ============================
# VIDEO TRANSFORMER
# ============================

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLO inference
        results = self.model(img, imgsz=640)
        annotated = results[0].plot()

        return annotated


# ============================
# STREAMLIT UI
# ============================

st.write("Klik tombol di bawah untuk mulai webcam:")

webrtc_streamer(
    key="yolo-webcam",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
