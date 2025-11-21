import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("Trash Detection YOLOv8")

model = YOLO("best.torchscript")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img_np = np.array(img)

    results = model(img_np)
    result_img = results[0].plot()

    st.image(result_img, channels="BGR")
