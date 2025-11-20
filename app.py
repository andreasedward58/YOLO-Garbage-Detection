import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

st.title("Trash Detection with YOLO")

model = YOLO("best.pt")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    results = model.predict(img)
    result_img = results[0].plot()

    st.image(result_img, caption="Prediction Result")
