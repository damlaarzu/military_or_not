import os
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image

default_model_path = Path(__file__).resolve().parent / "runs" / "train" / "military_or_not" / "weights" / "best.pt"
model_path = os.environ.get("MODEL_PATH", str(default_model_path))


model = YOLO(model_path)

# Title
st.title("Civilian or Military Vehicle Detection")
st.markdown("Upload a photo and let the model detect whether the vehicle is **civilian or military**.")

# image load
uploaded_file = st.file_uploader("Select an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Model is processing...")

    # Prediction
    results = model(image)

    # Annotated result image
    for r in results:
        im_array = r.plot()  # numpy array (BGR)
        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB
        st.image(im, caption='Detection Results', use_container_width=True)

    # Textual class and scores
    st.subheader("Detection Details")
    for r in results:
        boxes = r.boxes
        names = r.names
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"• {names[cls_id]} - Confidence: {conf:.2f}")
