import os
import streamlit as st
from utils.styling import load_css
import pandas as pd
from utils.config import Config
import sys
from pathlib import Path
import requests

root_path = Path(__file__).parent
sys.path.append(str(root_path))

st.set_page_config(page_title="Predicting Pneumonia", page_icon="🫁", layout="wide")
st.header("Pneumonia Prediction")
st.markdown("Please make sure that your image is a Chest X-Ray Image")

API_URL = f"http://api:8003/predict"


model = st.selectbox(
    "Select model:",
    options=["ResNet50V2", "MobileNetV2"]
)

if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

uploader = st.file_uploader("Upload Chest X-Ray Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploader:
    st.session_state.uploaded_files = uploader

    if st.button("Predict!"):
        with st.spinner("Predicting... please wait ⏳"):

            files = []
            for img in st.session_state.uploaded_files:
                files.append(("image_list", (img.name, img.read(), img.type)))

            response = requests.post(
                API_URL,
                files=files,
                data={"model": model}
            )

        if response.status_code == 200:
            st.session_state.predictions = response.json()["predictions"]
            st.success("Prediction complete!")
        else:
            st.error(f"Prediction failed: {response.text}")
            st.stop()

if st.session_state.predictions is not None:

    uploader = st.session_state.uploaded_files
    y_pred = st.session_state.predictions

    st.subheader("Results:")

    df_results = pd.DataFrame({
        "filename": [img.name for img in uploader],
        "prediction": ["Normal" if int(y_pred[i]) == 0 else "Pneumonia"
                       for i in range(len(uploader))]
    })

    csv = df_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    num_cols = 3
    cols = st.columns(num_cols)

    for idx, img in enumerate(uploader):
        col = cols[idx % num_cols]

        with col:
            with st.container(border=True):
                st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                label = df_results["prediction"][idx]
                st.write(f"Filename: {img.name}")
                st.write(f"Results: **{label}**")
