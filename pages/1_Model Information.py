import os
import streamlit as st
from utils.styling import load_css
import pandas as pd
from utils.config import Config
import plotly.express as px
import json
# Custom CSS untuk tabs
st.markdown("""
<style>
/* Semua tab */
button[role="tab"] {
    background-color: #262730 !important;
    color: white !important;
    font-weight: bold;
}

/* Tab aktif */
button[role="tab"][aria-selected="true"] {
    background-color: #1E90FF !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Training Information", layout="wide")
load_css()

@st.cache_data(ttl=3600)
def load_metrics():
    """Load and cache model metrica"""
    try:
        with open(Config.METRICS_DIR, "r") as f:
             data = json.load(f)

        df = pd.DataFrame.from_dict(data, orient="index")
        return df
    except Exception as e:
        st.error(f"Error loading model artifacts")
        return None
metrics = load_metrics()
st.title("Training Information")
st.markdown(
    'We use two models: **MobileNetV2** and **ResNet50V2**. '
    'The dataset we used to train the models is from: '
    '[Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)'
)

if metrics is not None:
    st.subheader("Model Performance Comparison")

    df_melted = metrics.reset_index().rename(columns={"index": "model"})
    df_melted = df_melted.melt(id_vars="model", var_name="metric", value_name="value")

    df_melted["value"] = df_melted["value"] * 100

    fig = px.bar(
        df_melted,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        text="value",
        title="Performance Metrics Comparison",
        height=600
    )

    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title="Metric",
        yaxis_tickformat=".0f%%",
        legend_title="Model"
    )

    st.plotly_chart(fig, use_container_width=True)
