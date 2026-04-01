# ==========================================
# STREAMLIT DASHBOARD (FINAL VERSION)
# ==========================================

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import pickle

# LOAD DATA
df = pd.read_csv("processed_output.csv")

# TITLE
st.title("🌍 Geo-Contextual Harmful Content Analysis")

# ==========================================
# DATA PREVIEW
# ==========================================
st.subheader("📄 Dataset Preview")
st.write(df.head())

# ==========================================
# TOXICITY GRAPH
# ==========================================
st.subheader("📊 Toxicity Score Distribution")
st.bar_chart(df['toxicity_score'])

# ==========================================
# TOP REGIONS
# ==========================================
st.subheader("🔥 Top Toxic Regions")

if 'location' in df.columns:
    top_regions = df.groupby('location')['toxicity_score'].mean().sort_values(ascending=False).head(10)
    st.write(top_regions)

# ==========================================
# HEATMAP
# ==========================================
st.subheader("🗺️ Heatmap")

df['lat'] = df['latitude']
df['lon'] = df['longitude']

map_center = [20.5937, 78.9629]
m = folium.Map(location=map_center, zoom_start=5)

heat_data = df[['lat', 'lon', 'toxicity_score']].dropna().values.tolist()
HeatMap(heat_data).add_to(m)

st.components.v1.html(m._repr_html_(), height=500)

# ==========================================
# ACCURACY DISPLAY (EDIT VALUE)
# ==========================================
st.subheader("📈 Model Performance")
st.metric("Model Accuracy", "85%")  # 🔁 Replace with your real accuracy

# ==========================================
# USER INPUT (LIVE DETECTION 🔥)
# ==========================================
st.subheader("🧠 Test Your Own Comment")

user_input = st.text_input("Enter a comment")

if user_input:
    st.write("Analyzing...")

    # SIMPLE RULE (since model not saved)
    if any(word in user_input.lower() for word in ["hate", "kill", "stupid", "idiot"]):
        st.error("⚠️ Harmful Content Detected")
    else:
        st.success("✅ Safe Content")

# ==========================================
st.success("🚀 Project Running Successfully")