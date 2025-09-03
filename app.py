import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import pandas as pd

# ------------------------------
# Load YOLOv8 model
# ------------------------------
model = YOLO("best.pt")  # update path if needed

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Helmet & Accident Detection 🚦", page_icon="🪖", layout="wide")

# ------------------------------
# Sidebar - Detection Settings
# ------------------------------
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
theme_choice = st.sidebar.radio("🎨 Choose Theme", ["Dark", "Light"])
st.sidebar.info("Adjust confidence & theme for better detection")

# ------------------------------
# Custom CSS for Modern Look
# ------------------------------
if theme_choice == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color: #121212; color: white;}
        h1,h2,h3,h4,h5,h6,p,label {color:white !important;}
        .stButton button, .stDownloadButton button {
            background-color:#ffffff; color:black; border-radius:12px; font-weight:bold; padding:0.5em 1em;
        }
        .stSidebar {background-color:#1f1f1f;}
        .card {border-radius:15px; padding:15px; background-color:#1e1e1e; box-shadow:0 0 15px rgba(0,0,0,0.3);}
        .badge {display:inline-block; padding:5px 12px; margin:3px; border-radius:12px; background-color:#00bcd4; color:white;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {background-color: #f9f9f9; color: black;}
        h1,h2,h3,h4,h5,h6,p,label {color:black !important;}
        .stButton button, .stDownloadButton button {
            background-color:black; color:white; border-radius:12px; font-weight:bold; padding:0.5em 1em;
        }
        .stSidebar {background-color:#e0e0e0;}
        .card {border-radius:15px; padding:15px; background-color:#ffffff; box-shadow:0 0 15px rgba(0,0,0,0.2);}
        .badge {display:inline-block; padding:5px 12px; margin:3px; border-radius:12px; background-color:#ff5722; color:white;}
        </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Main Title
# ------------------------------
st.markdown("<h1 style='text-align:center'>🪖 Helmet & Accident Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload an image or video for detection</p>", unsafe_allow_html=True)

# ------------------------------
# File Upload Section
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload Image/Video", type=["jpg","jpeg","png","mp4"])

if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Preview image
    if uploaded_file.type in ["image/jpeg","image/png","image/jpg"]:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", width=500)

    # Preview video
    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)

    st.markdown("</div>", unsafe_allow_html=True)

    # Process Detection
    if st.button("🚀 Run Detection"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if uploaded_file.type in ["image/jpeg","image/png","image/jpg"]:
            with st.spinner("🔍 Running detection..."):
                results = model.predict(img, conf=conf_threshold)
            res_img = results[0].plot()
            st.image(res_img, caption="Processed Image", width=500)

            # Detection Summary as badges
            class_counts = results[0].boxes.cls.cpu().numpy()
            labels = [results[0].names[int(i)] for i in class_counts]
            df = pd.Series(labels).value_counts()

            st.subheader("📊 Detection Summary")
            for cls, count in df.items():
                st.markdown(f"<span class='badge'>{cls}: {count}</span>", unsafe_allow_html=True)

        elif uploaded_file.type == "video/mp4":
            with st.spinner("🔍 Processing video..."):
                results = model.predict(source=video_path, save=True, conf=conf_threshold)
            output_dir = results[0].save_dir
            processed_video_path = os.path.join(output_dir, os.path.basename(video_path))
            st.success("✅ Video processed successfully!")
            st.subheader("📥 Download Processed Video")
            with open(processed_video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        st.markdown("</div>", unsafe_allow_html=True)
