import streamlit as st
import matplotlib.pyplot as plt
from segmentation import segment_organs, segment_tumor_option1


from feature_extraction import extract_tumor_features
from utils import read_image, overlay_mask

st.set_page_config(
    page_title="Tumor Segmentation System",
    layout="wide"
)

st.title("🧠 Tumor Segmentation & Analysis System")

st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Medical Image (JPG / PNG / DICOM)",
    type=["jpg", "png", "jpeg", "dcm"]
)

mode = st.sidebar.radio(
    "Segmentation Mode",
    ["Organ Segmentation", "Tumor Segmentation"]
)

run = st.sidebar.button("Run Segmentation")

if uploaded_file is not None:
    image = read_image(uploaded_file)

    if run:
        with st.spinner("Processing..."):
            if mode == "Organ Segmentation":
                mask = segment_organs(image)
                overlay = overlay_mask(image, mask)
                features_df = None
            else:
                mask = segment_tumor_option1(image)

                overlay = overlay_mask(image, mask)
                features_df = extract_tumor_features(mask, image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR")

        with col2:
            st.subheader("Segmentation Mask")
            st.image(mask, clamp=True)

        with col3:
            st.subheader("Overlay Result")
            st.image(overlay, channels="BGR")

        if features_df is not None and not features_df.empty:
            st.subheader("Tumor Features")
            st.dataframe(features_df)

            csv = features_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Tumor Features CSV",
                csv,
                "tumor_features.csv",
                "text/csv"
            )
else:
    st.info("Upload a medical image to begin.")
