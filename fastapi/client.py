import requests
from PIL import Image
import numpy as np
import pandas as pd
import io
import base64
import streamlit as st
import pydicom  # Make sure to add this to requirements.txt

# --- CONSTANTS ---
URL = "http://localhost:8000/predict"  # Changed from /predict to match your app.py


# --- HELPER FUNCTIONS ---
def get_display_image(uploaded_file):
    """Safely converts JPG/PNG/DICOM into a PIL Image for Streamlit display."""
    if uploaded_file.name.lower().endswith((".dcm", ".dicom")):
        # Read DICOM bytes
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
        uploaded_file.seek(0)  # Reset pointer for later use

        # Get pixel data and normalize for display (8-bit)
        img_array = ds.pixel_array
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    else:
        return Image.open(uploaded_file)


# --- UI SETUP ---
st.set_page_config(page_title="Chest X-ray Analyzer", layout="wide")
st.title("🩻 Chest X-ray Analyzer")
st.write("Upload a chest X-ray image (JPG, PNG, or DICOM) to detect potential pathologies.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "dcm", "dicom"])

if uploaded_file:
    col1, col2 = st.columns(2)

    try:
        # 1. Handle Image Preview (Works for both standard and DICOM)
        display_img = get_display_image(uploaded_file)

        with col1:
            st.subheader("Original Image")
            st.image(display_img, use_container_width=True)

        # 2. Analysis Button
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Server is processing..."):
                # Reset file pointer to ensure we send the whole file to the API
                uploaded_file.seek(0)

                # Prepare file for requests
                files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}

                # Send to FastAPI
                response = requests.post(URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    preds = data["predictions"]
                    heatmap_b64 = data["heatmap_b64"]

                    # Fix: Use single quotes inside f-string double quotes
                    top_path = data['top_prediction']['Pathology']
                    top_perc = data['top_prediction']['Percentage']
                    st.success(f"**Top Prediction:** {top_path} ({top_perc})")

                    st.subheader("📊 Full Analysis Results")
                    df = pd.DataFrame(preds)
                    st.dataframe(df[["Pathology", "Percentage"]], height=400, use_container_width=True)

                    with col2:
                        st.subheader("🔥 Attention Heatmap (Grad-CAM)")
                        # Decode and display the heatmap returned by the server
                        img_data = base64.b64decode(heatmap_b64)
                        heatmap = Image.open(io.BytesIO(img_data))
                        st.image(heatmap, use_container_width=True)
                        st.caption("Red areas highlight regions of interest for the model's prediction.")
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload an image to begin analysis.")