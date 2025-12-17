import streamlit as st
import torchxrayvision as xrv
import torch
from PIL import Image
import numpy as np
import time
import pydicom
from io import BytesIO
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Page config
st.set_page_config(page_title="Chest X-ray Analyzer", layout="wide")


# Load model once and cache it
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def process_dicom(file):
    """Convert DICOM to grayscale image and strip metadata"""
    ds = pydicom.dcmread(BytesIO(file.read()))
    # Strip identifying info
    tags_to_remove = ['PatientName', 'PatientID', 'PatientBirthDate']
    for tag in tags_to_remove:
        if tag in ds:
            delattr(ds, tag)

    img_array = ds.pixel_array
    # Normalize to 0-255
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def preprocess_image(img):
    """Convert to grayscale, resize, normalize"""
    start = time.time()

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to 224x224
    img = img.resize((224, 224))

    # Convert to numpy and normalize
    img_np = np.array(img)
    img_np = xrv.datasets.normalize(img_np, 255)

    # Convert to tensor with correct shape
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    preprocess_time = time.time() - start
    return img_tensor, img_np, preprocess_time


def get_predictions(model, img_tensor):
    """Run inference and return predictions"""
    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
    inference_time = time.time() - start
    return outputs, inference_time


def generate_gradcam(model, img_tensor, target_layer_name='classifier'):
    """Generate Grad-CAM heatmap"""
    # Get the last conv layer before classifier
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Generate CAM for the highest predicted class
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


def overlay_heatmap(original_img, heatmap):
    """Overlay heatmap on original image"""
    # Resize original to 224x224 and normalize to 0-1
    img_resized = cv2.resize(np.array(original_img), (224, 224))
    img_normalized = img_resized / 255.0

    # Create RGB version if grayscale
    if len(img_normalized.shape) == 2:
        img_normalized = np.stack([img_normalized] * 3, axis=-1)

    # Overlay
    visualization = show_cam_on_image(img_normalized, heatmap, use_rgb=True)
    return visualization


# Main app
st.title("Chest X-ray Pathology Analyzer")
st.write("Upload a chest X-ray image (JPG/PNG/DICOM) to detect potential pathologies")

# Load model
try:
    model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'png', 'dcm', 'dicom'])

if uploaded_file:
    # Validate file size (10MB limit)
    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
    if file_size > 10:
        st.error(f"File too large ({file_size:.2f} MB). Please upload a file smaller than 10 MB.")
        st.stop()

    try:
        # Load image based on type
        if uploaded_file.name.endswith(('.dcm', '.dicom')):
            st.info("DICOM file detected. De-identifying metadata...")
            img = process_dicom(uploaded_file)
        else:
            img = Image.open(uploaded_file)

        # Show original image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)

        # Preprocess
        img_tensor, img_np, preprocess_time = preprocess_image(img)

        # Get predictions
        outputs, inference_time = get_predictions(model, img_tensor)

        # Display metrics
        st.subheader("⏱️ Performance Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Preprocessing Time", f"{preprocess_time:.3f}s")
        with metric_col2:
            st.metric("Inference Time", f"{inference_time:.3f}s")

        # Display predictions
        st.subheader("📊 Prediction Results")

        # Convert outputs to probabilities
        probs = torch.sigmoid(outputs[0]).numpy()

        # Create results dataframe
        results = []
        for i, pathology in enumerate(model.pathologies):
            results.append({
                'Pathology': pathology,
                'Probability': f"{probs[i]:.4f}",
                'Percentage': f"{probs[i] * 100:.2f}%"
            })

        # Sort by probability
        results = sorted(results, key=lambda x: float(x['Probability']), reverse=True)

        # Show top 5 predictions
        st.table(results[:5])

        # Show all predictions in expander
        with st.expander("View all predictions"):
            st.table(results)

        # Highlight highest prediction
        top_pathology = results[0]['Pathology']
        top_prob = results[0]['Percentage']
        st.success(f"**Top Prediction:** {top_pathology} ({top_prob})")

        # Generate and display Grad-CAM
        with col2:
            st.subheader("🔥 Attention Heatmap (Grad-CAM)")
            with st.spinner("Generating heatmap..."):
                try:
                    heatmap = generate_gradcam(model, img_tensor)
                    visualization = overlay_heatmap(img, heatmap)
                    st.image(visualization, use_container_width=True)
                    st.caption("Red areas indicate regions the model focused on for its prediction")
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {e}")

        # Download results
        st.subheader("💾 Export Results")
        results_text = f"Chest X-ray Analysis Results\n\n"
        results_text += f"Top Prediction: {top_pathology} ({top_prob})\n\n"
        results_text += "All Predictions:\n"
        for r in results:
            results_text += f"{r['Pathology']}: {r['Percentage']}\n"

        st.download_button(
            label="Download Results as Text",
            data=results_text,
            file_name="xray_analysis_results.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please make sure you uploaded a valid chest X-ray image file.")

else:
    st.info("👆 Upload an X-ray image to get started")

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This is for educational purposes only. Always consult with healthcare professionals for medical diagnosis.")