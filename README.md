# 🩻 Chest X-ray Pathology Analyzer

A Streamlit web application that uses deep learning to analyze chest X-ray images and detect potential pathologies. Built with PyTorch and TorchXRayVision's pretrained DenseNet121 model.

## Features

- **Multi-format Support**: Upload JPG, PNG, or DICOM files
- **Automatic De-identification**: Strips sensitive metadata from DICOM files
- **Real-time Predictions**: Detects 18+ chest pathologies with confidence scores
- **Visual Explanations**: Grad-CAM heatmaps show what the model is focusing on
- **Performance Metrics**: Track preprocessing and inference times
- **Export Results**: Download analysis results as text file

## Detected Pathologies

The model can detect various conditions including:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Pneumonia
- Pneumothorax
- And more...

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Jayakrishna143/Chest_Xray_analysis.git
cd Chest_Xray_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## Usage

1. **Upload an Image**: Click "Choose an X-ray image" and select a chest X-ray file
   - Supported formats: JPG, PNG, DICOM (.dcm)
   - Max file size: 10 MB

2. **View Results**: 
   - See the original image on the left
   - Check prediction probabilities in the results table
   - View the Grad-CAM heatmap on the right to see model attention areas

3. **Analyze Performance**: Check preprocessing and inference times

4. **Export**: Download results as a text file for record-keeping

## How It Works

1. **Model Loading**: Uses a pretrained DenseNet121 model from TorchXRayVision
2. **Preprocessing**: 
   - Converts images to grayscale
   - Resizes to 224x224 pixels
   - Normalizes pixel values
3. **Inference**: Runs the preprocessed image through the model
4. **Explanation**: Generates Grad-CAM visualization to highlight important regions
5. **Results**: Displays probabilities for all pathologies, sorted by confidence

## File Structure
```
chest-xray-analyzer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── images/                # (Optional) Sample X-ray images for testing
```

## Technical Details

- **Model**: DenseNet121 trained on multiple chest X-ray datasets
- **Input Size**: 224x224 grayscale images
- **Output**: 18 pathology probabilities (sigmoid activation)
- **Explainability**: Grad-CAM on final convolutional layer
- **DICOM Handling**: Automatic metadata stripping for privacy

## Performance

Typical performance on consumer hardware:
- Preprocessing: ~0.01-0.05 seconds
- Inference: ~0.1-0.5 seconds (CPU), ~0.01-0.05 seconds (GPU)
- Grad-CAM generation: ~0.5-1.0 seconds

## Limitations & Disclaimer

⚠️ **IMPORTANT**: This tool is for **educational and research purposes only**. 

- NOT intended for clinical diagnosis
- NOT a substitute for professional medical advice
- Results should ALWAYS be reviewed by qualified healthcare professionals
- False positives and false negatives are possible
- Model performance may vary based on image quality and acquisition parameters

## Privacy & Security

- DICOM files are automatically de-identified (patient name, ID, birth date removed)
- No data is stored on servers
- All processing happens locally in your browser session
- Uploaded images are not saved after the session ends

## Troubleshooting

**Model won't load:**
- Ensure you have a stable internet connection (model downloads on first run)
- Check that all dependencies are correctly installed

**DICOM files not working:**
- Make sure the file has a `.dcm` or `.dicom` extension
- Some DICOM formats may not be supported

**Slow performance:**
- First run is slower as the model downloads (~100MB)
- Consider using a GPU for faster inference
- Reduce image file size if very large

**Out of memory errors:**
- Close other applications
- Try with smaller image files
- Reduce batch processing

## Contributing

Feel free to open issues or submit pull requests if you find bugs or want to add features.

## License

This project uses TorchXRayVision which is available under the Apache 2.0 license. Please check individual model licenses for commercial use.

## Acknowledgments

- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) for the pretrained models
- [Streamlit](https://streamlit.io/) for the web framework
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) for explainability visualizations

## References

- Cohen, J. P., et al. "TorchXRayVision: A library of chest X-ray datasets and models." Medical Imaging with Deep Learning (2020)
- Selvaraju, R. R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." ICCV (2017)

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Remember**: Always consult healthcare professionals for medical diagnosis and treatment decisions.
