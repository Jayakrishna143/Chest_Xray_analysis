from PIL.ImageOps import grayscale
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import torchxrayvision as xrv
import torch
import numpy as np
import pydicom
import cv2
from PIL import Image
import io
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ml_model ={}
#HELPER FUNCTIONS
def process_dicom(file_bytes):
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    img_array = ds.pixel_array
    img_array = ((img_array- img_array.min())/(img_array.max()-img_array.min())*255).astype(np.uint8)
    return Image.fromarray(img_array)

def preprocessing_image(img):
    if img.mode !="L":
        img = img.convert("L")
    img = img.resize((224,224))
    img_np = np.array(img)
    img_np = xrv.datasets.normalize(img_np,255)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor,img_np

def generate_gradcam_b64(model,img_tensor,original_img):
    target_layers = [model.features[-1]]
    cam = GradCAM(model = model,target_layers =  target_layers)
    grayscale_cam = cam(input_tensor = img_tensor,targets =None)[0,:]

    # 2. Overlay
    img_resized = cv2.resize(np.array(original_img), (224, 224))
    img_normalized = img_resized / 255.0
    if len(img_normalized.shape) == 2:
        img_normalized = np.stack([img_normalized] * 3, axis=-1)

    visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

    # 3. Convert to Base64 String to send via API
    pil_img = Image.fromarray(visualization)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


#LOADING THE MODEL
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


#CACHING
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["densenet"] = load_model()
    print("model loaded successfully")

    yield
#--SHUTDOWN--

    ml_model.clear()
    print("shutdown")

#INITITALIZE APP
app = FastAPI(title = "X_ray Analysis API",
              lifespan = lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = ml_model.get("densenet")
    if not model:
        raise HTTPException(status_code=500, detail="Model Not lodaded")
    try:
        contents = await file.read()
        if file.filename.endswith((".dcm",".dicom")):
            img = process_dicom(contents)
        else:
            img =Image.open(io.BytesIO(contents))

        #PREPROCESS
        img_tensor,img_np = preprocessing_image(img)

        with torch.no_grad():
            outputs = model(img_tensor)

        #process probabilities
        probs = torch.sigmoid(outputs[0]).numpy()
        results =[]
        for i ,pathology in enumerate(model.pathologies):
            results.append({
                "Pathology": pathology,
                "Probability": float(probs[i]),
                "Percentage" : f"{probs[i]*100:.2f}%"
            })
        #sort results
        results = sorted(results,key = lambda i: i["Probability"],reverse = True)

        heatmap_b64 = generate_gradcam_b64(model,img_tensor,img)
        return {
            "statusCode": 200,
            "predictions": results,
            "top_prediction": results[0],
            "heatmap_b64": heatmap_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host = "localhost", port = 8000)




