from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_onnx import Model
import os
from typing import List

# Defining the directory where uploaded images will be stored
IMAGEDIR = "images/"

# Defining a mapping of model output labels to human-readable age groups
id2label = {
    0: '01', 1: '02', 2: '03', 3: '04', 4: '05', 
    5: '06-07', 6: '08-09', 7: '10-12', 8: '13-15', 
    9: '16-20', 10: '21-25', 11: '26-30', 12: '31-35', 
    13: '36-40', 14: '41-45', 15: '46-50', 16: '51-55', 
    17: '56-60', 18: '61-65', 19: '66-70', 20: '71-80', 
    21: '81-90', 22: '90'
}
# Configuration settings for the ONNX quantized model. Can also use onnx model directly from the same directory.
model_path = 'models/dima806/facial_age_image_detection/onnx/model_quantized.onnx'
config = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "ViTFeatureExtractor",
    "image_std": [0.5, 0.5, 0.5],
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "height": 224,
        "width": 224
    }
}

# Initialize the FastAPI application
app = FastAPI()

# Configure Jinja2 templates for HTML responses
templates = Jinja2Templates(directory="templates")

# Serve static files from the 'images' directory
app.mount("/images", StaticFiles(directory="images"), name="images")

# Defining the root path that serves the main HTML template
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Defining the endpoint for file upload and age prediction
@app.post("/upload-files")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):
    model = Model(id2label, model_path, config)
    results = []
    for file in files:
        contents = await file.read()
        file_path = f"{IMAGEDIR}{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        if os.path.exists(file_path):
            age_prediction = model.predict_age(image_path=file_path)
            results.append({"file": file.filename, "age": age_prediction})
        else:
            results.append({"file": file.filename, "error": "File not found"})
    
    return templates.TemplateResponse("index.html", {"request": request, "results": results})

# Main block to run the app if this script is executed as the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
