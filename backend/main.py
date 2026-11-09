from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
from torchvision import transforms
from pathlib import Path

import torch
import io
import cv2
import numpy as np
import uuid

from gradcam_utils import generate_heatmap
from models.hybrid_model import HybridCNNTransformer

app = FastAPI(title="Leaf Disease Detection API")

# Serve generated heatmaps
app.mount(
    "/heatmaps",
    StaticFiles(directory="heatmaps"),
    name="heatmaps"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Cotton_Diseased_Leaf",
    "Cotton_Diseased_Plant",
    "Cotton_Healthy_Leaf",
    "Cotton_Healthy_Plant",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice_Bacterial_Leaf_Blight",
    "Rice_Brown_Spot",
    "Rice_Healthy",
    "Rice_Leaf_Smut",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Load model
model = HybridCNNTransformer(
    num_classes=29,
    pretrained_backbones=False
)

checkpoint_path = (
    Path(__file__).parent.parent
    / "model"
    / "best_model.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {
        "message": "Leaf Disease Detection API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)

    confidence, pred = torch.max(
        output,
        dim=1
    )

    # Prepare image for Grad-CAM
    image_resized = image.resize((224, 224))

    image_np = (
        np.array(image_resized)
        .astype(np.float32)
        / 255.0
    )

    # Generate heatmap
    heatmap = generate_heatmap(
        model=model,
        input_tensor=x,
        image_np=image_np
    )

    # Save heatmap
    filename = f"{uuid.uuid4().hex}.jpg"

    heatmap_path = (
        Path(__file__).parent
        / "heatmaps"
        / filename
    )

    cv2.imwrite(
        str(heatmap_path),
        cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    return {
        "disease": CLASS_NAMES[pred.item()],
        "confidence": round(
            float(confidence.item() * 100),
            2
        ),
        "heatmap": filename,
        "heatmap_url": f"/heatmaps/{filename}"
    }