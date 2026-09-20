import io
import base64
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Autonomous Venomous Snake Identification & Advisory API")

# 1. Device and Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)
class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
num_classes = len(class_names)

weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load("mobilenet_snake_classifier_v2.pth", map_location=device))
model.to(device)

# Ensure parameters allow gradient tracking for Grad-CAM
for param in model.parameters():
    param.requires_grad = True

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict_specimen(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare tensor
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 1. Vision Prediction
    model.eval()
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probs, 0)
    predicted_species = class_names[pred_idx.item()]
    confidence_val = float(conf.item() * 100)

    # 2. XAI Grad-CAM Generation
    rgb_img = cv2.resize(np.array(pil_img), (224, 224))
    rgb_img = np.float32(rgb_img) / 255.0

    targets = [ClassifierOutputTarget(pred_idx.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert heatmap to base64 JPEG
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    is_venomous = "VENOMOUS" in predicted_species.upper()

    return JSONResponse({
        "species": predicted_species,
        "confidence": round(confidence_val, 2),
        "is_venomous": is_venomous,
        "heatmap_base64": heatmap_base64
    })