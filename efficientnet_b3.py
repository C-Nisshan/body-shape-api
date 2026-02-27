from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms, models
from PIL import Image
import io
import os
from typing import Dict
import torch.nn as nn

app = FastAPI(
    title="Body Shape Classification API",
    description="Predicts one of 5 female body shapes from uploaded photos",
    version="1.1.0"
)

# ────────────────────────────────────────────────
# Configuration – match EfficientNet-B3 training
# ────────────────────────────────────────────────

body_shapes = ['apple', 'hourglass', 'inverted_triangle', 'pear', 'rectangle']
NUM_CLASSES = len(body_shapes)
IMG_SIZE = 300
MODEL_PATH = "EfficientNet-B3_best_model.pth"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B3
model = models.efficientnet_b3(weights="DEFAULT")
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES)
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found.\n"
        f"Current directory: {os.getcwd()}\n"
        f"Expected file: {MODEL_PATH}"
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"EfficientNet-B3 model loaded on device: {device}")

# ────────────────────────────────────────────────
# Prediction endpoint
# ────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, etc.)")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        predicted = body_shapes[pred_idx]

        probabilities: Dict[str, float] = {
            shape: round(float(probs[i]), 6)
            for i, shape in enumerate(body_shapes)
        }

        return {
            "predicted_shape": predicted,
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Server error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "classes": body_shapes
    }