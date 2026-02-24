from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms, models
from PIL import Image
import io
import os
from typing import Dict

app = FastAPI(
    title="Body Shape Classification API",
    description="Predicts one of 5 female body shapes from uploaded photos",
    version="1.0.0"
)

# ────────────────────────────────────────────────
# Configuration – must match your training setup
# ────────────────────────────────────────────────

body_shapes = [
    "apple",
    "hourglass",
    "inverted_triangle",
    "pear",
    "rectangle"
]

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.inception_v3(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(body_shapes))

MODEL_PATH = "best_body_shape_model.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found.\n"
        f"Current directory: {os.getcwd()}\n"
        f"Expected file: {MODEL_PATH}"
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"Model loaded successfully on device: {device}")

# ────────────────────────────────────────────────
# Main prediction endpoint
# ────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a photo → get predicted body shape
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (jpg, png, etc.)")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform & predict
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