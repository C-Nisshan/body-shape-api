# Body Shape Classification API

A simple, fast, and production-ready REST API that predicts one of 5 female body shapes (**apple**, **hourglass**, **inverted_triangle**, **pear**, **rectangle**) from uploaded photos using a fine-tuned **InceptionV3** model.

Built with **FastAPI** (automatic Swagger/OpenAPI documentation included).

## Features

- Upload image → get predicted body shape + confidence + full probability distribution
- Clean JSON responses
- Interactive Swagger UI for testing (`/docs`)
- Health check endpoint (`/health`)
- Runs locally with virtual environment
- CPU / GPU support (auto-detects CUDA if available)

## Model Information

- Architecture: InceptionV3 (transfer learning)
- Training dataset: Roboflow Woman-Body-Shape (~522 original images, augmented to ~1092)
- Best validation accuracy achieved: ~68–69%
- Test set accuracy: ~69–71%
- Input image size: resized & cropped to 299×299
- Output: one of 5 classes with softmax probabilities

## Project Structure
```
body-shape-api/
├── main.py                       # FastAPI application
├── best_body_shape_model.pth     # Trained model weights
├── venv/                         # Virtual environment (git ignored)
└── README.md                     # This file
```

## Quick Start

### 1. Prerequisites

- Python 3.8+ (Ubuntu usually has `python3`)
- Git (optional)

### 2. Setup

```bash
# Clone or enter your project folder
cd ~/Desktop/body-shape-api

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install fastapi uvicorn torch torchvision torchaudio pillow python-multipart
```

### 3. Run the API
```
# Start server with auto-reload (good for development)
uvicorn main:app --reload --port 8000
```
You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 4. Test the API
Open in your browser
Interactive Swagger UI (recommended for testing):
```
http://127.0.0.1:8000/docs
```

- **Click POST /predict**
- **Click Try it out**
- **Click Choose File → select a photo**
- **Click Execute**
- **See the JSON result**

Alternative health check
```
http://127.0.0.1:8000/health
```
Example successful response:
```
{
  "predicted_shape": "pear",
  "confidence": 0.9972,
  "probabilities": {
    "apple": 0.0008,
    "hourglass": 0.0011,
    "inverted_triangle": 0.0004,
    "pear": 0.9972,
    "rectangle": 0.0005
  },
  "filename": "photo.jpg"
}
```

### Development Notes

- Model file must be named exactly:  
  **`best_body_shape_model.pth`**

- If you move the model file → update the `MODEL_PATH` variable in `main.py`

- For production:
  - Remove `--reload`
  - Use **gunicorn + uvicorn workers**
  - Set `host="0.0.0.0"`

---

### Troubleshooting

| Problem | Solution |
|----------|----------|
| `python: command not found` | Use `python3` instead of `python` |
| Model file not found | Make sure `best_body_shape_model.pth` is in the project root |
| Slow prediction | This is normal on CPU. Install CUDA-enabled PyTorch if you have an NVIDIA GPU |
| Port 8000 already in use | Change port:<br>`uvicorn main:app --port 8001 --reload` |
| 422 Unprocessable Entity | Make sure you selected a file in Swagger |
| 500 Server Error | Check terminal logs (most common: wrong model path or incompatible torch version) |

---
