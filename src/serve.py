import io
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model

app = FastAPI(title="MLOps Model Serving")

MODEL_PATH = Path(os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

val_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

def load_checkpoint():
    global model
    if MODEL_PATH.exists() and model is None:
        model = get_model("resnet18", num_classes=10)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

@app.get("/health")
def health():
    load_checkpoint()
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint not found")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    load_checkpoint()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].tolist()
        
    return {"probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)}}