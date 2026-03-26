from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()


model = torch.load("bone_model.pth")
model.eval()

CLASSES = [
    "Avulsion", "Comminuted", "Fracture-Dislocation", "Greenstick", 
    "Hairline", "Impacted", "Longitudinal", "Oblique", "Pathological", "Spiral"
]

@app.post("/predict")
async def predict_fracture(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return {
        "fracture_type": CLASSES[predicted.item()],
        "confidence": torch.nn.functional.softmax(outputs, dim=1).max().item()
    }