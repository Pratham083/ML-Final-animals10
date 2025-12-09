import io
import json

import numpy as np
from PIL import Image

import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

with open("labels_it.json", "r") as f:
    LABELS_IT = json.load(f)

with open("labels_en.json", "r") as f:
    LABELS_EN = json.load(f)

# Italian -> English mapping
IT_TO_EN = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

def to_english(label: str) -> str:
    # If it's Italian, translate; if it's already English, return as is
    return IT_TO_EN.get(label, label)

SESSIONS = {
    "model1": {
        "session": ort.InferenceSession("model1_animals10_DA.onnx", providers=["CPUExecutionProvider"]),
        "labels": LABELS_IT,
    },
    "model2": {
        "session": ort.InferenceSession("model2_animals10_weighted.onnx", providers=["CPUExecutionProvider"]),
        "labels": LABELS_IT,
    },

    "model3": {
    "session": ort.InferenceSession("mobilenetv3_animals10.onnx",
                                    providers=["CPUExecutionProvider"]),
    "labels": LABELS_EN,
},
    "model4": {
        "session": ort.InferenceSession("convnet_scratch.onnx", providers=["CPUExecutionProvider"]),
        "labels": LABELS_EN,
    },
}

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def prepare_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # [1, 3, H, W]
    return x.numpy().astype("float32")

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(
    modelId: str = Form(...),
    image: UploadFile = File(...)
):
    if modelId not in SESSIONS:
        return {"error": f"Unknown modelId: {modelId}"}

    info = SESSIONS[modelId]
    session = info["session"]
    labels = info["labels"]

    file_bytes = await image.read()
    input_tensor = prepare_image(file_bytes)

    inputs = {session.get_inputs()[0].name: input_tensor}
    logits = session.run(None, inputs)[0]

    probs = softmax(logits)[0]
    top_idx = int(probs.argmax())
    raw_label = labels[top_idx]          # could be Italian or English
    top_label_en = to_english(raw_label) # guaranteed English

    # convert all probs to English keys
    all_probs = {}
    for i, p in enumerate(probs):
        raw = labels[i]
        en = to_english(raw)
        all_probs[en] = float(p)

    return {
        "topClass": top_label_en,         # always English
        "probability": float(probs[top_idx]),
        "allProbs": all_probs,            # dict with English keys
        "rawLabel": raw_label,            # optional: original label (for debugging/report)
    }
