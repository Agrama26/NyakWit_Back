import io
import os
import logging
from pathlib import Path
import gdown

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Konfigurasi Logging ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Konstanta ────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "model_c/best_model_phase2.pth")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224
NUM_CLASSES = 5
DROPOUT     = 0.5

# Kelas dan metadata (urutan harus sama dengan class_to_idx saat training)
# Sesuaikan urutan ini dengan class_to_idx yang tercetak saat training!
CLASS_METADATA = {
    "Dryness": {
        "name_id"    : "Kekeringan (Dryness)",
        "severity"   : "medium",
        "description": "Daun mengering akibat kekurangan air atau gangguan fisiologis.",
        "treatment"  : [
            "Tingkatkan frekuensi penyiraman atau irigasi.",
            "Periksa sistem drainase lahan.",
            "Tambahkan mulsa untuk menjaga kelembapan tanah.",
            "Evaluasi kondisi iklim dan musim kemarau.",
        ],
        "prevention": [
            "Jadwalkan irigasi secara rutin.",
            "Monitor kelembapan tanah secara berkala.",
        ],
    },
    "Fungal_Disease": {
        "name_id"    : "Penyakit Jamur (Fungal Disease)",
        "severity"   : "high",
        "description": "Infeksi jamur yang menimbulkan bercak-bercak pada daun.",
        "treatment"  : [
            "Semprotkan fungisida berbahan aktif mancozeb atau tembaga.",
            "Pangkas dan musnahkan daun yang terinfeksi parah.",
            "Perbaiki sirkulasi udara di sekitar tanaman.",
            "Hindari pemupukan nitrogen berlebih.",
        ],
        "prevention": [
            "Rotasi fungisida agar tidak terjadi resistensi.",
            "Jaga kebersihan lahan dari sisa tanaman.",
        ],
    },
    "Healthy": {
        "name_id"    : "Sehat (Healthy)",
        "severity"   : "none",
        "description": "Daun dalam kondisi sehat, tidak terdeteksi penyakit.",
        "treatment"  : [
            "Pertahankan program pemupukan yang sudah berjalan.",
            "Lakukan monitoring rutin setiap 2 minggu.",
        ],
        "prevention": [
            "Jaga kebersihan dan sanitasi lahan secara rutin.",
            "Pantau hama dan penyakit sejak dini.",
        ],
    },
    "Magnesium_Deficiency": {
        "name_id"    : "Kekurangan Magnesium",
        "severity"   : "medium",
        "description": "Daun menguning (klorosis) akibat kekurangan unsur Magnesium.",
        "treatment"  : [
            "Aplikasikan pupuk Kieserit (MgSO4) sesuai dosis anjuran.",
            "Lakukan uji tanah untuk memastikan kadar Mg.",
            "Semprot daun dengan larutan MgSO4 2% sebagai penanganan cepat.",
            "Periksa pH tanah — Mg sulit diserap pada pH terlalu asam.",
        ],
        "prevention": [
            "Sertakan unsur Mg dalam program pemupukan rutin.",
            "Lakukan analisis daun dan tanah setiap tahun.",
        ],
    },
    "Scale_Insect": {
        "name_id"    : "Kutu Perisai (Scale Insect)",
        "severity"   : "high",
        "description": "Serangan hama kutu perisai yang mengisap cairan tanaman.",
        "treatment"  : [
            "Semprotkan insektisida sistemik (imidakloprid atau abamektin).",
            "Gunakan sabun insektisida atau minyak hortikultura.",
            "Perkenalkan musuh alami seperti predator Chilocorus.",
            "Pangkas bagian yang terserang berat.",
        ],
        "prevention": [
            "Inspeksi rutin terutama pada bagian bawah daun.",
            "Jaga kesehatan tanaman agar daya tahan optimal.",
        ],
    },
}

# ImageNet normalization (sesuai preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Inisialisasi FastAPI ─────────────────────────────────────────────────────
app = FastAPI(
    title       = "Nyakwit API",
    description = "API Klasifikasi Penyakit Daun Kelapa Sawit — ResNet-50",
    version     = "1.0.0",
)

# CORS — izinkan request dari React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Load Model ──────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> nn.Module:
    """Bangun arsitektur ResNet-50 + classifier head (sama persis saat training)."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(model_path: str) -> tuple[nn.Module, dict]:
    """Muat model dari checkpoint .pth dan kembalikan (model, class_to_idx)."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    checkpoint    = torch.load(model_path, map_location=DEVICE)
    model         = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint.get("class_to_idx", {})
    logger.info(f"Model dimuat dari: {model_path}")
    logger.info(f"  Epoch      : {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Val Acc    : {checkpoint.get('val_acc', 0)*100:.2f}%")
    logger.info(f"  Device     : {DEVICE}")
    logger.info(f"  class_to_idx: {class_to_idx}")
    return model, class_to_idx


# Muat saat startup
try:
    MODEL, CLASS_TO_IDX = load_model(MODEL_PATH)
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    logger.info("Model siap digunakan!")
except FileNotFoundError as e:
    logger.warning(f"Model tidak ditemukan: {e}")
    logger.warning("Server tetap berjalan — endpoint /predict akan mengembalikan error.")
    MODEL        = None
    CLASS_TO_IDX = {}
    IDX_TO_CLASS = {}

# ─── Transform ───────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─── Schema Response ─────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    success          : bool
    predicted        : str           # key kelas, contoh: "Fungal_Disease"
    predicted_name   : str           # nama Indonesia
    confidence       : float         # 0.0–1.0
    severity         : str           # "none" | "low" | "medium" | "high"
    description      : str
    treatment        : list[str]
    prevention       : list[str]
    probabilities    : dict[str, float]  # semua kelas beserta skornya
    device_used      : str


# ─── Endpoint ────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status"      : "ok",
        "model_loaded": MODEL is not None,
        "device"      : str(DEVICE),
        "classes"     : list(CLASS_TO_IDX.keys()),
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"key": k, **{f: v for f, v in meta.items()}}
            for k, meta in CLASS_METADATA.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # ── Validasi file ─────────────────────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG/PNG).")

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model belum dimuat. Pastikan file ada di: {MODEL_PATH}"
        )

    # ── Baca & preprocess gambar ──────────────────────────────────────
    try:
        contents  = await file.read()
        image     = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor    = preprocess(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca gambar: {str(e)}")

    # ── Inferensi ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs     = MODEL(tensor)
        probs       = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    # ── Petakan probabilitas ke nama kelas ────────────────────────────
    prob_dict = {}
    for idx, prob in enumerate(probs):
        class_name = IDX_TO_CLASS.get(idx, f"class_{idx}")
        prob_dict[class_name] = float(prob)

    predicted_key   = max(prob_dict, key=prob_dict.get)
    confidence      = prob_dict[predicted_key]
    meta            = CLASS_METADATA.get(predicted_key, {})

    logger.info(f"Prediksi: {predicted_key} ({confidence*100:.2f}%) | File: {file.filename}")

    return PredictionResponse(
        success        = True,
        predicted      = predicted_key,
        predicted_name = meta.get("name_id", predicted_key),
        confidence     = confidence,
        severity       = meta.get("severity", "unknown"),
        description    = meta.get("description", ""),
        treatment      = meta.get("treatment", []),
        prevention     = meta.get("prevention", []),
        probabilities  = prob_dict,
        device_used    = str(DEVICE),
    )