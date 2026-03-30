import torch
import os
import tempfile
import shutil
import requests
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.wallet_analyze_model import WalletClassifier
from models.coin_analyze_model import ScamCoinClassifier
from models.deepfake_detection_engine import load_model as load_deepfake_model, predict_media
from db.supabase import get_client as get_supabase
from dotenv import load_dotenv

load_dotenv()
ETHERSCAN_API_KEY = os.getenv("MY_KEY")

# ── Request schemas ──────────────────────────────────────────────────────────

class WalletRequest(BaseModel):
    address: str
    blockchain: str = "ethereum"

class CoinRequest(BaseModel):
    coin: str
    blockchain: str = "ethereum"

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load wallet model ────────────────────────────────────────────────────────

STATIC_DIM = 10
SEQ_DIM = 5
HIDDEN_DIM = 128
GRU_HIDDEN = 64
SEQ_LENGTH = 20

wallet_model = WalletClassifier(
    static_dim=STATIC_DIM,
    seq_dim=SEQ_DIM,
    hidden_dim=HIDDEN_DIM,
    gru_hidden=GRU_HIDDEN,
)

wallet_model_path = os.path.join(
    os.path.dirname(__file__), "saved_models", "wallet_classifier.pth"
)
if os.path.exists(wallet_model_path):
    wallet_model.load_state_dict(torch.load(wallet_model_path, map_location="cpu"))
wallet_model.eval()

# ── Load coin model ──────────────────────────────────────────────────────────

NUMERIC_DIM = 10
BINARY_DIM = 5
TEXT_DIM = 128
COIN_HIDDEN_DIM = 64

coin_model = ScamCoinClassifier(
    numeric_dim=NUMERIC_DIM,
    binary_dim=BINARY_DIM,
    text_dim=TEXT_DIM,
    hidden_dim=COIN_HIDDEN_DIM,
)

coin_model_path = os.path.join(
    os.path.dirname(__file__), "saved_models", "coin_classifier.pth"
)
if os.path.exists(coin_model_path):
    coin_model.load_state_dict(torch.load(coin_model_path, map_location="cpu"))
coin_model.eval()

# ── Load deepfake model ─────────────────────────────────────────────────────

deepfake_model = load_deepfake_model()

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_wallet_features(tx_data):
    """
    Convert Etherscan transaction JSON into static features and
    per-transaction sequence features matching the wallet model's expected inputs.
    """
    txs = tx_data.get("result", [])
    num_txs = len(txs)

    total_value = 0
    failed_txs = 0
    unique_receivers = set()
    gas_prices = []
    time_deltas = []

    prev_time = None
    for tx in txs:
        value = int(tx.get("value", "0"))
        total_value += value

        if tx.get("isError") == "1":
            failed_txs += 1

        unique_receivers.add(tx.get("to"))

        gas_price = int(tx.get("gasPrice", "0"))
        gas_prices.append(gas_price)

        ts = int(tx.get("timeStamp", "0"))
        if prev_time is not None:
            time_deltas.append(ts - prev_time)
        prev_time = ts

    avg_value = total_value / num_txs if num_txs > 0 else 0
    unique_receivers_count = len(unique_receivers)
    avg_gas_price = float(np.mean(gas_prices)) if gas_prices else 0
    std_gas_price = float(np.std(gas_prices)) if gas_prices else 0
    avg_time_delta = float(np.mean(time_deltas)) if time_deltas else 0
    min_time_delta = float(np.min(time_deltas)) if time_deltas else 0
    fail_ratio = failed_txs / num_txs if num_txs > 0 else 0

    static = [
        num_txs,
        total_value,
        avg_value,
        failed_txs,
        fail_ratio,
        unique_receivers_count,
        avg_gas_price,
        std_gas_price,
        avg_time_delta,
        min_time_delta,
    ]
    if len(static) < STATIC_DIM:
        static += [0.0] * (STATIC_DIM - len(static))
    else:
        static = static[:STATIC_DIM]

    seq = []
    for tx in txs[:SEQ_LENGTH]:
        row = [
            float(int(tx.get("value", "0"))),
            float(int(tx.get("gasPrice", "0"))),
            float(int(tx.get("gasUsed", "0"))),
            1.0 if tx.get("isError") == "1" else 0.0,
            float(int(tx.get("timeStamp", "0"))),
        ]
        if len(row) < SEQ_DIM:
            row += [0.0] * (SEQ_DIM - len(row))
        else:
            row = row[:SEQ_DIM]
        seq.append(row)

    while len(seq) < SEQ_LENGTH:
        seq.append([0.0] * SEQ_DIM)

    return static, seq


def extract_coin_features(coin_address: str):
    """
    Build numeric, binary, and text-embedding vectors for a token.
    Currently returns zero-padded placeholders -- replace with real
    on-chain / CoinGecko data when available.
    """
    numeric = [0.0] * NUMERIC_DIM
    binary = [0.0] * BINARY_DIM
    text_embed = [0.0] * TEXT_DIM
    return numeric, binary, text_embed


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/analyze-wallet")
def analyze_wallet(request: WalletRequest):
    address = request.address
    url = (
        f"https://api.etherscan.io/api?module=account&action=txlist"
        f"&address={address}&startblock=0&endblock=99999999"
        f"&sort=asc&apikey={ETHERSCAN_API_KEY}"
    )
    response = requests.get(url)
    tx_data = response.json()

    static, seq = extract_wallet_features(tx_data)

    static_tensor = torch.tensor([static], dtype=torch.float32)
    seq_tensor = torch.tensor([seq], dtype=torch.float32)

    with torch.no_grad():
        output = wallet_model(static_tensor, seq_tensor)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    score = probs[0][prediction].item()

    result = {
        "suspicious": bool(prediction),
        "score": round(score, 4),
        "address": address,
        "blockchain": request.blockchain,
    }

    # Persist to Supabase
    try:
        sb = get_supabase()
        sb.table("wallet_scans").insert({
            "address": address,
            "blockchain": request.blockchain,
            "suspicious": bool(prediction),
            "score": round(score, 4),
        }).execute()
    except Exception as e:
        print(f"[Supabase] Failed to save wallet scan: {e}")

    return result


@app.post("/analyze-coin")
def analyze_coin(request: CoinRequest):
    numeric, binary, text_embed = extract_coin_features(request.coin)

    numeric_t = torch.tensor([numeric], dtype=torch.float32)
    binary_t = torch.tensor([binary], dtype=torch.float32)
    text_t = torch.tensor([text_embed], dtype=torch.float32)

    with torch.no_grad():
        output = coin_model(numeric_t, binary_t, text_t)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    score = probs[0][prediction].item()

    labels = ["Legitimate", "Scam"]
    result = {
        "label": labels[prediction],
        "is_scam": bool(prediction),
        "score": round(score, 4),
        "coin": request.coin,
        "blockchain": request.blockchain,
    }

    # Persist to Supabase
    try:
        sb = get_supabase()
        sb.table("coin_scans").insert({
            "coin": request.coin,
            "blockchain": request.blockchain,
            "is_scam": bool(prediction),
            "label": labels[prediction],
            "score": round(score, 4),
        }).execute()
    except Exception as e:
        print(f"[Supabase] Failed to save coin scan: {e}")

    return result


# ── History / stats endpoints ────────────────────────────────────────────────

@app.get("/recent-wallet-scans")
def recent_wallet_scans(limit: int = Query(default=10, le=50)):
    """Return the most recent wallet scans, newest first."""
    try:
        sb = get_supabase()
        resp = (
            sb.table("wallet_scans")
            .select("*")
            .order("scanned_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data
    except Exception as e:
        print(f"[Supabase] Failed to fetch wallet scans: {e}")
        return []


@app.get("/recent-coin-scans")
def recent_coin_scans(limit: int = Query(default=10, le=50)):
    """Return the most recent coin scans, newest first."""
    try:
        sb = get_supabase()
        resp = (
            sb.table("coin_scans")
            .select("*")
            .order("scanned_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data
    except Exception as e:
        print(f"[Supabase] Failed to fetch coin scans: {e}")
        return []


@app.get("/stats")
def get_stats():
    """Return aggregate stats for the dashboard."""
    try:
        sb = get_supabase()

        wallet_resp = sb.table("wallet_scans").select("*", count="exact").execute()
        coin_resp = sb.table("coin_scans").select("*", count="exact").execute()

        total_scans = (wallet_resp.count or 0) + (coin_resp.count or 0)

        high_risk_wallets = (
            sb.table("wallet_scans")
            .select("*", count="exact")
            .eq("suspicious", True)
            .execute()
        )
        high_risk_coins = (
            sb.table("coin_scans")
            .select("*", count="exact")
            .eq("is_scam", True)
            .execute()
        )
        high_risk = (high_risk_wallets.count or 0) + (high_risk_coins.count or 0)

        return {
            "total_scans": total_scans,
            "wallet_scans": wallet_resp.count or 0,
            "coin_scans": coin_resp.count or 0,
            "high_risk_detected": high_risk,
        }
    except Exception as e:
        print(f"[Supabase] Failed to fetch stats: {e}")
        return {
            "total_scans": 0,
            "wallet_scans": 0,
            "coin_scans": 0,
            "high_risk_detected": 0,
        }


# ── Deepfake endpoint ───────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",  # video
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",  # image
}

@app.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Write upload to a temp file so OpenCV can read it
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename or "upload" + ext)
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = predict_media(deepfake_model, tmp_path)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
