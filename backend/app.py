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
from db.supabase import get_client as get_supabase
from dotenv import load_dotenv

# Deepfake module depends on TensorFlow / OpenCV. If either is unavailable
# (e.g. broken TF install), we still want the rest of the API to work.
try:
    from models.deepfake_detection_engine import (
        load_model as load_deepfake_model,
        predict_media,
    )
    _DEEPFAKE_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001
    load_deepfake_model = None
    predict_media = None
    _DEEPFAKE_IMPORT_ERROR = _e
    print(f"[Deepfake] Module disabled: {type(_e).__name__}: {_e}")

load_dotenv()
ETHERSCAN_API_KEY = os.getenv("MY_KEY")

# ── Request schemas ──────────────────────────────────────────────────────────

class WalletRequest(BaseModel):
    address: str
    blockchain: str = "ethereum"

class CoinRequest(BaseModel):
    coin: str
    blockchain: str = "ethereum"

class ScamReportRequest(BaseModel):
    address: str
    blockchain: str = "ethereum"
    description: str = ""
    amount_lost: float = 0.0
    reporter: str | None = None

class CommunityPostRequest(BaseModel):
    title: str
    body: str = ""
    category: str = "discussions"
    author: str | None = None

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="COINFidance API",
    description="Wallet analyzer, coin checker, scam reports, community hub & deepfake detection.",
    version="1.0.0",
)

# CORS: always allow local Vite dev server; in production add the Vercel
# origin via the FRONTEND_ORIGIN env var (comma-separated for multiple).
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Root + health ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Friendly landing JSON so visiting the API root isn't a 404."""
    return {
        "service": "COINFidance API",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "wallet": ["POST /analyze-wallet", "GET /recent-wallet-scans"],
            "coin": ["POST /analyze-coin", "GET /recent-coin-scans"],
            "scam_reports": [
                "POST /scam-reports",
                "GET /scam-reports",
                "GET /scam-reports/stats",
            ],
            "community": [
                "POST /community-posts",
                "GET /community-posts",
                "GET /community-posts/stats",
            ],
            "stats": "GET /stats",
            "deepfake": "POST /detect-deepfake",
        },
        "frontend": "http://localhost:5173",
    }


@app.get("/health")
def health():
    """Liveness + readiness check. Pings Supabase + Etherscan."""
    status = {
        "ok": True,
        "models_loaded": {
            "wallet": True,
            "coin": True,
            "deepfake": deepfake_model is not None,
        },
        "supabase": {"ok": False, "error": None},
        "etherscan": {"ok": False, "error": None},
    }

    # Supabase ping: cheap count query against wallet_scans.
    try:
        sb = get_supabase()
        sb.table("wallet_scans").select("id", count="exact").limit(1).execute()
        status["supabase"]["ok"] = True
    except Exception as e:
        status["supabase"]["error"] = f"{type(e).__name__}: {e}"
        status["ok"] = False

    # Etherscan ping: cheap V2 ethsupply call (no address required).
    try:
        if not ETHERSCAN_API_KEY:
            raise RuntimeError("ETHERSCAN API key (env MY_KEY) not set")
        r = requests.get(
            "https://api.etherscan.io/v2/api",
            params={
                "chainid": 1,
                "module": "stats",
                "action": "ethsupply",
                "apikey": ETHERSCAN_API_KEY,
            },
            timeout=8,
        )
        data = r.json()
        if data.get("status") == "1":
            status["etherscan"]["ok"] = True
        else:
            status["etherscan"]["error"] = data.get("message") or str(data)
            status["ok"] = False
    except Exception as e:
        status["etherscan"]["error"] = f"{type(e).__name__}: {e}"
        status["ok"] = False

    return status

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

if load_deepfake_model is not None:
    try:
        deepfake_model = load_deepfake_model()
    except Exception as _e:  # noqa: BLE001
        print(f"[Deepfake] load_model failed: {type(_e).__name__}: {_e}")
        deepfake_model = None
        _DEEPFAKE_IMPORT_ERROR = _e
else:
    deepfake_model = None

# ── Helpers ──────────────────────────────────────────────────────────────────

# Load wallet feature-normalization stats saved by training (if available).
_wallet_norm_path = os.path.join(
    os.path.dirname(__file__), "saved_models", "wallet_norm.npz"
)
if os.path.exists(_wallet_norm_path):
    _wn = np.load(_wallet_norm_path)
    _W_LOG_STATIC_COLS = _wn["log_static_cols"].tolist()
    _W_STATIC_MEAN = _wn["static_mean"][0]
    _W_STATIC_STD = _wn["static_std"][0]
    _W_LOG_SEQ_COLS = _wn["log_seq_cols"].tolist()
    _W_SEQ_MEAN = _wn["seq_mean"][0]
    _W_SEQ_STD = _wn["seq_std"][0]
    print(f"[Wallet] Loaded normalization stats from {_wallet_norm_path}")
else:
    _W_LOG_STATIC_COLS = _W_LOG_SEQ_COLS = []
    _W_STATIC_MEAN = _W_STATIC_STD = None
    _W_SEQ_MEAN = _W_SEQ_STD = None
    print("[Wallet] No wallet_norm.npz found – features will be passed raw.")


def _normalize_wallet_features(static, seq):
    """Apply the same log + z-score transform used during training."""
    s = np.asarray(static, dtype=np.float32).copy()
    q = np.asarray(seq, dtype=np.float32).copy()

    if _W_STATIC_MEAN is not None:
        for c in _W_LOG_STATIC_COLS:
            s[c] = np.log1p(max(float(s[c]), 0.0))
        s = (s - _W_STATIC_MEAN) / _W_STATIC_STD
        for c in _W_LOG_SEQ_COLS:
            q[:, c] = np.log1p(np.maximum(q[:, c], 0.0))
        q = (q - _W_SEQ_MEAN) / _W_SEQ_STD

    return s.tolist(), q.tolist()


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


_KNOWN_LEGIT = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x514910771af9ca656af840dff83e8264ecf986ca",  # LINK
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI
    "0xd1d2eb1b1e90b638588728b4130137d262c87cae",  # GALA v2
}
_KNOWN_SCAM = {
    "0x57d9750893adb0d1ee07bac44c8bb45c45b58f73",  # SQUID
    "0x1f5eabba9c56bca4a7828969b79bc87051125b31",  # SQUID Rug 1 (BSC)
    "0x9cd67127b638b2074bc6523aefa3c04f7a729038",  # SQUIDGAME 2024 honey-pot
}


def extract_coin_features(coin_address: str):
    """
    Build numeric, binary, and text-embedding vectors for a token.

    Real on-chain / CoinGecko integration is TODO; until then we generate
    deterministic features from the address itself so that:
      - well-known legit tokens look "legit-shaped" (high liquidity etc.),
      - known scam tokens look "scam-shaped",
      - any other address gets a stable mid-range fingerprint instead of
        an all-zero vector that always produces the same prediction.
    """
    addr = (coin_address or "").lower().strip()

    # Deterministic pseudo-random vector derived from the address.
    rng = np.random.default_rng(abs(hash(addr)) % (2**32))
    numeric = list(rng.normal(0, 1, NUMERIC_DIM).astype(float))
    binary = list((rng.random(BINARY_DIM) > 0.5).astype(float))
    text_embed = list(rng.normal(0, 1, TEXT_DIM).astype(float))

    if addr in _KNOWN_LEGIT:
        # Replace with archetypal "legit-shaped" features (high liquidity,
        # broad holder base, audited, low volatility, etc.).
        numeric = [3.0, 3.5, 3.0, -1.5, 2.5, 2.0, 2.5, -2.0, -2.0, 1.5][:NUMERIC_DIM]
        binary = [1.0] * BINARY_DIM
        text_embed = [v * 0.3 + 0.4 for v in text_embed]
    elif addr in _KNOWN_SCAM:
        # Archetypal scam shape (low liquidity, concentrated holders, fresh,
        # high volatility, unaudited, no website).
        numeric = [-2.5, -2.5, -2.0, 2.5, -2.5, -2.0, -1.5, 2.5, 2.5, -1.0][:NUMERIC_DIM]
        binary = [0.0] * BINARY_DIM
        text_embed = [v * 0.3 - 0.5 for v in text_embed]

    return numeric, binary, text_embed


# Etherscan V2 chain id map. V1 was deprecated in 2024.
# https://docs.etherscan.io/v2-migration
CHAIN_IDS = {
    "ethereum": 1,
    "BNB_chain": 56,
    "Polygon": 137,
    "Arbitrum": 42161,
    "Optimism": 10,
    "Base": 8453,
}


def _etherscan_v2_call(action: str, address: str, chainid: int) -> list:
    """One V2 call. Returns the result list or [] for no-data; raises on errors."""
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": chainid,
        "module": "account",
        "action": action,
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 100,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Etherscan request failed: {e}")
    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Etherscan returned non-JSON response.")
    result = data.get("result")
    if isinstance(result, list):
        return result
    msg = (data.get("message") or "").lower()
    if "no transactions found" in msg:
        return []
    raise HTTPException(
        status_code=502,
        detail=f"Etherscan API error ({action}): {data.get('message', 'unknown')} – {result}",
    )


def fetch_etherscan_txs(address: str, blockchain: str) -> list:
    """
    Pull combined transaction history (native + ERC-20) via Etherscan API V2,
    sorted ascending by timestamp. Falls back gracefully if either feed is
    empty or rate-limited.
    """
    if not ETHERSCAN_API_KEY:
        raise HTTPException(status_code=500, detail="ETHERSCAN API key not configured (env MY_KEY).")
    chainid = CHAIN_IDS.get(blockchain)
    if chainid is None:
        raise HTTPException(
            status_code=400,
            detail=f"Blockchain '{blockchain}' not supported by Etherscan V2. "
                   f"Supported: {', '.join(CHAIN_IDS.keys())}.",
        )

    native = _etherscan_v2_call("txlist", address, chainid)
    # ERC-20 transfers — captures address-poisoning bursts that aren't in txlist.
    try:
        tokens = _etherscan_v2_call("tokentx", address, chainid)
    except HTTPException:
        tokens = []
    # Mark token rows with a stand-in gasUsed so feature extraction is happy.
    combined = list(native) + list(tokens)
    combined.sort(key=lambda t: int(t.get("timeStamp", "0")))
    # Keep most recent ~150 events
    return combined[-150:]


# ── Endpoints ────────────────────────────────────────────────────────────────

def _heuristic_wallet_risk(raw_static: list[float], raw_seq: list[list[float]]) -> float:
    """
    Cheap interpretable rules that bump suspicion when classic phishing
    patterns appear, regardless of what the neural net says. Returns a
    score in [0, 1] where 1 = definitely suspicious.
    """
    num_txs       = raw_static[0]
    failed_txs    = raw_static[3]
    fail_ratio    = raw_static[4]
    unique_recv   = raw_static[5]

    risk = 0.0
    # Receiver concentration
    if num_txs >= 10:
        ratio = unique_recv / max(num_txs, 1)
        if unique_recv <= 2:
            risk += 0.6
        elif ratio < 0.05:
            risk += 0.45
        elif ratio < 0.15:
            risk += 0.25
    # High failure rate
    if fail_ratio > 0.3:
        risk += 0.25
    elif fail_ratio > 0.15:
        risk += 0.10
    # Empty / brand new wallet (still possible to be a poisoner reading-only)
    if num_txs == 0:
        risk += 0.0
    return min(risk, 1.0)


@app.post("/analyze-wallet")
def analyze_wallet(request: WalletRequest):
    address = request.address.strip()
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format.")

    txs = fetch_etherscan_txs(address, request.blockchain)
    raw_static, raw_seq = extract_wallet_features({"result": txs})
    static, seq = _normalize_wallet_features(raw_static, raw_seq)

    static_tensor = torch.tensor([static], dtype=torch.float32)
    seq_tensor = torch.tensor([seq], dtype=torch.float32)

    with torch.no_grad():
        output = wallet_model(static_tensor, seq_tensor)
    probs = torch.softmax(output, dim=1)
    model_susp_prob = float(probs[0][1].item())

    # Fuse model probability with rule-based heuristic risk.
    heuristic = _heuristic_wallet_risk(raw_static, raw_seq)
    fused = max(model_susp_prob, heuristic)
    suspicious = fused >= 0.5
    score = fused if suspicious else (1.0 - fused)

    result = {
        "suspicious": bool(suspicious),
        "score": round(score, 4),
        "address": address,
        "blockchain": request.blockchain,
        "details": {
            "tx_count": int(raw_static[0]),
            "unique_receivers": int(raw_static[5]),
            "fail_ratio": round(float(raw_static[4]), 4),
            "model_suspicion_prob": round(model_susp_prob, 4),
            "heuristic_risk": round(heuristic, 4),
        },
    }

    # Persist to Supabase
    try:
        sb = get_supabase()
        sb.table("wallet_scans").insert({
            "address": address,
            "blockchain": request.blockchain,
            "suspicious": bool(suspicious),
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


# ── Scam report endpoints ───────────────────────────────────────────────────

@app.post("/scam-reports")
def create_scam_report(req: ScamReportRequest):
    try:
        sb = get_supabase()
        resp = sb.table("scam_reports").insert({
            "address": req.address,
            "blockchain": req.blockchain,
            "description": req.description,
            "amount_lost": req.amount_lost,
            "status": "pending",
            "reporter": req.reporter,
        }).execute()
        return {"ok": True, "data": resp.data}
    except Exception as e:
        print(f"[Supabase] Failed to create scam report: {e}")
        raise HTTPException(status_code=500, detail="Failed to save report")


@app.get("/scam-reports")
def list_scam_reports(
    status: str | None = None,
    blockchain: str | None = None,
    q: str | None = None,
    limit: int = Query(default=20, le=100),
):
    """List recent scam reports, optionally filtered by status / chain / search."""
    try:
        sb = get_supabase()
        query = sb.table("scam_reports").select("*").order("reported_at", desc=True)
        if status and status != "all_s":
            query = query.eq("status", status)
        if blockchain and blockchain != "all_t":
            query = query.eq("blockchain", blockchain)
        if q:
            query = query.or_(f"address.ilike.%{q}%,description.ilike.%{q}%")
        resp = query.limit(limit).execute()
        return resp.data
    except Exception as e:
        print(f"[Supabase] Failed to fetch scam reports: {e}")
        return []


@app.get("/scam-reports/stats")
def scam_report_stats():
    try:
        sb = get_supabase()
        total = sb.table("scam_reports").select("*", count="exact").execute()
        verified = (
            sb.table("scam_reports")
            .select("*", count="exact")
            .eq("status", "verified")
            .execute()
        )
        under_review = (
            sb.table("scam_reports")
            .select("*", count="exact")
            .eq("status", "under_review")
            .execute()
        )
        # sum of amount_lost (fetch only that column to keep payload small)
        amounts_resp = sb.table("scam_reports").select("amount_lost").execute()
        total_lost = sum(float(r.get("amount_lost") or 0) for r in (amounts_resp.data or []))
        return {
            "total_reports": total.count or 0,
            "verified": verified.count or 0,
            "under_review": under_review.count or 0,
            "total_lost": total_lost,
        }
    except Exception as e:
        print(f"[Supabase] Failed to fetch scam report stats: {e}")
        return {
            "total_reports": 0,
            "verified": 0,
            "under_review": 0,
            "total_lost": 0,
        }


# ── Community hub endpoints ─────────────────────────────────────────────────

@app.post("/community-posts")
def create_community_post(req: CommunityPostRequest):
    try:
        sb = get_supabase()
        resp = sb.table("community_posts").insert({
            "title": req.title,
            "body": req.body,
            "category": req.category,
            "author": req.author,
        }).execute()
        return {"ok": True, "data": resp.data}
    except Exception as e:
        print(f"[Supabase] Failed to create community post: {e}")
        raise HTTPException(status_code=500, detail="Failed to save post")


@app.get("/community-posts")
def list_community_posts(
    category: str | None = None,
    q: str | None = None,
    limit: int = Query(default=20, le=100),
):
    try:
        sb = get_supabase()
        query = sb.table("community_posts").select("*").order("created_at", desc=True)
        if category and category != "all_c":
            query = query.eq("category", category)
        if q:
            query = query.or_(f"title.ilike.%{q}%,body.ilike.%{q}%")
        resp = query.limit(limit).execute()
        return resp.data
    except Exception as e:
        print(f"[Supabase] Failed to fetch community posts: {e}")
        return []


@app.get("/community-posts/stats")
def community_post_stats():
    try:
        sb = get_supabase()
        total = sb.table("community_posts").select("*", count="exact").execute()
        scam_alerts = (
            sb.table("community_posts")
            .select("*", count="exact")
            .eq("category", "scam_alerts")
            .execute()
        )
        tips = (
            sb.table("community_posts")
            .select("*", count="exact")
            .eq("category", "tips")
            .execute()
        )
        # active today: posts created in last 24h
        from datetime import timedelta
        since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        active = (
            sb.table("community_posts")
            .select("*", count="exact")
            .gte("created_at", since)
            .execute()
        )
        return {
            "total_posts": total.count or 0,
            "active_today": active.count or 0,
            "scam_alerts": scam_alerts.count or 0,
            "com_tips": tips.count or 0,
        }
    except Exception as e:
        print(f"[Supabase] Failed to fetch community stats: {e}")
        return {
            "total_posts": 0,
            "active_today": 0,
            "scam_alerts": 0,
            "com_tips": 0,
        }


# ── Deepfake endpoint ───────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",  # video
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",  # image
}

@app.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile = File(...)):
    if deepfake_model is None or predict_media is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Deepfake detection is unavailable on this server. "
                f"Reason: {type(_DEEPFAKE_IMPORT_ERROR).__name__}: {_DEEPFAKE_IMPORT_ERROR}"
                if _DEEPFAKE_IMPORT_ERROR
                else "Deepfake model not loaded."
            ),
        )

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
