"""
Cache-Aware Inference Handler — L1 pod RAM only (no Redis).

On every request:
  1. Check L1 TTLCache (pod RAM) keyed by instanceId
     HIT  → run inference directly — all other steps = 0ms
     MISS → full pipeline:
       a. Registry lookup  — get zip S3 URI from Model Registry
       b. S3 download      — download single zip (one HTTP GET)
       c. Unzip            — extract 5 files in memory (no disk)
       d. Model load       — np.load + joblib.load × 4 → Python objects
       e. Cache store      — insert loaded objects into L1 LRU TTLCache
       f. Inference        — run prediction on loaded objects

Timing is captured for every step individually.
Cache stores loaded model objects — cache hit skips steps a-e entirely.
"""

import io
import os
import socket
import threading
import time
import zipfile

from cachetools import TTLCache

import boto3
import joblib
import numpy as np
import requests

# ── Pod identity ───────────────────────────────────────────────────────────────
POD_ID = socket.gethostname()

# ── Configuration ──────────────────────────────────────────────────────────────
REGISTRY_ENDPOINT = os.environ.get(
    "REGISTRY_ENDPOINT",
    "http://model-registry.inference-benchmark.svc.cluster.local:8080",
)
REGISTRY_API = f"{REGISTRY_ENDPOINT}/api/model_registry/v1alpha3"

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "inference-models")

MAX_CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE",   "30"))
CACHE_TTL_S    = int(os.environ.get("MODEL_CACHE_TTL_S",  "1200"))

# ── L1 cache — pod RAM, LRU+TTL ───────────────────────────────────────────────
# Stores fully loaded model objects keyed by instanceId.
# Cache hit = zero S3, zero unzip, zero model load.
_MODEL_CACHE: TTLCache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_S)
_cache_lock = threading.Lock()

# ── S3 client (created once per pod) ──────────────────────────────────────────
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
    return _s3_client


# ── Step 1: Registry lookup ────────────────────────────────────────────────────

def _fetch_zip_uri(instance_id: str) -> str:
    """
    Look up the S3 URI of the zip for this instanceId.
    Asset name in registry: asset_{instanceId}
    """
    asset_name = f"asset_{instance_id}"

    resp = requests.get(
        f"{REGISTRY_API}/registered_models",
        params={"pageSize": 1000},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    match = next((m for m in items if m["name"] == asset_name), None)
    if not match:
        raise ValueError(f"No asset found for {instance_id} (looked for '{asset_name}')")

    vr = requests.get(
        f"{REGISTRY_API}/registered_models/{match['id']}/versions",
        timeout=10,
    )
    vr.raise_for_status()
    versions = vr.json().get("items", [])
    if not versions:
        raise ValueError(f"No versions for {asset_name}")

    ar = requests.get(
        f"{REGISTRY_API}/model_versions/{versions[0]['id']}/artifacts",
        timeout=10,
    )
    ar.raise_for_status()
    artifacts = ar.json().get("items", [])
    if not artifacts:
        raise ValueError(f"No artifacts for {asset_name}")

    return artifacts[0]["uri"]


# ── Step 2: S3 download ────────────────────────────────────────────────────────

def _download_zip(uri: str) -> bytes:
    """Download the zip from MinIO. Returns raw bytes."""
    key = uri.replace(f"s3://{MINIO_BUCKET}/", "")
    buf = io.BytesIO()
    _get_s3().download_fileobj(MINIO_BUCKET, key, buf)
    return buf.getvalue()


# ── Step 3: Unzip ──────────────────────────────────────────────────────────────

def _unzip(zip_bytes: bytes) -> dict[str, bytes]:
    """Extract all files from zip bytes. Returns {filename: raw_bytes}."""
    files = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            files[name] = zf.read(name)
    return files


# ── Step 4: Model load ─────────────────────────────────────────────────────────

def _load_models(raw_files: dict[str, bytes]) -> dict:
    """
    Load each file into a Python object.
    model1_*.npz  → dict {"weights": np.array, "bias": np.array}
    model*_*.joblib → sklearn Pipeline object
    Returns {short_name: loaded_object}
    """
    models = {}
    for filename, data in raw_files.items():
        buf = io.BytesIO(data)
        if filename.endswith(".npz"):
            npz = np.load(buf, allow_pickle=False)
            models[filename] = {
                "weights": npz["weights"],
                "bias":    npz["bias"],
            }
        elif filename.endswith(".joblib"):
            models[filename] = joblib.load(buf)
    return models


# ── Step 5 (inline): Cache store ──────────────────────────────────────────────
# Measured in run() below with _cache_lock timing.

# ── Step 6: Inference ──────────────────────────────────────────────────────────

def _run_inference(models: dict) -> float:
    """
    Run lightweight inference on the loaded model objects.
    No I/O, no deserialization — operates on Python objects already in RAM.
    """
    dummy = np.random.randn(1, 100).astype(np.float32)
    scores = []

    for filename, obj in models.items():
        if filename.endswith(".npz"):
            W = obj["weights"]
            x = np.random.randn(W.shape[1]).astype(np.float32)
            scores.append(float(np.tanh(np.dot(W[0], x) / 100)))
        elif filename.endswith(".joblib"):
            clf = obj
            try:
                n_feat = clf.n_features_in_
                proba  = clf.predict_proba(dummy[:, :n_feat])
                scores.append(float(proba[0, 1]))
            except Exception:
                pred = clf.predict(dummy[:, :getattr(clf, "n_features_in_", 100)])
                scores.append(float(pred[0]))

    return round(float(np.mean(scores)), 4) if scores else 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

def run(instance_id: str) -> dict:
    t_total = time.perf_counter()

    registry_s    = 0.0
    s3_download_s = 0.0
    unzip_s       = 0.0
    model_load_s  = 0.0
    cache_store_s = 0.0
    zip_size_mb   = 0.0

    # ── L1 check ───────────────────────────────────────────────────────────────
    with _cache_lock:
        models = _MODEL_CACHE.get(instance_id)
    cache_hit = models is not None

    if not cache_hit:
        # Step 1 — Registry lookup
        t0 = time.perf_counter()
        zip_uri = _fetch_zip_uri(instance_id)
        registry_s = round(time.perf_counter() - t0, 3)

        # Step 2 — S3 download
        t0 = time.perf_counter()
        zip_bytes = _download_zip(zip_uri)
        s3_download_s = round(time.perf_counter() - t0, 3)
        zip_size_mb   = round(len(zip_bytes) / (1024 * 1024), 1)

        # Step 3 — Unzip
        t0 = time.perf_counter()
        raw_files = _unzip(zip_bytes)
        unzip_s = round(time.perf_counter() - t0, 3)

        # Step 4 — Load models
        t0 = time.perf_counter()
        models = _load_models(raw_files)
        model_load_s = round(time.perf_counter() - t0, 3)

        # Step 5 — Store in L1 LRU cache
        t0 = time.perf_counter()
        with _cache_lock:
            _MODEL_CACHE[instance_id] = models
        cache_store_s = round(time.perf_counter() - t0, 3)

    # Step 6 — Inference (always runs)
    t0 = time.perf_counter()
    prediction = _run_inference(models)
    inference_s = round(time.perf_counter() - t0, 3)

    total_s = round(time.perf_counter() - t_total, 3)

    with _cache_lock:
        cache_size = len(_MODEL_CACHE)

    return {
        "instanceId":   instance_id,
        "pod_id":       POD_ID,
        "prediction":   prediction,
        "cache_hit":    cache_hit,
        "cache_size":   cache_size,
        "zip_size_mb":  zip_size_mb,
        "timing": {
            "registry_lookup_s": registry_s,
            "s3_download_s":     s3_download_s,
            "unzip_s":           unzip_s,
            "model_load_s":      model_load_s,
            "cache_store_s":     cache_store_s,
            "inference_s":       inference_s,
            "total_s":           total_s,
        },
    }
