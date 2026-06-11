#!/usr/bin/env python3
"""
Phase 1 — subprocess predict handler.

All imports are at module level intentionally. This is the bottleneck:
Python re-imports every module from scratch on every subprocess invocation.
The import cost (5-15s) is what Phase 2 eliminates with importlib caching.

Called by app.py:
  python predict_handler.py --instance-id instance-001 --start-time 1234567890.123

Prints a single JSON object to stdout and exits.
"""

# ── Heavy imports at module level (this IS the bottleneck being measured) ─────
import time
import argparse
import json
import os
import sys
import io

import requests
import boto3
import numpy as np
import pandas as pd
import joblib
import scipy.stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Configuration (from environment, set by Knative Service manifest) ─────────

REGISTRY_ENDPOINT = os.environ.get(
    "REGISTRY_ENDPOINT",
    "http://model-registry.inference-benchmark.svc.cluster.local:8080",
)
REGISTRY_API = f"{REGISTRY_ENDPOINT}/api/model_registry/v1alpha3"

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "inference-models")

NUM_MODELS = 5   # model1…model5 per instanceId


# ── Registry helpers ──────────────────────────────────────────────────────────

def fetch_artifact_uris(instance_id: str) -> dict[str, str]:
    """
    Walk the 3-level Model Registry hierarchy for all 5 models of this instance.
    Returns {model_name: s3_uri}.
    This is the realistic slow path: fetch all 500 models, filter client-side.
    """
    resp = requests.get(
        f"{REGISTRY_API}/registered_models",
        params={"pageSize": 1000},
        timeout=15,
    )
    resp.raise_for_status()
    all_models = resp.json().get("items", [])

    # Filter to models belonging to this instanceId
    target_names = {f"model{n}_{instance_id}" for n in range(1, NUM_MODELS + 1)}
    matched = [m for m in all_models if m["name"] in target_names]

    uris = {}
    for model in matched:
        rm_id = model["id"]
        # Get version
        v_resp = requests.get(
            f"{REGISTRY_API}/registered_models/{rm_id}/versions",
            timeout=10,
        )
        v_resp.raise_for_status()
        versions = v_resp.json().get("items", [])
        if not versions:
            continue
        version_id = versions[0]["id"]

        # Get artifact URI
        a_resp = requests.get(
            f"{REGISTRY_API}/model_versions/{version_id}/artifacts",
            timeout=10,
        )
        a_resp.raise_for_status()
        artifacts = a_resp.json().get("items", [])
        if artifacts:
            uris[model["name"]] = artifacts[0]["uri"]

    return uris


# ── S3 helpers ────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )


def download_model(s3, uri: str) -> bytes:
    """Download a model file from S3. URI format: s3://bucket/key"""
    path = uri.replace(f"s3://{MINIO_BUCKET}/", "")
    buf = io.BytesIO()
    s3.download_fileobj(MINIO_BUCKET, path, buf)
    return buf.getvalue()


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(models_data: dict[str, bytes], instance_id: str) -> float:
    """
    Load each model from bytes and run a dummy prediction.
    Returns a single float score (average of all model outputs).
    """
    scores = []
    dummy_input = np.random.randn(1, 100).astype(np.float32)

    for name, data in models_data.items():
        if name.startswith("model1_"):
            # numpy weight matrix — simulate forward pass
            arr = np.load(io.BytesIO(data), allow_pickle=False)
            W = arr["weights"]   # (1600, 1600)
            x = np.random.randn(1600).astype(np.float32)
            score = float(np.dot(W[0], x))
            scores.append(np.tanh(score / 100))   # squash to [-1, 1]
        else:
            # sklearn pipeline
            clf = joblib.load(io.BytesIO(data))
            try:
                proba = clf.predict_proba(dummy_input[:, :clf.n_features_in_])
                scores.append(float(proba[0, 1]))
            except Exception:
                # Fall back to predict() if version skew breaks predict_proba
                pred = clf.predict(dummy_input[:, :clf.n_features_in_])
                scores.append(float(pred[0]))

    return round(float(np.mean(scores)), 4) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--start-time",  type=float, required=True,
                        help="time.time() recorded by app.py before subprocess.run()")
    args = parser.parse_args()

    # First line of main() — all module-level imports have already run.
    # The gap between --start-time and now IS the subprocess spawn + import cost.
    t_after_imports = time.time()
    import_overhead_s = round(t_after_imports - args.start_time, 3)

    s3 = get_s3_client()

    # ── Registry lookup ───────────────────────────────────────────────────────
    t0 = time.time()
    uris = fetch_artifact_uris(args.instance_id)
    registry_lookup_s = round(time.time() - t0, 3)

    if not uris:
        print(json.dumps({"error": f"No models found for {args.instance_id}"}))
        sys.exit(1)

    # ── S3 download ───────────────────────────────────────────────────────────
    t0 = time.time()
    models_data = {}
    for name, uri in uris.items():
        models_data[name] = download_model(s3, uri)
    s3_download_s = round(time.time() - t0, 3)

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.time()
    prediction = run_inference(models_data, args.instance_id)
    inference_s = round(time.time() - t0, 3)

    # ── Output (app.py reads this from stdout) ────────────────────────────────
    print(json.dumps({
        "instanceId": args.instance_id,
        "prediction": prediction,
        "models_loaded": len(models_data),
        "timing": {
            "import_overhead_s": import_overhead_s,
            "registry_lookup_s": registry_lookup_s,
            "s3_download_s":     s3_download_s,
            "inference_s":       inference_s,
        },
    }))


if __name__ == "__main__":
    main()
