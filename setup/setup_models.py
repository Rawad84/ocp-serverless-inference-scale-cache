#!/usr/bin/env python3
"""
setup_models.py  — ZIP asset variant

Generates 100 instanceIds. Each instanceId produces ONE zip file containing
5 model files, mirroring the actual customer asset structure:

  {instanceId}.zip
    ├── model1_{instanceId}.npz     ~30 MB  (numpy weight matrix, PyTorch sim)
    ├── model2_{instanceId}.joblib  ~3-5 MB (sklearn LogisticRegression pipeline)
    ├── model3_{instanceId}.joblib  ~3-5 MB (sklearn RandomForest pipeline)
    ├── model4_{instanceId}.joblib  ~4-6 MB (sklearn GradientBoosting pipeline)
    └── model5_{instanceId}.joblib  ~3-5 MB (sklearn RandomForest pipeline)

  Target zip size: ~45-55 MB per instanceId  (avg 50 MB, max ~100 MB)

MinIO layout (100 objects total, not 500):
  inference-models/{instanceId}/{instanceId}.zip

Model Registry (100 registrations total, not 500):
  RegisteredModel  → name: asset_{instanceId}
  ModelVersion     → v1
  ModelArtifact    → uri: s3://inference-models/{instanceId}/{instanceId}.zip
                      custom property: instance_id, file_count=5

Usage:
  python3 setup_models.py                    # 100 instances, in-cluster mode
  python3 setup_models.py --instances 5      # quick smoke test
  python3 setup_models.py --skip-registry    # MinIO upload only
"""

import argparse
import io
import os
import socket
import subprocess
import sys
import time
import zipfile

import boto3
import joblib
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────

IN_CLUSTER = os.environ.get("IN_CLUSTER", "false").lower() in ("true", "1", "yes")

MINIO_LOCAL_PORT    = 19000
REGISTRY_LOCAL_PORT = 18080

_default_minio_ep    = ("http://minio.minio.svc.cluster.local:9000"
                        if IN_CLUSTER else f"http://localhost:{MINIO_LOCAL_PORT}")
_default_registry_ep = ("http://model-registry.inference-benchmark.svc.cluster.local:8080"
                        if IN_CLUSTER else f"http://localhost:{REGISTRY_LOCAL_PORT}")

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   _default_minio_ep)
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "inference-models")

REGISTRY_ENDPOINT = os.environ.get("REGISTRY_ENDPOINT", _default_registry_ep)
REGISTRY_API      = f"{REGISTRY_ENDPOINT}/api/model_registry/v1alpha3"

INSTANCE_PREFIX = "instance"
NUM_FEATURES    = 100
NUM_SAMPLES     = 3000

# model1: numpy weight matrix — target ~40 MB as float32
# 3200 × 3200 × 4 bytes = ~41 MB uncompressed
# Random float data has high entropy — barely shrinks inside zip
MODEL1_ROWS = 3200
MODEL1_COLS = 3200

LOCAL_INSTANCE_LIMIT = 3


# ── Port-forward helpers ───────────────────────────────────────────────────────

def start_port_forward(namespace, service, local_port, remote_port):
    return subprocess.Popen(
        ["oc", "port-forward", f"svc/{service}",
         f"{local_port}:{remote_port}", "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


# ── Model generation ───────────────────────────────────────────────────────────

def generate_model1_bytes(instance_id: str, seed: int) -> bytes:
    """
    Numpy weight matrix — simulates a PyTorch model checkpoint.
    Unique weights per instanceId via seed.
    Saved as .npz (uncompressed). Target ~30 MB.
    """
    rng = np.random.RandomState(seed)
    weights = rng.randn(MODEL1_ROWS, MODEL1_COLS).astype(np.float32)
    bias    = rng.randn(MODEL1_ROWS).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, weights=weights, bias=bias,
             instance_id=np.array(instance_id))
    return buf.getvalue()


def generate_sklearn_bytes(model_num: int, seed: int) -> bytes:
    """
    Train a sklearn pipeline with enough complexity to produce a realistic
    file size (~3-6 MB). Unique coefficients per seed.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
    y = ((X @ rng.randn(NUM_FEATURES)) > 0).astype(int)

    if model_num == 2:
        # RandomForest instead of LogReg — LR produces a near-zero file
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=80,
                max_depth=int(rng.randint(5, 10)),
                random_state=seed,
            )),
        ])
    elif model_num == 3:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=120,
                max_depth=int(rng.randint(6, 12)),
                random_state=seed,
            )),
        ])
    elif model_num == 4:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=float(rng.uniform(0.05, 0.15)),
                max_depth=int(rng.randint(4, 7)),
                random_state=seed,
            )),
        ])
    else:  # model_num == 5
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_features=int(rng.randint(10, 30)),
                random_state=seed,
            )),
        ])

    clf.fit(X, y)
    buf = io.BytesIO()
    joblib.dump(clf, buf, compress=0)
    return buf.getvalue()


# ── Zip packaging ──────────────────────────────────────────────────────────────

def build_zip(instance_id: str, seed: int) -> tuple[bytes, dict[str, float]]:
    """
    Generate all 5 model files and pack them into a single in-memory zip.
    Returns (zip_bytes, {filename: size_mb}).
    """
    zip_buf  = io.BytesIO()
    sizes_mb = {}

    with zipfile.ZipFile(zip_buf, mode="w",
                         compression=zipfile.ZIP_STORED) as zf:
        # model1 — numpy/PyTorch sim (~30 MB)
        name1  = f"model1_{instance_id}.npz"
        data1  = generate_model1_bytes(instance_id, seed)
        zf.writestr(name1, data1)
        sizes_mb[name1] = len(data1) / (1024 * 1024)

        # model2-5 — sklearn pipelines (~3-6 MB each)
        for model_num in range(2, 6):
            name  = f"model{model_num}_{instance_id}.joblib"
            data  = generate_sklearn_bytes(model_num, seed + model_num * 13)
            zf.writestr(name, data)
            sizes_mb[name] = len(data) / (1024 * 1024)

    return zip_buf.getvalue(), sizes_mb


# ── MinIO upload ───────────────────────────────────────────────────────────────

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )


def upload_zip(s3, instance_id: str, zip_bytes: bytes) -> tuple[str, float]:
    """Upload zip to MinIO. Returns (s3_uri, size_mb)."""
    key = f"{instance_id}/{instance_id}.zip"
    s3.put_object(Bucket=MINIO_BUCKET, Key=key, Body=zip_bytes)
    return f"s3://{MINIO_BUCKET}/{key}", len(zip_bytes) / (1024 * 1024)


# ── Model Registry ─────────────────────────────────────────────────────────────

def register_asset(instance_id: str, s3_uri: str, zip_size_mb: float,
                   file_sizes: dict[str, float]) -> str:
    """
    Register one asset (one zip) per instanceId.
    Hierarchy: RegisteredModel → ModelVersion → ModelArtifact (S3 URI).
    Returns artifact id.
    """
    asset_name = f"asset_{instance_id}"

    # 1. RegisteredModel
    resp = requests.post(
        f"{REGISTRY_API}/registered_models",
        json={
            "name": asset_name,
            "description": f"Zip asset for {instance_id} — 5 models inside",
            "customProperties": {
                "instance_id": {
                    "string_value": instance_id,
                    "metadataType": "MetadataStringValue",
                },
                "asset_format": {
                    "string_value": "zip",
                    "metadataType": "MetadataStringValue",
                },
                "file_count": {
                    "string_value": "5",
                    "metadataType": "MetadataStringValue",
                },
                "zip_size_mb": {
                    "string_value": f"{zip_size_mb:.1f}",
                    "metadataType": "MetadataStringValue",
                },
            },
        },
        timeout=10,
    )
    if resp.status_code == 409:
        all_r = requests.get(f"{REGISTRY_API}/registered_models",
                             params={"pageSize": 1000}, timeout=10)
        all_r.raise_for_status()
        match = next((m for m in all_r.json().get("items", [])
                      if m["name"] == asset_name), None)
        if not match:
            raise RuntimeError(f"409 on create but {asset_name} not found")
        rm_id = match["id"]
    else:
        resp.raise_for_status()
        rm_id = resp.json()["id"]

    # 2. ModelVersion
    resp = requests.post(
        f"{REGISTRY_API}/model_versions",
        json={
            "name": "v1",
            "description": "Initial version",
            "registeredModelId": rm_id,
            "customProperties": {},
        },
        timeout=10,
    )
    if resp.status_code == 409:
        vr = requests.get(f"{REGISTRY_API}/registered_models/{rm_id}/versions",
                          timeout=10)
        vr.raise_for_status()
        versions = vr.json().get("items", [])
        if not versions:
            raise RuntimeError(f"409 on version but no versions for {asset_name}")
        version_id = versions[0]["id"]
    else:
        resp.raise_for_status()
        version_id = resp.json()["id"]

    # 3. ModelArtifact — stores the S3 URI of the zip
    resp = requests.post(
        f"{REGISTRY_API}/model_versions/{version_id}/artifacts",
        json={
            "name": asset_name,
            "description": f"Zip artifact — {zip_size_mb:.1f} MB, 5 files",
            "uri": s3_uri,
            "artifactType": "model-artifact",
            "customProperties": {
                "instance_id": {
                    "string_value": instance_id,
                    "metadataType": "MetadataStringValue",
                },
                "contains": {
                    "string_value": ",".join(file_sizes.keys()),
                    "metadataType": "MetadataStringValue",
                },
            },
        },
        timeout=10,
    )
    if resp.status_code == 409:
        ar = requests.get(f"{REGISTRY_API}/model_versions/{version_id}/artifacts",
                          timeout=10)
        ar.raise_for_status()
        arts = ar.json().get("items", [])
        return arts[0]["id"] if arts else "existing"
    resp.raise_for_status()
    return resp.json()["id"]


def lookup_asset_uri(instance_id: str) -> str | None:
    """Look up the S3 URI for an instanceId via the registry."""
    asset_name = f"asset_{instance_id}"
    resp = requests.get(f"{REGISTRY_API}/registered_models",
                        params={"pageSize": 1000}, timeout=10)
    resp.raise_for_status()
    match = next((m for m in resp.json().get("items", [])
                  if m["name"] == asset_name), None)
    if not match:
        return None
    vr = requests.get(
        f"{REGISTRY_API}/registered_models/{match['id']}/versions", timeout=10)
    vr.raise_for_status()
    versions = vr.json().get("items", [])
    if not versions:
        return None
    ar = requests.get(
        f"{REGISTRY_API}/model_versions/{versions[0]['id']}/artifacts", timeout=10)
    ar.raise_for_status()
    arts = ar.json().get("items", [])
    return arts[0]["uri"] if arts else None


# ── Progress ───────────────────────────────────────────────────────────────────

def progress(current: int, total: int, instance_id: str,
             total_mb: float, elapsed: float):
    pct = int((current / total) * 40)
    bar = "█" * pct + "░" * (40 - pct)
    print(
        f"\r  [{bar}] {current:>3}/{total}  {instance_id}  "
        f"{total_mb:>7.1f} MB  {elapsed:>6.1f}s",
        end="", flush=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 100 zip model assets and upload to MinIO + Model Registry")
    parser.add_argument("--instances",     type=int, default=100)
    parser.add_argument("--skip-registry", action="store_true",
                        help="Upload to MinIO only, skip registry registration")
    parser.add_argument("--force-local",   action="store_true",
                        help="Allow large local runs (not recommended)")
    args = parser.parse_args()

    if not IN_CLUSTER and args.instances > LOCAL_INSTANCE_LIMIT and not args.force_local:
        print("ERROR: Refusing to run a large setup locally.")
        print(f"  Requested: {args.instances} instances × ~50 MB = ~{args.instances*50} MB")
        print(f"  Safe local limit: {LOCAL_INSTANCE_LIMIT}")
        print()
        print("Run in-cluster instead:")
        print("  oc apply -f setup-zip/setup-job.yaml -n inference-benchmark")
        print()
        print("Quick local smoke-test (3 instances):")
        print("  python3 setup_models.py --instances 3")
        sys.exit(1)

    n = args.instances

    print("=" * 70)
    print("  INFERENCE BENCHMARK — ZIP MODEL ASSET SETUP")
    print(f"  {n} instanceIds × 1 zip each = {n} objects in MinIO")
    print(f"  Each zip: 5 files  (~30 MB model1.npz + 4×~4 MB .joblib)")
    print(f"  Target zip size: ~45-55 MB  |  Total: ~{n*50} MB")
    print("=" * 70)

    pf_minio = pf_registry = None

    if IN_CLUSTER:
        print(f"\n[1/5] In-cluster mode")
        print(f"  MinIO:    {MINIO_ENDPOINT}")
        print(f"  Registry: {REGISTRY_ENDPOINT}")
    else:
        print(f"\n[1/5] Starting port-forwards...")
        pf_minio    = start_port_forward("minio", "minio", MINIO_LOCAL_PORT, 9000)
        pf_registry = start_port_forward(
            "inference-benchmark", "model-registry", REGISTRY_LOCAL_PORT, 8080)

        if not wait_for_port(MINIO_LOCAL_PORT, timeout=30):
            print(f"ERROR: MinIO port-forward not ready on port {MINIO_LOCAL_PORT}")
            sys.exit(1)
        print(f"  MinIO ready    localhost:{MINIO_LOCAL_PORT}")

        if not wait_for_port(REGISTRY_LOCAL_PORT, timeout=30):
            print(f"ERROR: Registry port-forward not ready on port {REGISTRY_LOCAL_PORT}")
            sys.exit(1)
        print(f"  Registry ready localhost:{REGISTRY_LOCAL_PORT}")

    s3 = get_s3()

    try:
        # ── Connectivity check ─────────────────────────────────────────────────
        print("\n[2/5] Verifying connectivity...")
        try:
            s3.head_bucket(Bucket=MINIO_BUCKET)
            print(f"  MinIO bucket '{MINIO_BUCKET}' accessible")
        except Exception as e:
            print(f"  ERROR: MinIO — {e}")
            sys.exit(1)

        try:
            resp = requests.get(f"{REGISTRY_API}/registered_models", timeout=5)
            existing = resp.json().get("size", 0)
            print(f"  Model Registry accessible ({existing} existing registrations)")
        except Exception as e:
            print(f"  ERROR: Registry — {e}")
            sys.exit(1)

        # ── Generate, pack, upload ─────────────────────────────────────────────
        print(f"\n[3/5] Generating {n} zip assets...")
        print(f"  model1: numpy {MODEL1_ROWS}×{MODEL1_COLS} float32"
              f"  (~{MODEL1_ROWS*MODEL1_COLS*4/(1024**2):.0f} MB)")
        print(f"  model2-5: sklearn pipelines (RF×3, GBM×1) ~3-6 MB each")
        print()

        total_mb  = 0.0
        run_start = time.time()

        for i in range(1, n + 1):
            instance_id = f"{INSTANCE_PREFIX}-{i:03d}"
            seed        = i * 137

            # Build zip in memory
            zip_bytes, file_sizes = build_zip(instance_id, seed)
            zip_mb = len(zip_bytes) / (1024 * 1024)

            # Upload single zip to MinIO
            s3_uri, _ = upload_zip(s3, instance_id, zip_bytes)
            total_mb += zip_mb

            # Register one asset in Model Registry
            if not args.skip_registry:
                register_asset(instance_id, s3_uri, zip_mb, file_sizes)

            progress(i, n, instance_id, total_mb, time.time() - run_start)

        print()

        # ── Verify MinIO ───────────────────────────────────────────────────────
        print("\n[4/5] Verifying MinIO upload...")
        paginator = s3.get_paginator("list_objects_v2")
        total_objects = sum(
            page.get("KeyCount", 0)
            for page in paginator.paginate(Bucket=MINIO_BUCKET)
        )
        print(f"  Objects in bucket: {total_objects}  (expected {n})")
        print(f"  Status: {'PASS' if total_objects >= n else 'WARN'}")

        # Size of first zip as sanity check
        head = s3.head_object(Bucket=MINIO_BUCKET,
                              Key=f"instance-001/instance-001.zip")
        actual_mb = head["ContentLength"] / (1024 * 1024)
        print(f"  instance-001.zip size: {actual_mb:.1f} MB")

        # ── Verify Registry ────────────────────────────────────────────────────
        if not args.skip_registry:
            print("\n[5/5] Verifying Model Registry...")
            resp = requests.get(f"{REGISTRY_API}/registered_models",
                                params={"pageSize": 1000}, timeout=10)
            total_reg = resp.json().get("size", 0)
            print(f"  Registered assets: {total_reg}  (expected {n})")
            print(f"  Status: {'PASS' if total_reg >= n else 'WARN'}")

            print("\n  Spot-check: looking up URI for instance-001...")
            uri = lookup_asset_uri("instance-001")
            print(f"  URI: {uri}" if uri else "  WARN: not found")

        elapsed = time.time() - run_start
        print("\n" + "=" * 70)
        print("  SETUP COMPLETE")
        print(f"  Instances:   {n}")
        print(f"  MinIO objs:  {n} zip files")
        print(f"  Total data:  {total_mb:.1f} MB  ({total_mb/1024:.2f} GB)")
        print(f"  Avg zip:     {total_mb/n:.1f} MB")
        print(f"  Duration:    {elapsed:.0f}s  ({elapsed/60:.1f} min)")
        print("=" * 70)

    finally:
        if pf_minio:
            pf_minio.terminate()
        if pf_registry:
            pf_registry.terminate()


if __name__ == "__main__":
    main()
