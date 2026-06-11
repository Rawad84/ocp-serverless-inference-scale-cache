#!/usr/bin/env python3
"""
Phase 1 — FastAPI inference server.

Every /inference/predict request spawns a fresh Python subprocess.
This replicates the customer's current architecture where the entire
Python environment (imports, model downloads) is paid on every request.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

POD_ID = os.environ.get("POD_NAME", socket.gethostname())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Phase 1 — Subprocess Inference", version="1.0.0")

HANDLER = str(Path(__file__).parent / "predict_handler.py")
PYTHON  = sys.executable


class PredictRequest(BaseModel):
    instanceId: str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/inference/health")
def health():
    return {
        "status": "ready",
        "phase": 1,
        "mode": "subprocess-per-request",
        "handler": HANDLER,
    }


# ── Predict ───────────────────────────────────────────────────────────────────

@app.post("/inference/predict")
def predict(req: PredictRequest):
    t0 = time.time()   # recorded BEFORE subprocess.run() — captures spawn + imports

    try:
        proc = subprocess.run(
            [
                PYTHON, HANDLER,
                "--instance-id", req.instanceId,
                "--start-time",  str(t0),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Subprocess timed out after 120s")

    total_s = round(time.time() - t0, 3)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error":  "subprocess failed",
                "stderr": proc.stderr[-2000:],   # last 2000 chars
                "returncode": proc.returncode,
            },
        )

    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail={"error": "handler returned invalid JSON", "stdout": proc.stdout[:500]},
        )

    # Add total as measured by app.py (subprocess spawn + imports + work)
    result["timing"]["total_s"] = total_s
    result["pod_id"] = POD_ID

    return result
