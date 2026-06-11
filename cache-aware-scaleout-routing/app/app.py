#!/usr/bin/env python3
"""
Cache-Aware Scale-Out Routing — FastAPI inference server.

Handlers are loaded from a PVC mount (HANDLER_BASE_PATH) keyed by version.
Each version is loaded once per pod lifetime via importlib and cached in
_module_cache. Subsequent requests for the same version pay zero import cost.

Handler path resolution:
  {HANDLER_BASE_PATH}/{handlerVersion}/predict_handler.py

Handlers are NOT bundled in the image. They must be present on the mounted PVC
before the service receives traffic. Run the handlers-loader Job first.
Default HANDLER_BASE_PATH = /mnt/handlers (PVC mount point).
"""

import importlib.util
import os
import time
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Cache-Aware Inference — Scale-Out", version="2.0.0")

HANDLER_BASE = Path(os.environ.get("HANDLER_BASE_PATH", "/mnt/handlers"))

_module_cache: dict = {}
_module_cache_lock = threading.Lock()


def _load_handler(version: str):
    with _module_cache_lock:
        if version in _module_cache:
            return _module_cache[version], False

    path = HANDLER_BASE / version / "predict_handler.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Handler not found: {path}. "
            f"Check HANDLER_BASE_PATH={HANDLER_BASE} and that version '{version}' is mounted."
        )

    spec = importlib.util.spec_from_file_location(f"predict_handler_{version}", str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with _module_cache_lock:
        _module_cache[version] = mod
    return mod, True


class PredictRequest(BaseModel):
    instanceId: str
    handlerVersion: str = "v1"


@app.get("/health")
@app.get("/inference/health")
def health():
    with _module_cache_lock:
        loaded_versions = list(_module_cache.keys())
    model_cache_size, pod_id = 0, None
    if loaded_versions:
        try:
            mod = _module_cache[loaded_versions[0]]
            model_cache_size = len(mod._MODEL_CACHE)
            pod_id = mod.POD_ID
        except Exception:
            pass
    return {
        "status":           "ready",
        "handler_base":     str(HANDLER_BASE),
        "loaded_versions":  loaded_versions,
        "model_cache_size": model_cache_size,
        "pod_id":           pod_id,
    }


@app.post("/inference/predict")
def predict(req: PredictRequest, request: Request):
    t_start = time.time()
    t_before_load = time.time()
    try:
        handler, first_load = _load_handler(req.handlerVersion)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load handler: {e}")
    import_overhead_s = round(time.time() - t_before_load, 3)

    try:
        result = handler.run(req.instanceId)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["timing"]["import_overhead_s"] = import_overhead_s
    result["timing"]["total_s"]           = round(time.time() - t_start, 3)
    result["first_load"]                  = first_load
    result["handler_version"]             = req.handlerVersion
    result["x_instance_id_header"]        = request.headers.get("x-instance-id", "")
    return result
