"""
Cache-Aware Queue Inference — FastAPI inference server (async mode).

Flow:
  KafkaSource reads inference-requests topic
       ↓
  delivers as HTTP POST CloudEvent to this service
       ↓
  load handler from PVC (importlib cache — zero cost after first load)
       ↓
  check L1 model cache (TTLCache — zero S3 cost on hit)
       ↓
  run inference
       ↓
  write result to inference-results Kafka topic

The client sends to Kafka and reads results from Kafka.
The service never replies in the HTTP body — returns 202 Accepted.
"""

import importlib.util
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from kafka import KafkaProducer

app = FastAPI(title="Cache-Aware Queue Inference", version="1.0.0")

HANDLER_BASE   = Path(os.environ.get("HANDLER_BASE_PATH", "/mnt/handlers"))
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "inference-kafka-kafka-bootstrap.inference-benchmark.svc:9092")
RESULTS_TOPIC   = os.environ.get("KAFKA_RESULTS_TOPIC", "inference-results")

_module_cache: dict = {}
_module_cache_lock = threading.Lock()

_producer = None
_producer_lock = threading.Lock()


def _get_producer() -> KafkaProducer:
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=1,
                retries=3,
            )
    return _producer


def _load_handler(version: str):
    with _module_cache_lock:
        if version in _module_cache:
            return _module_cache[version], False

    path = HANDLER_BASE / version / "predict_handler.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Handler not found: {path}. "
            f"Check HANDLER_BASE_PATH={HANDLER_BASE} and that version '{version}' is on the PVC."
        )

    spec = importlib.util.spec_from_file_location(f"predict_handler_{version}", str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with _module_cache_lock:
        _module_cache[version] = mod
    return mod, True


@app.get("/health")
@app.get("/inference/health")
def health():
    with _module_cache_lock:
        loaded_versions = list(_module_cache.keys())
    return {
        "status":          "ready",
        "mode":            "queue-async",
        "handler_base":    str(HANDLER_BASE),
        "loaded_versions": loaded_versions,
        "kafka_bootstrap": KAFKA_BOOTSTRAP,
        "results_topic":   RESULTS_TOPIC,
    }


@app.post("/")
@app.post("/inference/predict")
async def predict(request: Request):
    """
    Accepts CloudEvents from KafkaSource.
    CloudEvent body is the raw Kafka message value (JSON).
    Expected payload: {"instanceId": "...", "handlerVersion": "v1"}
    Returns 202 Accepted immediately — result is written to Kafka results topic.
    """
    t_start = time.time()

    body = await request.json()

    # KafkaSource wraps the Kafka message value in CloudEvent data field
    # or delivers it directly depending on content-mode.
    # Support both patterns.
    if "data" in body and isinstance(body["data"], dict):
        payload = body["data"]
    else:
        payload = body

    instance_id     = payload.get("instanceId", "")
    handler_version = payload.get("handlerVersion", "v1")
    request_id      = request.headers.get("ce-id", "") or payload.get("requestId", "")

    if not instance_id:
        return Response(status_code=400, content="instanceId is required")

    try:
        handler, first_load = _load_handler(handler_version)
    except FileNotFoundError as e:
        error_result = {
            "requestId":      request_id,
            "instanceId":     instance_id,
            "handlerVersion": handler_version,
            "error":          str(e),
            "status":         "handler_not_found",
        }
        _get_producer().send(RESULTS_TOPIC, key=instance_id, value=error_result)
        return Response(status_code=202)

    try:
        result = handler.run(instance_id)
    except Exception as e:
        error_result = {
            "requestId":      request_id,
            "instanceId":     instance_id,
            "handlerVersion": handler_version,
            "error":          str(e),
            "status":         "inference_error",
        }
        _get_producer().send(RESULTS_TOPIC, key=instance_id, value=error_result)
        return Response(status_code=202)

    result["requestId"]       = request_id
    result["handlerVersion"]  = handler_version
    result["first_load"]      = first_load
    result["queue_overhead_s"] = round(time.time() - t_start - result["timing"]["total_s"], 3)
    result["status"]          = "ok"

    _get_producer().send(RESULTS_TOPIC, key=instance_id, value=result)

    return Response(status_code=202)
