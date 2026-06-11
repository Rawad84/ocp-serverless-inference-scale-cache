"""
End-to-end queue inference test.

Sends N messages to inference-requests Kafka topic.
Reads results from inference-results Kafka topic.
Reports per-request latency, cache hit rate, and pods served.

Usage (run inside cluster via oc exec):
  oc exec -n inference-benchmark <pod> -c user-container -- \
    python3 /tmp/test_e2e_queue.py

Or copy to pod first:
  oc cp tests/test_e2e_queue.py \
    inference-benchmark/<pod>:/tmp/test_e2e_queue.py -c user-container

Arguments (env vars):
  BOOTSTRAP        Kafka bootstrap (default: inference-kafka-kafka-bootstrap...)
  INSTANCES        Number of unique instanceIds (default: 10)
  REPEATS          Requests per instanceId (default: 3)
  HANDLER_VERSION  Handler version to use (default: v1)
"""

import json
import os
import time
import uuid

from kafka import KafkaConsumer, KafkaProducer, TopicPartition

BOOTSTRAP = os.environ.get(
    "BOOTSTRAP",
    "inference-kafka-kafka-bootstrap.inference-benchmark.svc:9092",
)
REQUEST_TOPIC   = "inference-requests"
RESULT_TOPIC    = "inference-results"
N_INSTANCES     = int(os.environ.get("INSTANCES", "10"))
REPEATS         = int(os.environ.get("REPEATS", "3"))
HANDLER_VERSION = os.environ.get("HANDLER_VERSION", "v1")
RESULT_TIMEOUT  = int(os.environ.get("RESULT_TIMEOUT_MS", "120000"))
N_PARTITIONS    = 3


def run():
    print(f"Queue E2E Test")
    print(f"  Bootstrap      : {BOOTSTRAP}")
    print(f"  Instances      : {N_INSTANCES}  x{REPEATS} repeats = {N_INSTANCES * REPEATS} messages")
    print(f"  Handler        : {HANDLER_VERSION}")
    print(f"  Result timeout : {RESULT_TIMEOUT}ms")
    print()

    # ── Step 1: Start consumer and seek to end BEFORE sending ────────────────
    # Using manual assign + seek_to_end avoids group coordinator lag and
    # guarantees we are positioned at the exact current end of the topic.
    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP,
        group_id=f"e2e-test-{uuid.uuid4().hex[:8]}",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode()),
        consumer_timeout_ms=RESULT_TIMEOUT,
    )
    tps = [TopicPartition(RESULT_TOPIC, i) for i in range(N_PARTITIONS)]
    consumer.assign(tps)
    consumer.seek_to_end(*tps)
    start_positions = {tp: consumer.position(tp) for tp in tps}
    print(f"Consumer seeked to end: {start_positions}")

    # ── Step 2: Send all messages ─────────────────────────────────────────────
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
        key_serializer=lambda k: k.encode(),
        acks=1,
    )
    instance_ids = [f"instance-{i:03d}" for i in range(1, N_INSTANCES + 1)
                    for _ in range(REPEATS)]
    send_times: dict[str, float] = {}

    print(f"Sending {len(instance_ids)} messages to {REQUEST_TOPIC}...")
    for iid in instance_ids:
        rid = str(uuid.uuid4())
        send_times[rid] = time.time()
        producer.send(REQUEST_TOPIC, key=iid,
                      value={"requestId": rid, "instanceId": iid,
                             "handlerVersion": HANDLER_VERSION})
    producer.flush()
    print("All sent. Waiting for results...")
    print()

    # ── Step 3: Collect results ───────────────────────────────────────────────
    received = 0
    results  = []

    for msg in consumer:
        r   = msg.value
        rid = r.get("requestId", "")
        if rid not in send_times:
            continue

        e2e = round(time.time() - send_times[rid], 3)
        t   = r.get("timing", {})
        row = {
            "instanceId":     r.get("instanceId", ""),
            "cache_hit":      r.get("cache_hit", False),
            "status":         r.get("status", "unknown"),
            "e2e_s":          e2e,
            "s3_download_s":  t.get("s3_download_s", 0),
            "model_load_s":   t.get("model_load_s", 0),
            "inference_s":    t.get("inference_s", 0),
            "total_s":        t.get("total_s", 0),
            "pod":            r.get("pod_id", "")[-8:],
        }
        results.append(row)
        received += 1

        h = "HIT " if row["cache_hit"] else "MISS"
        print(f"  [{h}] {row['instanceId']:<20} e2e={e2e:.3f}s  "
              f"inf={row['inference_s']:.3f}s  pod={row['pod']}  {row['status']}")

        if received >= len(instance_ids):
            break

    consumer.close()
    producer.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    hits   = [r for r in results if r["cache_hit"]]
    misses = [r for r in results if not r["cache_hit"]]
    errors = [r for r in results if r["status"] != "ok"]
    pods   = set(r["pod"] for r in results)

    print()
    print("=" * 60)
    print(f"Results   : {received}/{len(instance_ids)}  "
          f"MISS={len(misses)}  HIT={len(hits)}  ERR={len(errors)}")

    if misses:
        avg_e2e  = sum(r["e2e_s"]         for r in misses) / len(misses)
        avg_s3   = sum(r["s3_download_s"]  for r in misses) / len(misses)
        avg_load = sum(r["model_load_s"]   for r in misses) / len(misses)
        avg_inf  = sum(r["inference_s"]    for r in misses) / len(misses)
        print(f"MISS e2e  : {avg_e2e:.3f}s avg  "
              f"(s3={avg_s3:.3f}s  load={avg_load:.3f}s  inf={avg_inf:.3f}s)")

    if hits:
        avg_e2e = sum(r["e2e_s"]      for r in hits) / len(hits)
        avg_inf = sum(r["inference_s"] for r in hits) / len(hits)
        print(f"HIT  e2e  : {avg_e2e:.3f}s avg  (inf={avg_inf:.3f}s)")

    if misses and hits:
        speedup = (sum(r["e2e_s"] for r in misses) / len(misses)) / \
                  (sum(r["e2e_s"] for r in hits)   / len(hits))
        print(f"Speedup   : {speedup:.1f}x  (MISS/HIT e2e ratio)")

    print(f"Pods      : {len(pods)}  {pods}")

    if received < len(instance_ids):
        print(f"\nWARNING: only received {received}/{len(instance_ids)} "
              f"results before timeout.")


if __name__ == "__main__":
    run()
