#!/usr/bin/env python3
"""
Subprocess baseline — concurrent ramp test.

Sends 3 ramps of concurrent requests (2, 5, 10 threads × 10 requests each).
Demonstrates that without importlib reuse or model caching:
  - Every request pays the full Python subprocess startup cost (~2-4s per request).
  - Latency gets WORSE as concurrency increases (more subprocesses competing for CPU).
  - Adding more pods (scale-out) spreads the load but does NOT reduce per-request cost.
  - There are no cache hits — every request downloads from S3 fresh.

Compare with cache-aware-scaleout-routing/tests/test_scale_out.py.

Usage:
  pip install requests
  python test_ramp.py
  python test_ramp.py --url https://<knative-service-url>
"""

import argparse, csv, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests as http_requests
from collections import defaultdict

DEFAULT_URL = "http://localhost:8080"
RESULTS_DIR = Path(__file__).parent.parent / "results"

ALL_INSTANCES = [f"instance-{n:03d}" for n in range(1, 21)]
RAMPS = [
    {"name": "Ramp-1", "threads":  2, "requests_per_thread": 10},
    {"name": "Ramp-2", "threads":  5, "requests_per_thread": 10},
    {"name": "Ramp-3", "threads": 10, "requests_per_thread": 10},
]
RAMP_PAUSE_S = 30

CSV_FIELDS = ["ramp","thread_id","instance_id","request_num","pod_id",
              "total_s","import_overhead_s","registry_lookup_s","s3_download_s","inference_s",
              "status","error","timestamp"]


def run_request(base_url, instance_id, thread_id, req_num, ramp_name):
    row = {f: None for f in CSV_FIELDS}
    row.update({"ramp": ramp_name, "thread_id": thread_id,
                "instance_id": instance_id, "request_num": req_num,
                "timestamp": datetime.now(timezone.utc).isoformat()})
    try:
        r = http_requests.post(
            f"{base_url}/inference/predict",
            json={"instanceId": instance_id},
            headers={"X-Instance-ID": instance_id},
            timeout=120,
            verify=False)
        row["status"] = r.status_code
        if r.status_code == 200:
            d = r.json(); t = d.get("timing", {})
            row["total_s"]           = t.get("total_s")
            row["import_overhead_s"] = t.get("import_overhead_s")
            row["registry_lookup_s"] = t.get("registry_lookup_s")
            row["s3_download_s"]     = t.get("s3_download_s")
            row["inference_s"]       = t.get("inference_s")
            row["pod_id"]            = d.get("pod_id")
        else:
            row["error"] = r.text[:200]
    except Exception as e:
        row["status"] = "ERROR"; row["error"] = str(e)
    return row


def thread_worker(base_url, instance_id, thread_id, n_requests, ramp_name):
    rows = []
    for i in range(1, n_requests + 1):
        row = run_request(base_url, instance_id, thread_id, i, ramp_name)
        if row["status"] == 200:
            pod_short = (row["pod_id"] or "unknown")[-12:]
            print(f"  [{ramp_name}] t{thread_id:02d} {instance_id} req{i:>2}: "
                  f"total={row['total_s']:>5.2f}s  imports={row['import_overhead_s']:>5.2f}s  "
                  f"s3={row['s3_download_s']:>5.3f}s  pod=...{pod_short}")
        else:
            print(f"  [{ramp_name}] t{thread_id:02d} {instance_id} req{i:>2}: ERROR {str(row['error'])[:60]}")
        rows.append(row)
    return rows


def run_ramp(base_url, ramp, all_rows):
    name = ramp["name"]; n_threads = ramp["threads"]; n_req = ramp["requests_per_thread"]
    assignments = [ALL_INSTANCES[i % len(ALL_INSTANCES)] for i in range(n_threads)]
    print(f"\n{'─'*72}")
    print(f"  {name}: {n_threads} threads × {n_req} requests = {n_threads*n_req} total")
    for tid, iid in enumerate(assignments):
        print(f"    thread-{tid:02d} → {iid}")
    print(f"{'─'*72}")
    ramp_start = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(thread_worker, base_url, assignments[tid], tid, n_req, name): tid
                   for tid in range(n_threads)}
        for f in as_completed(futures):
            all_rows.extend(f.result())
    elapsed = round(time.time() - ramp_start, 1)
    good = [r for r in all_rows if r["ramp"] == name and r["status"] == 200]
    pods_seen = {r["pod_id"] for r in good if r["pod_id"]}
    avg     = sum(r["total_s"] for r in good) / len(good) if good else 0
    avg_imp = sum(r["import_overhead_s"] for r in good) / len(good) if good else 0
    avg_s3  = sum(r["s3_download_s"] for r in good) / len(good) if good else 0
    print(f"\n  {name} summary ({elapsed}s wall time):")
    print(f"    Requests OK  : {len(good)}/{n_threads*n_req}")
    print(f"    Pods seen    : {len(pods_seen)}  {sorted(p[-12:] for p in pods_seen)}")
    print(f"    Avg total    : {avg:.3f}s  (every request pays full cost)")
    print(f"    Avg imports  : {avg_imp:.3f}s  (Python subprocess startup — no reuse)")
    print(f"    Avg s3       : {avg_s3:.3f}s  (always downloaded from S3 — no cache)")
    print(f"    Cache hits   : 0  (subprocess has no in-process cache)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    default=DEFAULT_URL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else RESULTS_DIR / "ramp-results.csv"

    import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r = http_requests.get(f"{args.url}/inference/health", timeout=10, verify=False)
    print(f"[preflight] {r.json()}")

    print(f"\n{'='*72}")
    print(f"  Subprocess Ramp Test — no importlib reuse, no model cache, no consistent hashing")
    print(f"  Every request pays: Python startup + registry lookup + S3 download + inference")
    print(f"{'='*72}")

    all_rows = []
    for idx, ramp in enumerate(RAMPS):
        run_ramp(args.url, ramp, all_rows)
        if idx < len(RAMPS) - 1:
            print(f"\n  [{ramp['name']} done] Pausing {RAMP_PAUSE_S}s...")
            time.sleep(RAMP_PAUSE_S)

    print(f"\n{'='*72}")
    print(f"  Final Summary — Subprocess (no caching)")
    print(f"{'='*72}")
    for ramp in RAMPS:
        good = [r for r in all_rows if r["ramp"] == ramp["name"] and r["status"] == 200]
        if not good: continue
        pods = {r["pod_id"] for r in good if r["pod_id"]}
        avg = sum(r["total_s"] for r in good) / len(good)
        avg_imp = sum(r["import_overhead_s"] for r in good) / len(good)
        print(f"\n  {ramp['name']} ({ramp['threads']} threads):")
        print(f"    Pods active : {len(pods)}")
        print(f"    Requests OK : {len(good)}/{ramp['threads']*ramp['requests_per_thread']}")
        print(f"    Avg total   : {avg:.3f}s")
        print(f"    Avg imports : {avg_imp:.3f}s  (Python startup — paid on every request)")
        print(f"    Cache hits  : 0  (no cache — always fresh subprocess)")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader(); w.writerows(all_rows)
    print(f"\n  Results saved → {output}")


if __name__ == "__main__":
    main()
