#!/usr/bin/env python3
"""
Cache-Aware Scale-Out — Ramp Latency Test.

30 instanceIds, 3 ramps of increasing concurrency.
Each ramp adds more threads to push Knative KPA to scale out.

Per-request timing breakdown:
  MISS (first time a pod sees an instanceId):
    registry_lookup_s — fetch zip S3 URI from Model Registry
    s3_download_s     — download ~50 MB zip from MinIO
    unzip_s           — extract 5 files in memory
    model_load_s      — np.load + joblib.load × 4 → Python objects
    cache_store_s     — insert loaded objects into L1 LRU TTLCache
    inference_s       — run prediction on loaded objects
    total_s           — wall time end-to-end

  HIT (same instanceId hits the same pod again):
    All steps above = 0.000s  (objects already in pod RAM)
    inference_s       — run prediction on cached objects
    total_s           — wall time (≈ inference_s + import_overhead_s)

Ramp design:
  Ramp-1:  8 threads × 5  requests — low load, warms first 8 instances
  Ramp-2: 20 threads × 8  requests — medium load, forces scale-out
  Ramp-3: 30 threads × 10 requests — full load, max pods, mostly hits

Results saved to results/<timestamp>/
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEALTH_PATH   = "/inference/health"
PREDICT_PATH  = "/inference/predict"
HANDLER_VERSION = "v1"   # override with --handler-version

INSTANCES = [f"instance-{n:03d}" for n in range(1, 31)]   # 30 instances

RAMPS = [
    {"name": "Ramp-1", "threads":  8, "requests_per_thread":  5},
    {"name": "Ramp-2", "threads": 20, "requests_per_thread":  8},
    {"name": "Ramp-3", "threads": 30, "requests_per_thread": 10},
]

RAMP_PAUSE_S = 15   # brief pause between ramps — let KPA observe the load shift

CSV_FIELDS = [
    "ramp", "thread_id", "instance_id", "request_num",
    "pod_id", "cache_hit", "cache_size", "zip_size_mb",
    "import_overhead_s",
    "registry_lookup_s", "s3_download_s", "unzip_s",
    "model_load_s", "cache_store_s", "inference_s", "total_s",
    "status", "error", "timestamp",
]


def wait_healthy(base_url: str, timeout: int = 90) -> bool:
    url = f"{base_url}{HEALTH_PATH}"
    print(f"[preflight] GET {url}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = http_requests.get(url, timeout=10, verify=False)
            if r.status_code == 200:
                print(f"[preflight] healthy — {r.json()}")
                return True
        except Exception as e:
            print(f"[preflight] waiting... ({e})")
        time.sleep(3)
    return False


def do_request(base_url: str, instance_id: str,
               thread_id: int, req_num: int, ramp_name: str) -> dict:
    row = {f: None for f in CSV_FIELDS}
    row.update({
        "ramp":        ramp_name,
        "thread_id":   thread_id,
        "instance_id": instance_id,
        "request_num": req_num,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })
    try:
        r = http_requests.post(
            f"{base_url}{PREDICT_PATH}",
            headers={"Content-Type": "application/json",
                     "X-Instance-ID": instance_id},
            json={"instanceId": instance_id, "handlerVersion": HANDLER_VERSION},
            timeout=180,
            verify=False,
        )
        row["status"] = r.status_code
        if r.status_code == 200:
            d = r.json()
            t = d.get("timing", {})
            row["pod_id"]            = d.get("pod_id")
            row["cache_hit"]         = d.get("cache_hit")
            row["cache_size"]        = d.get("cache_size")
            row["zip_size_mb"]       = d.get("zip_size_mb", 0)
            row["import_overhead_s"] = t.get("import_overhead_s", 0)
            row["registry_lookup_s"] = t.get("registry_lookup_s", 0)
            row["s3_download_s"]     = t.get("s3_download_s", 0)
            row["unzip_s"]           = t.get("unzip_s", 0)
            row["model_load_s"]      = t.get("model_load_s", 0)
            row["cache_store_s"]     = t.get("cache_store_s", 0)
            row["inference_s"]       = t.get("inference_s", 0)
            row["total_s"]           = t.get("total_s", 0)
        else:
            row["error"] = r.text[:200]
    except Exception as e:
        row["status"] = "ERROR"
        row["error"]  = str(e)
    return row


def thread_worker(base_url: str, instance_id: str,
                  thread_id: int, n_requests: int, ramp_name: str) -> list[dict]:
    rows = []
    for i in range(1, n_requests + 1):
        row = do_request(base_url, instance_id, thread_id, i, ramp_name)
        _print_row(row, ramp_name)
        rows.append(row)
    return rows


def _print_row(row: dict, ramp_name: str):
    if row["status"] != 200:
        print(f"  [{ramp_name}] t{row['thread_id']:02d} {row['instance_id']} "
              f"req{row['request_num']:>2}: ERROR {str(row['error'])[:60]}")
        return

    hit  = row["cache_hit"]
    pod  = (row["pod_id"] or "unknown")[-12:]

    if hit:
        print(
            f"  [{ramp_name}] t{row['thread_id']:02d} {row['instance_id']} "
            f"req{row['request_num']:>2}: HIT   "
            f"infer={row['inference_s']:>6.3f}s  "
            f"total={row['total_s']:>6.3f}s  "
            f"cache={row['cache_size']:>2}  pod=...{pod}"
        )
    else:
        print(
            f"  [{ramp_name}] t{row['thread_id']:02d} {row['instance_id']} "
            f"req{row['request_num']:>2}: MISS  "
            f"reg={row['registry_lookup_s']:>5.3f}s  "
            f"s3={row['s3_download_s']:>5.3f}s({row['zip_size_mb']:.0f}MB)  "
            f"unzip={row['unzip_s']:>5.3f}s  "
            f"load={row['model_load_s']:>5.3f}s  "
            f"infer={row['inference_s']:>5.3f}s  "
            f"total={row['total_s']:>6.3f}s  "
            f"cache={row['cache_size']:>2}  pod=...{pod}"
        )


def run_ramp(base_url: str, ramp: dict, all_rows: list):
    name      = ramp["name"]
    n_threads = ramp["threads"]
    n_req     = ramp["requests_per_thread"]

    # Distribute 30 instances evenly across threads (round-robin)
    assignments = [INSTANCES[i % len(INSTANCES)] for i in range(n_threads)]

    print(f"\n{'─'*80}")
    print(f"  {name}: {n_threads} threads × {n_req} requests = {n_threads*n_req} total")
    print(f"  Instance assignments (thread → instanceId):")
    for tid in range(0, n_threads, 5):
        chunk = [f"t{t:02d}→{assignments[t].split('-')[1]}"
                 for t in range(tid, min(tid+5, n_threads))]
        print(f"    {' | '.join(chunk)}")
    print(f"{'─'*80}")

    ramp_start = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(thread_worker, base_url, assignments[tid], tid, n_req, name): tid
            for tid in range(n_threads)
        }
        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    elapsed = round(time.time() - ramp_start, 1)
    _print_ramp_summary(name, n_threads, n_req, elapsed, all_rows)


def _print_ramp_summary(name: str, n_threads: int, n_req: int,
                        elapsed: float, all_rows: list):
    good   = [r for r in all_rows if r["ramp"] == name and r["status"] == 200]
    hits   = [r for r in good if r["cache_hit"]]
    misses = [r for r in good if not r["cache_hit"]]
    pods   = {r["pod_id"] for r in good if r["pod_id"]}

    def avg(rows, field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    print(f"\n  {name} summary  ({elapsed}s wall time)")
    print(f"  {'─'*60}")
    print(f"  Requests OK  : {len(good)}/{n_threads*n_req}")
    print(f"  Pods active  : {len(pods)}  {sorted(p[-12:] for p in pods)}")
    print(f"  Cache HITs   : {len(hits)}   Cache MISSes : {len(misses)}")

    if misses:
        print(f"\n  MISS breakdown (avg across {len(misses)} misses):")
        print(f"    registry_lookup : {avg(misses,'registry_lookup_s'):>7.3f}s")
        print(f"    s3_download     : {avg(misses,'s3_download_s'):>7.3f}s"
              f"  (zip ~{avg(misses,'zip_size_mb'):.0f} MB)")
        print(f"    unzip           : {avg(misses,'unzip_s'):>7.3f}s")
        print(f"    model_load      : {avg(misses,'model_load_s'):>7.3f}s")
        print(f"    cache_store     : {avg(misses,'cache_store_s'):>7.3f}s")
        print(f"    inference       : {avg(misses,'inference_s'):>7.3f}s")
        print(f"    ─────────────────────────")
        print(f"    total (avg)     : {avg(misses,'total_s'):>7.3f}s")

    if hits:
        print(f"\n  HIT breakdown (avg across {len(hits)} hits):")
        print(f"    inference       : {avg(hits,'inference_s'):>7.3f}s")
        print(f"    total (avg)     : {avg(hits,'total_s'):>7.3f}s")
        if misses:
            speedup = avg(misses, "total_s") / avg(hits, "total_s") if avg(hits, "total_s") > 0 else 0
            print(f"    speedup vs MISS : {speedup:.0f}×")


def print_final_summary(all_rows: list):
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY — Cache-Aware Scale-Out Latency Breakdown")
    print(f"{'='*80}")
    print(f"  {'Ramp':<10}  {'Threads':>7}  {'OK':>5}  {'Hits':>5}  "
          f"{'Miss':>5}  {'Pods':>5}  {'Avg MISS (s)':>12}  {'Avg HIT (s)':>11}")
    print(f"  {'─'*75}")

    for ramp in RAMPS:
        name  = ramp["name"]
        n_t   = ramp["threads"]
        good  = [r for r in all_rows if r["ramp"] == name and r["status"] == 200]
        hits  = [r for r in good if r["cache_hit"]]
        misses= [r for r in good if not r["cache_hit"]]
        pods  = {r["pod_id"] for r in good if r["pod_id"]}

        def avg(rows, field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        miss_avg = avg(misses, "total_s") if misses else 0.0
        hit_avg  = avg(hits,   "total_s") if hits   else 0.0

        print(f"  {name:<10}  {n_t:>7}  {len(good):>5}  {len(hits):>5}  "
              f"{len(misses):>5}  {len(pods):>5}  "
              f"{miss_avg:>12.3f}  {hit_avg:>11.3f}")

    print(f"\n  Step-by-step avg across ALL misses in the run:")
    all_misses = [r for r in all_rows if r["status"] == 200 and not r["cache_hit"]]
    all_hits   = [r for r in all_rows if r["status"] == 200 and r["cache_hit"]]

    def avg_all(rows, field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    if all_misses:
        total_miss = avg_all(all_misses, "total_s")
        print(f"    registry_lookup : {avg_all(all_misses,'registry_lookup_s'):>7.3f}s")
        print(f"    s3_download     : {avg_all(all_misses,'s3_download_s'):>7.3f}s")
        print(f"    unzip           : {avg_all(all_misses,'unzip_s'):>7.3f}s")
        print(f"    model_load      : {avg_all(all_misses,'model_load_s'):>7.3f}s")
        print(f"    cache_store     : {avg_all(all_misses,'cache_store_s'):>7.3f}s")
        print(f"    inference       : {avg_all(all_misses,'inference_s'):>7.3f}s")
        print(f"    import_overhead : {avg_all(all_misses,'import_overhead_s'):>7.3f}s")
        print(f"    ─────────────────────────")
        print(f"    TOTAL MISS avg  : {total_miss:>7.3f}s")

    if all_hits:
        print(f"\n    TOTAL HIT avg   : {avg_all(all_hits,'total_s'):>7.3f}s"
              f"  (inference only)")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",             default="http://localhost:8080")
    parser.add_argument("--results-dir",     default=None)
    parser.add_argument("--handler-version", default="v1")
    args = parser.parse_args()

    global HANDLER_VERSION
    HANDLER_VERSION = args.handler_version

    base_url = args.url.rstrip("/")
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    results_root = Path(args.results_dir) if args.results_dir else \
                   Path(__file__).parent.parent / "results"
    results_dir  = results_root / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    if not wait_healthy(base_url):
        print("ERROR: service not healthy.")
        sys.exit(1)

    print(f"""
{'='*80}
  Cache-Aware Scale-Out — Ramp Latency Test
  Instances   : {len(INSTANCES)} ({INSTANCES[0]} … {INSTANCES[-1]})
  Ramps       : {[r['name'] for r in RAMPS]}
  Results dir : {results_dir}
  Note: MISS pays full pipeline. HIT = inference only.
{'='*80}
""")

    all_rows = []

    for idx, ramp in enumerate(RAMPS):
        run_ramp(base_url, ramp, all_rows)
        if idx < len(RAMPS) - 1:
            print(f"\n  Pausing {RAMP_PAUSE_S}s between ramps...")
            time.sleep(RAMP_PAUSE_S)

    print_final_summary(all_rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = results_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Results saved → {csv_path}")


if __name__ == "__main__":
    main()
