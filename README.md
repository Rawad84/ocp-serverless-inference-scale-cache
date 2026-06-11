# ocp-knative-model-cache-lab

A lab that demonstrates how to build efficient, cache-aware CPU inference services
on **OpenShift AI 3.x using Knative (OpenShift Serverless)** — the recommended
serving path for CPU workloads following the removal of ModelMesh from RHOAI 3.x.

Three techniques are combined and measured against a baseline to show their
real impact on latency and horizontal scalability:

- **importlib handler cache** — inference handler loaded once per pod lifetime, not per request
- **In-memory model artifact cache** — model objects held in pod RAM (L1 TTLCache), keyed by instance ID
- **Async queue-based inference** — AMQ Streams (Kafka) + Knative Eventing for decoupled, event-driven serving

---

## Repository Layout

```
ocp-knative-model-cache-lab/
├── prerequisites/                        # One-time cluster setup
│   ├── 01-knative-serverless/            # OpenShift Serverless operator + KnativeServing
│   ├── 02-minio/                         # MinIO deployment + bucket creation
│   └── 03-model-registry/               # ODH Model Registry
├── setup/                               # Model generation, upload, registry registration
│   ├── setup_models.py                  # Generates 100 instanceIds as single zip archives (~50 MB each)
│   ├── setup-job.yaml                   # Kubernetes Job (recommended — 4 CPU / 8 Gi)
│   └── deploy-setup-job.sh              # Uploads script as ConfigMap and submits job
├── subprocess-baseline/                 # Phase 1 baseline: subprocess per request, no caching
│   ├── app/
│   ├── manifests/knative-service.yaml
│   ├── tests/test_ramp.py
│   ├── deploy.sh
│   └── run-test.sh
├── cache-aware-scaleout-routing/        # Phase 2: importlib + L1 TTLCache, HTTP serving
│   ├── app/
│   ├── handlers/                        # Handler versions (source of truth — loaded via PVC)
│   │   ├── v1/predict_handler.py        # Basic mean scoring
│   │   ├── v2/predict_handler.py        # + confidence interval
│   │   └── v3/predict_handler.py        # + weighted ensemble
│   ├── manifests/
│   │   ├── knative-service.yaml
│   │   ├── handlers-pvc.yaml            # PVC for handler storage (ReadWriteOnce)
│   │   ├── handlers-configmap.yaml      # Handler code embedded as ConfigMap
│   │   ├── handlers-loader-job.yaml     # Job that copies handlers from ConfigMap to PVC
│   │   └── generate-handlers-configmap.sh
│   ├── tests/test_scale_out.py
│   ├── results/
│   ├── deploy.sh
│   └── run-test.sh
└── amqstreams-knative-eventing/         # Phase 3: async queue-based inference via Kafka
    ├── app/
    │   ├── app_queue.py                 # FastAPI — accepts CloudEvents, writes results to Kafka
    │   ├── requirements.txt
    │   └── Dockerfile
    ├── handlers/                        # Same PVC handler versioning pattern
    │   ├── v1/predict_handler.py
    │   ├── v2/predict_handler.py
    │   └── v3/predict_handler.py
    ├── manifests/
    │   ├── knative-service-queue.yaml   # Knative Service for queue-based inference
    │   ├── kafka-source.yaml            # KafkaSource: inference-requests → Knative Service
    │   ├── handlers-pvc.yaml
    │   ├── handlers-configmap.yaml
    │   ├── handlers-loader-job.yaml
    │   └── generate-handlers-configmap.sh
    ├── prerequisites/
    │   ├── 01-amq-streams/              # AMQ Streams operator + Kafka cluster + topics
    │   │   ├── subscription.yaml
    │   │   ├── kafka-cluster.yaml       # KRaft mode, v4.1.0, single broker
    │   │   └── kafka-topics.yaml        # inference-requests + inference-results (3 partitions)
    │   └── 02-knative-eventing/         # Knative Eventing + KnativeKafka CR
    │       ├── knative-eventing.yaml
    │       └── knative-kafka.yaml       # Connects Knative Eventing to AMQ Streams
    ├── tests/
    │   └── test_e2e_queue.py            # E2E latency test: send to Kafka, read results
    └── deploy.sh
```

---

## Prerequisites

| Component       | Namespace           | Notes                                 |
|-----------------|---------------------|---------------------------------------|
| Knative Serving | knative-serving     | Via OpenShift Serverless operator     |
| S3 storage      | minio               | Replace with ODF NooBaa in production |
| Model Registry  | inference-benchmark | ODH Model Registry (standalone)       |

```bash
oc apply -f prerequisites/01-knative-serverless/operatorgroup.yaml
oc apply -f prerequisites/01-knative-serverless/subscription.yaml
oc apply -f prerequisites/01-knative-serverless/knative-serving.yaml
oc apply -f prerequisites/02-minio/minio.yaml
bash prerequisites/02-minio/create-buckets.sh
oc apply -f prerequisites/03-model-registry/model-registry.yaml
```

---

## Setup

Each instanceId is stored as a **single zip file (~50 MB)** containing 5 model files:
- `model1_{instanceId}.npz` — numpy weight matrix (~39 MB)
- `model2–5_{instanceId}.joblib` — sklearn pipelines (~2–5 MB each)

This matches real production asset packaging where models ship as a single archive.

```bash
# Run as a Kubernetes Job (recommended)
bash setup/deploy-setup-job.sh

# Watch progress
oc logs -n inference-benchmark -l job-name=setup-models-zip -f

# Confirm completion
oc wait --for=condition=Complete job/setup-models-zip -n inference-benchmark --timeout=3600s
```

---

## Run: Subprocess Baseline

```bash
cd subprocess-baseline
bash deploy.sh
bash run-test.sh
```

---

## Run: Cache-Aware Scale-Out

```bash
cd cache-aware-scaleout-routing
bash deploy.sh
bash run-test.sh
```

Watch pods scale in a second terminal:
```bash
watch oc get pods -n inference-benchmark -l serving.knative.dev/service=cache-aware-inference
```

Results written to `results/<timestamp>/results.csv`.

---

## Traffic Flow — Current Implementation (Knative URL)

Both implementations use the Knative URL so that KPA receives concurrency
statistics and autoscaling fires correctly.

```
Client
  │
  ▼
HAProxy (OCP Router)          ← wildcard *.apps DNS resolves here
  │
  ▼
Kourier (Knative ingress)     ← routes by Knative hostname
  │
  ▼
Activator                     ← buffers requests, pushes concurrency stats to KPA
  │                              KPA scales pods up/down based on these stats
  ▼
queue-proxy (port 8012)       ← sidecar in every pod; also reports stats
  │
  ▼
App container (port 8080)     ← importlib cache + L1 TTLCache
```

---

## Results

### Subprocess Baseline

Every request spawns a fresh Python subprocess. All imports and model downloads
repeat from scratch every single time.

| Ramp | Threads | Requests | Avg latency | Import overhead | Hit rate | Pods |
|------|---------|----------|-------------|-----------------|----------|------|
| 1    | 2       | 20       | 3.210 s     | 2.384 s (74%)   | 0%       | 2    |
| 2    | 5       | 50       | 3.397 s     | 2.459 s (72%)   | 0%       | 3    |
| 3    | 10      | 100      | 4.245 s     | 3.121 s (73%)   | 0%       | 4    |

Knative scales out to handle the load but scaling adds no benefit. Every request
on every pod still pays the full import + download cost from scratch.

---

### Cache-Aware Scale-Out

Each instanceId is a single zip (~50 MB). On a cache MISS the pod downloads,
unzips, and loads models into RAM. On a cache HIT only inference runs.

**Scale-out observed:**
- Ramp-1 (8 threads): 1 pod
- Ramp-2 (20 threads): 5 pods — KPA scaled out correctly
- Ramp-3 (30 threads): 5 pods — held at max

**MISS cost breakdown (avg across 154 misses):**

```
registry_lookup :   0.329s   ← Model Registry REST calls
s3_download     :   0.886s   ← single ~49 MB zip from MinIO
unzip           :   0.032s   ← in-memory extraction, negligible
model_load      :   0.908s   ← np.load + 4× joblib.load → Python objects
cache_store     :   0.000s   ← TTLCache insert, negligible
inference       :   0.078s
────────────────────────────
TOTAL MISS avg  :   2.315s
```

**HIT cost: 0.166s avg** — only inference runs, all other steps are zero.

**Speedup vs MISS:**

| Ramp   | Threads | Pods | HIT avg | Speedup |
|--------|---------|------|---------|---------|
| Ramp-1 | 8       | 1    | 0.043s  | 64×     |
| Ramp-2 | 20      | 5    | 0.094s  | 25×     |
| Ramp-3 | 30      | 5    | 0.219s  | 10×     |

HIT inference time grows with concurrency (0.043s → 0.219s) because at 30
threads all hitting 5 pods, CPU contention on sklearn/numpy operations increases.
It is still always far cheaper than a MISS.

`registry_lookup (0.329s)` and `model_load (0.908s)` are nearly equal in cost —
both significant bottlenecks on a MISS. Caching eliminates both completely.

---

## When the MISS Penalty Is Still Too High

If MISS cost (~2.3s) is unacceptable, the root cause is that Knative's random
load balancing routes any request to any pod — a new pod that has never seen a
given instanceId always pays the full pipeline cost.

The fix is **sticky routing**: guarantee the same instanceId always reaches the
same pod so the L1 cache is always warm after the first request.

### Option A — Gateway API with consistent hashing (no Knative autoscaling)

Route traffic through a Kubernetes Gateway API HTTPRoute backed by an Istio
`DestinationRule` with `RING_HASH` on the `X-Instance-ID` header. The Gateway
Envoy proxy enforces stickiness at the load-balancer level.

```
Client
  │
  ▼
Gateway API (HTTPRoute)       ← matches on path, forwards X-Instance-ID header
  │
  ▼
Envoy (DestinationRule)       ← RING_HASH on X-Instance-ID header
  │                              same instanceId always routes to same pod
  ▼
ClusterIP → queue-proxy       ← Activator is NOT in this path
  │                              KPA receives no stats → no autoscaling
  ▼
App container                 ← L1 cache always hit after first request per pod
```

**Trade-off:** traffic bypasses the Knative Activator entirely, so KPA never
receives concurrency statistics and autoscaling does not fire. Replica count must
be managed manually. The Knative Service can still be used for deployment
lifecycle, but it will not scale automatically.

### Option B — StatefulSet with deterministic pod identity

Replace the Knative Service with a plain Kubernetes `StatefulSet`. Each pod gets
a stable DNS name (`pod-0`, `pod-1`, ...). Route `instanceId % N` to the
corresponding pod using a thin routing layer or header-based HTTPRoute match rules.

```
Client
  │
  ▼
Gateway API (HTTPRoute)       ← or any ingress / load balancer
  │
  ▼
Routing layer                 ← computes instanceId % N, selects pod index
  │
  ▼
StatefulSet pod-N             ← stable identity: pod-0, pod-1, pod-2 ...
  │                              no Knative, no Activator, no queue-proxy
  ▼
App container                 ← L1 cache always hit after first request per pod
```

**Trade-off:** no scale-to-zero, no Knative lifecycle management. You own replica
count, rolling updates, and readiness probes. Simplest approach for a fixed-size
warm pool.

---

## How Knative KPA Decides to Scale

Knative's Pod Autoscaler measures average in-flight requests per pod over a
rolling window (10s in these manifests). When that average exceeds the concurrency
target (2), KPA adds a pod.

Traffic must flow through the **Knative URL** (via Activator → queue-proxy) for
KPA to receive concurrency statistics. Traffic routed directly to pod ClusterIP
via Gateway API bypasses the Activator — autoscaling will not fire regardless of load.

---

## Key Architecture Notes

**L1 cache stores loaded Python objects, not raw bytes.**
After a MISS, the zip bytes and raw file bytes are discarded immediately. Only the
final deserialized objects (numpy arrays, sklearn Pipelines) remain in the TTLCache.
RAM footprint per slot is ~150–400 MB depending on model size — size
`MODEL_CACHE_SIZE` accordingly.

**Pod restart clears the cache.**
On restart, the first request for each previously cached instanceId pays the full
MISS cost again until the cache is rebuilt.

**No cross-pod cache sharing.**
Two pods serving the same instanceId each maintain their own independent copy in
RAM. Without sticky routing, scale-out increases the chance of cold misses.

---

## Caveats

- `MODEL_CACHE_SIZE` (default 30) limits how many instanceIds each pod holds.
  With ~200 MB per slot, 30 slots ≈ 6 GB peak RAM per pod. Tune to your memory limit.
- Registry lookup (0.33s) is paid on every MISS. Consider an in-process URI cache
  keyed by instanceId if the Model Registry becomes a bottleneck at scale.
- Real production inference (PyTorch forward pass, ensemble scoring) typically takes
  0.5–3s on CPU, which provides natural computation time for KPA to observe
  concurrency and trigger scale-out without any artificial delays.

---

## Phase 3 — Async Queue Inference (AMQ Streams + Knative Eventing)

Replaces the synchronous HTTP request/response model with a fully async
event-driven pipeline. The client sends a message to a Kafka topic and reads
the result from a separate results topic. The Knative Service never blocks waiting
for the client.

### Additional Prerequisites

| Component | Namespace | Notes |
|-----------|-----------|-------|
| AMQ Streams operator | openshift-operators | Via OperatorHub |
| Knative Eventing | knative-eventing | Via OpenShift Serverless operator |
| KnativeKafka CR | knative-eventing | Connects Eventing to AMQ Streams |

```bash
# AMQ Streams operator (approve InstallPlan if required)
oc apply -f amqstreams-knative-eventing/prerequisites/01-amq-streams/subscription.yaml
oc apply -f amqstreams-knative-eventing/prerequisites/01-amq-streams/kafka-cluster.yaml
oc apply -f amqstreams-knative-eventing/prerequisites/01-amq-streams/kafka-topics.yaml

# Knative Eventing + KnativeKafka
oc apply -f amqstreams-knative-eventing/prerequisites/02-knative-eventing/knative-eventing.yaml
oc apply -f amqstreams-knative-eventing/prerequisites/02-knative-eventing/knative-kafka.yaml
```

### Traffic Flow

```
External client (oc exec)
        │
        │  KafkaProducer — TCP :9092 (internal)
        ▼
AMQ Streams — inference-requests topic (3 partitions)
        │
        │  KafkaSource polls topic
        ▼
KafkaSource (knative-eventing namespace)
        │
        │  HTTP POST CloudEvent to Knative Service
        │  ce-type: dev.knative.kafka.event
        │  body: { instanceId, handlerVersion, requestId }
        ▼
Knative Service — cache-aware-inference-queue
        │
        ├── importlib handler cache (v1 / v2 / v3 from PVC)
        ├── L1 TTLCache — model objects in pod RAM
        ├── MISS → Model Registry lookup + MinIO S3 download
        └── HIT  → inference only (0.024s)
        │
        │  KafkaProducer writes result
        ▼
AMQ Streams — inference-results topic (3 partitions)
        │
        │  KafkaConsumer reads result
        ▼
External client — receives result + timing breakdown
```

### Handler Versioning via PVC

Handlers are never burned into the image. The image contains only the app
infrastructure. Handler code is stored on a PVC and loaded at runtime via
`importlib`. All three versions are available simultaneously — the request
body `handlerVersion` field selects which runs.

```bash
# Generate ConfigMap from handler source files
bash amqstreams-knative-eventing/manifests/generate-handlers-configmap.sh

# Apply ConfigMap + loader Job to populate PVC
oc apply -f amqstreams-knative-eventing/manifests/handlers-configmap.yaml
oc apply -f amqstreams-knative-eventing/manifests/handlers-loader-job.yaml
```

### Deploy

```bash
cd amqstreams-knative-eventing
bash deploy.sh
```

### Run E2E Test

The test script sends messages directly to Kafka from inside the inference pod
(Kafka has no external listener in this lab setup) and reads results back.

```bash
# Get current pod name
POD=$(oc get pods -n inference-benchmark \
  -l serving.knative.dev/service=cache-aware-inference-queue \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

# Copy test script to pod
oc cp amqstreams-knative-eventing/tests/test_e2e_queue.py \
  inference-benchmark/${POD}:/tmp/test_e2e_queue.py -c user-container

# Run test (default: 10 instanceIds x 3 repeats = 30 messages)
oc exec -n inference-benchmark ${POD} -c user-container -- \
  python3 /tmp/test_e2e_queue.py

# Run with different parameters
oc exec -n inference-benchmark ${POD} -c user-container -- \
  env INSTANCES=30 REPEATS=5 HANDLER_VERSION=v2 \
  python3 /tmp/test_e2e_queue.py
```

### Phase 3 Results

```
Cache MISS (first request per instanceId):
  registry lookup : ~0.2s
  S3 download     : ~0.6s
  unzip + load    : ~0.2s
  inference       : ~0.01s
  total           : ~0.8–1.0s

Cache HIT (repeat request, same instanceId, same pod):
  inference only  : ~0.024s
  total           : ~0.024s

Speedup           : ~30x

Autoscaling: pods scaled 1 → 2 under 30-message burst, returned to 1 at idle
```

### Kafka Topics

| Topic | Partitions | Retention | Purpose |
|-------|-----------|-----------|---------|
| inference-requests | 3 | 1 hour | Inbound inference requests |
| inference-results | 3 | 1 hour | Outbound inference results |

### Handler Versions

| Version | What it adds |
|---------|-------------|
| v1 | Basic mean score across all models |
| v2 | + confidence interval (std dev across model scores) |
| v3 | + weighted ensemble (joblib models weight 2×, npz weight 1×) + per-model breakdown |
