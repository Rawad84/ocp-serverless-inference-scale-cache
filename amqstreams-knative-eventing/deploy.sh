#!/bin/bash
# Deploy phase-4 queue-based inference.
# Run order:
#   1. AMQ Streams operator + Kafka cluster + topics
#   2. Knative Eventing + KnativeKafka (connects Eventing to Kafka)
#   3. KafkaSource (reads requests topic → HTTP to Knative Service)
#   4. Build and deploy cache-aware-inference-queue Knative Service
set -euo pipefail

NAMESPACE=inference-benchmark
SERVICE=cache-aware-inference-queue
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. AMQ Streams operator ───────────────────────────────────────────────────
echo "=== Installing AMQ Streams operator ==="
oc apply -f "${SCRIPT_DIR}/prerequisites/01-amq-streams/subscription.yaml"

echo "Waiting for AMQ Streams operator to be ready..."
until oc get csv -n openshift-operators 2>/dev/null | grep -q "amq-streams.*Succeeded"; do
  echo -n "."; sleep 10
done
echo " Ready"

# ── 2. Kafka cluster + topics ─────────────────────────────────────────────────
echo ""
echo "=== Creating Kafka cluster ==="
oc apply -f "${SCRIPT_DIR}/prerequisites/01-amq-streams/kafka-cluster.yaml"

echo "Waiting for Kafka cluster to be Ready (this takes ~2 minutes)..."
oc wait kafka/inference-kafka -n "${NAMESPACE}" \
  --for=condition=Ready --timeout=300s

echo ""
echo "=== Creating Kafka topics ==="
oc apply -f "${SCRIPT_DIR}/prerequisites/01-amq-streams/kafka-topics.yaml"

# ── 3. Knative Eventing ───────────────────────────────────────────────────────
echo ""
echo "=== Creating Knative Eventing ==="
oc apply -f "${SCRIPT_DIR}/prerequisites/02-knative-eventing/knative-eventing.yaml"

echo "Waiting for Knative Eventing to be Ready..."
oc wait knativeeventing/knative-eventing -n knative-eventing \
  --for=condition=Ready --timeout=300s

echo ""
echo "=== Connecting Knative Eventing to Kafka ==="
oc apply -f "${SCRIPT_DIR}/prerequisites/02-knative-eventing/knative-kafka.yaml"

echo "Waiting for KnativeKafka to be Ready..."
sleep 30  # KnativeKafka takes a moment to reconcile
oc get knativekafka -n knative-eventing || true

# ── 4. KafkaSource ────────────────────────────────────────────────────────────
echo ""
echo "=== Creating KafkaSource ==="
oc apply -f "${SCRIPT_DIR}/manifests/kafka-source.yaml"

# ── 5. Build app image ────────────────────────────────────────────────────────
echo ""
echo "=== Building ${SERVICE} image ==="
oc new-build --name="${SERVICE}" \
  --binary --strategy=docker \
  -n "${NAMESPACE}" 2>/dev/null || true
oc start-build "${SERVICE}" --from-dir="${SCRIPT_DIR}/app" --follow -n "${NAMESPACE}"

# ── 6. Deploy Knative Service ─────────────────────────────────────────────────
echo ""
echo "=== Deploying Knative Service ==="
oc apply -f "${SCRIPT_DIR}/manifests/knative-service-queue.yaml"

echo "Waiting for Knative Service to be Ready..."
oc wait --for=condition=Ready ksvc "${SERVICE}" -n "${NAMESPACE}" --timeout=180s

# ── Summary ───────────────────────────────────────────────────────────────────
KSVC_URL=$(oc get ksvc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.status.url}')
KAFKA_BS="inference-kafka-kafka-bootstrap.${NAMESPACE}.svc:9092"

echo ""
echo "=== Deployed ==="
echo "  Knative Service : ${KSVC_URL}"
echo "  Health          : ${KSVC_URL}/health"
echo "  Request topic   : inference-requests"
echo "  Results topic   : inference-results"
echo "  Bootstrap       : ${KAFKA_BS}"
echo ""
echo "=== Run test (from inside cluster or with port-forward) ==="
echo "  python tests/test_queue_inference.py \\"
echo "    --bootstrap ${KAFKA_BS} \\"
echo "    --instances 10 \\"
echo "    --handler-version v1"
