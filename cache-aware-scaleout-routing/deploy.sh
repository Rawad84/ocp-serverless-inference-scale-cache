#!/bin/bash
# Build and deploy cache-aware-inference.
# Handlers are NOT bundled in the image — they live on a PVC populated by a loader Job.
# Traffic flows through Knative URL (Kourier → Activator) so KPA can autoscale.
#
# Deploy order:
#   1. PVC          — persistent volume for handler files
#   2. ConfigMap    — packages all handler versions (generated from handlers/)
#   3. Loader Job   — copies ConfigMap files onto the PVC
#   4. Image build  — app infrastructure only, no handler code
#   5. Knative Svc  — mounts PVC, reads handlers from /mnt/handlers
#
# To add or update a handler version (no image rebuild needed):
#   1. Edit handlers/v2/predict_handler.py
#   2. ./manifests/generate-handlers-configmap.sh
#   3. oc apply -f manifests/handlers-configmap.yaml
#   4. oc delete job handlers-loader -n inference-benchmark --ignore-not-found
#   5. oc apply -f manifests/handlers-loader-job.yaml

set -euo pipefail

NAMESPACE=inference-benchmark
SERVICE=cache-aware-inference

# ── 0. Knative PVC feature flag check ────────────────────────────────────────
echo "=== Checking Knative PVC feature flag ==="
PVC_FLAG=$(oc get configmap config-features -n knative-serving \
  -o jsonpath='{.data.kubernetes\.podspec-persistent-volume-claim}' 2>/dev/null || echo "")
if [[ "${PVC_FLAG}" != "enabled" ]]; then
  echo ""
  echo "WARNING: Knative PVC support is not enabled."
  echo "Run the following before deploying:"
  echo "  oc patch configmap config-features -n knative-serving --type=merge \\"
  echo "    -p '{\"data\":{\"kubernetes.podspec-persistent-volume-claim\":\"enabled\"}}'"
  echo ""
  read -rp "Continue anyway? [y/N] " confirm
  [[ "${confirm}" =~ ^[Yy]$ ]] || exit 1
fi

# ── 1. PVC ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Applying PVC ==="
oc apply -f manifests/handlers-pvc.yaml

# ── 2. ConfigMap ──────────────────────────────────────────────────────────────
echo ""
echo "=== Applying handlers ConfigMap ==="
if [[ ! -f manifests/handlers-configmap.yaml ]]; then
  echo "handlers-configmap.yaml not found — generating now..."
  ./manifests/generate-handlers-configmap.sh
fi
oc apply -f manifests/handlers-configmap.yaml

# ── 3. Loader Job ─────────────────────────────────────────────────────────────
echo ""
echo "=== Running handlers loader Job ==="
oc delete job handlers-loader -n "${NAMESPACE}" --ignore-not-found
oc apply -f manifests/handlers-loader-job.yaml

echo "Waiting for loader Job to complete..."
oc wait --for=condition=Complete job/handlers-loader \
  -n "${NAMESPACE}" --timeout=120s

echo "Handlers on PVC:"
oc logs job/handlers-loader -n "${NAMESPACE}" | grep predict_handler || true

# ── 4. Image build ────────────────────────────────────────────────────────────
echo ""
echo "=== Building ${SERVICE} image (no handler code bundled) ==="
oc new-build --name="${SERVICE}" \
  --binary --strategy=docker \
  -n "${NAMESPACE}" 2>/dev/null || true
oc start-build "${SERVICE}" --from-dir=./app --follow -n "${NAMESPACE}"

# ── 5. Knative Service ────────────────────────────────────────────────────────
echo ""
echo "=== Applying Knative Service ==="
oc apply -f manifests/knative-service.yaml

echo "=== Waiting for Knative Service to be Ready ==="
oc wait --for=condition=Ready ksvc "${SERVICE}" -n "${NAMESPACE}" --timeout=180s

KSVC_URL=$(oc get ksvc "${SERVICE}" -n "${NAMESPACE}" -o jsonpath='{.status.url}')
echo ""
echo "=== Deployed ==="
echo "  Knative URL : ${KSVC_URL}"
echo "  Health      : ${KSVC_URL}/inference/health"
echo "  Test        : bash run-test.sh"
echo ""
echo "=== Handler versions available ==="
echo "  v1 — basic mean scoring"
echo "  v2 — adds confidence interval"
echo "  v3 — weighted ensemble + model agreement"
echo ""
echo "=== To deploy a new handler version (no rebuild) ==="
echo "  1. Edit handlers/v2/predict_handler.py"
echo "  2. ./manifests/generate-handlers-configmap.sh"
echo "  3. oc apply -f manifests/handlers-configmap.yaml"
echo "  4. oc delete job handlers-loader -n ${NAMESPACE} --ignore-not-found"
echo "  5. oc apply -f manifests/handlers-loader-job.yaml"
