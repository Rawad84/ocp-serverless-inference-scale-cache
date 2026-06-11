#!/bin/bash
# Build and deploy the subprocess-baseline inference service.
set -e

NAMESPACE=inference-benchmark
SERVICE=subprocess-inference

echo "=== Building ${SERVICE} image ==="
oc new-build --name=${SERVICE} \
  --binary --strategy=docker \
  -n ${NAMESPACE} 2>/dev/null || true
oc start-build ${SERVICE} --from-dir=./app --follow -n ${NAMESPACE}

echo "=== Applying Knative Service ==="
oc apply -f manifests/knative-service.yaml

echo "=== Waiting for Knative Service to be Ready ==="
oc wait --for=condition=Ready ksvc ${SERVICE} -n ${NAMESPACE} --timeout=180s

KSVC_URL=$(oc get ksvc ${SERVICE} -n ${NAMESPACE} -o jsonpath='{.status.url}')
echo ""
echo "=== Deployed ==="
echo "  Knative URL: ${KSVC_URL}"
echo "  Health:      ${KSVC_URL}/inference/health"
echo "  Test:        bash run-test.sh"
