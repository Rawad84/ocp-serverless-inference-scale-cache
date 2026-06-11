#!/bin/bash
# Run the cache-aware scale-out ramp test.
# Deletes existing pods first so every run starts with a cold cache.
set -e

NAMESPACE=inference-benchmark
SERVICE=cache-aware-inference

BASE_URL=$(oc get ksvc ${SERVICE} -n ${NAMESPACE} -o jsonpath='{.status.url}' 2>/dev/null)
if [ -z "${BASE_URL}" ]; then
  echo "ERROR: could not get URL for ksvc ${SERVICE}. Is it deployed?"
  exit 1
fi
echo "=== Service URL: ${BASE_URL} ==="

echo "=== Resetting pod cache ==="
oc delete pods -l serving.knative.dev/service=${SERVICE} -n ${NAMESPACE} --ignore-not-found
sleep 3
oc wait --for=condition=Ready pod -l serving.knative.dev/service=${SERVICE} \
  -n ${NAMESPACE} --timeout=60s
echo "  Pod restarted — L1 cache is empty."

echo ""
echo "=== Watch scale-out in another terminal: ==="
echo "  watch oc get pods -n ${NAMESPACE} -l serving.knative.dev/service=${SERVICE}"
echo ""
python3 tests/test_scale_out.py --url "${BASE_URL}"
