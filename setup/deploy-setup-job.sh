#!/bin/bash
# Deploy the setup-zip job to the cluster.
# Uploads setup_models.py as a ConfigMap then submits the Job.
set -e

NAMESPACE=inference-benchmark

echo "=== Uploading setup_models.py as ConfigMap ==="
oc create configmap setup-models-zip-script \
  --from-file=setup_models.py=setup-zip/setup_models.py \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | oc apply -f -

echo ""
echo "=== Deleting previous job if exists ==="
oc delete job setup-models-zip -n ${NAMESPACE} --ignore-not-found

echo ""
echo "=== Submitting Job ==="
oc apply -f setup-zip/setup-job.yaml

echo ""
echo "=== Waiting for job pod to start ==="
sleep 5
oc get pods -n ${NAMESPACE} -l component=setup-zip

echo ""
echo "=== Stream logs with: ==="
echo "  oc logs -f -l component=setup-zip -n ${NAMESPACE}"
