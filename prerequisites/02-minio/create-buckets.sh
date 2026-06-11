#!/bin/bash
# Run after MinIO pod is Ready
# Creates the two buckets needed for the benchmark

MINIO_POD=$(oc get pod -n minio -l app=minio -o jsonpath='{.items[0].metadata.name}')

oc exec -n minio $MINIO_POD -- sh -c "
  mkdir -p /tmp/mc-config && \
  mc --config-dir /tmp/mc-config alias set local http://localhost:9000 minioadmin minioadmin123 && \
  mc --config-dir /tmp/mc-config mb local/inference-models && \
  mc --config-dir /tmp/mc-config mb local/inference-state && \
  mc --config-dir /tmp/mc-config ls local
"
