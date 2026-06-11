#!/usr/bin/env bash
# Generates handlers-configmap.yaml from the handlers/ source directory.
# Run this whenever you change or add a handler version, then commit the result.
#
# Usage:
#   ./manifests/generate-handlers-configmap.sh
#   git add manifests/handlers-configmap.yaml
#   git commit -m "feat(handlers): update v2 preprocessing logic"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HANDLERS_DIR="${REPO_ROOT}/handlers"
OUT="${SCRIPT_DIR}/handlers-configmap.yaml"
NAMESPACE="inference-benchmark"

if [[ ! -d "${HANDLERS_DIR}" ]]; then
  echo "ERROR: handlers/ directory not found at ${HANDLERS_DIR}"
  exit 1
fi

# Collect all predict_handler.py files across version subdirectories
FROM_FILES=()
for version_dir in "${HANDLERS_DIR}"/*/; do
  version=$(basename "${version_dir}")
  handler="${version_dir}predict_handler.py"
  if [[ -f "${handler}" ]]; then
    # Key name: v1-predict_handler.py, v2-predict_handler.py, etc.
    FROM_FILES+=("--from-file=${version}-predict_handler.py=${handler}")
  fi
done

if [[ ${#FROM_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no predict_handler.py files found under ${HANDLERS_DIR}"
  exit 1
fi

echo "Generating ConfigMap from:"
for version_dir in "${HANDLERS_DIR}"/*/; do
  version=$(basename "${version_dir}")
  handler="${version_dir}predict_handler.py"
  [[ -f "${handler}" ]] && echo "  ${version} → ${handler}"
done

kubectl create configmap inference-handlers \
  --namespace="${NAMESPACE}" \
  --dry-run=client \
  -o yaml \
  "${FROM_FILES[@]}" > "${OUT}"

echo ""
echo "Written: ${OUT}"
echo "Versions included: $(ls "${HANDLERS_DIR}")"
