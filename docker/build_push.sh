#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-codemaivanngu/rlcsd:b200-cu13-vllm024-sm100-sm120}"

docker build \
  --platform linux/amd64 \
  --build-arg FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-100;120}" \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0;12.0+PTX}" \
  --build-arg CUDAARCHS="${CUDAARCHS:-100;120}" \
  --build-arg CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-100;120}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-4}" \
  --build-arg INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}" \
  -f docker/Dockerfile.rlc-runtime \
  -t "${IMAGE}" \
  .

docker run --rm --gpus all "${IMAGE}" python3 /workspace/RLCSD/docker/smoke_test.py
docker push "${IMAGE}"
