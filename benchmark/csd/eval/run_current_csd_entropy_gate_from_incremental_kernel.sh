#!/usr/bin/env bash
# Run the current CSD entropy-gate benchmark using the incrementally-built sgl-kernel.
#
# Build/update kernel first:
#   bash benchmark/csd/eval/run_current_csd_entropy_gate_from_incremental_kernel.sh build-kernel
#
# Smoke test the loaded kernel:
#   bash benchmark/csd/eval/run_current_csd_entropy_gate_from_incremental_kernel.sh test-kernel
#
# Run a small benchmark smoke test:
#   LIMIT=8 TASK_FILTER=gsm8k METHOD_SET="vanilla csd_plain_table" \
#     bash benchmark/csd/eval/run_current_csd_entropy_gate_from_incremental_kernel.sh bench
#
# Run the default benchmark set:
#   bash benchmark/csd/eval/run_current_csd_entropy_gate_from_incremental_kernel.sh bench

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

KERNEL_DIR="${REPO_ROOT}/sgl-kernel"
KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:-${KERNEL_DIR}/build}"
KERNEL_BUILD_TARGET="${KERNEL_BUILD_TARGET:-common_ops_sm90_build}"
KERNEL_PARALLEL="${KERNEL_PARALLEL:-48}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

export PATH="/root/miniconda3/envs/sglang/bin:${CUDA_HOME}/bin:/home/ccuser/.local/bin:${PATH}"
export CUDA_HOME
export CUDACXX
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Force Python to load the source-tree sglang + the local sgl-kernel package whose sm90
# common_ops.abi3.so is copied from ${KERNEL_BUILD_DIR}/sm90.
export SGLANG_SRC="${SGLANG_SRC:-${REPO_ROOT}/python}"
export SGL_KERNEL_SRC="${SGL_KERNEL_SRC:-${KERNEL_DIR}/python}"
export LIGHTEVAL_SRC="${LIGHTEVAL_SRC:-${REPO_ROOT}/benchmark/csd/lighteval/src}"
export PYTHONPATH="${SGL_KERNEL_SRC}:${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"

copy_kernel() {
  local src="${KERNEL_BUILD_DIR}/sm90/common_ops.abi3.so"
  local dst_dir="${SGL_KERNEL_SRC}/sgl_kernel/sm90"
  local dst="${dst_dir}/common_ops.abi3.so"
  if [[ ! -f "${src}" ]]; then
    echo "Built kernel not found: ${src}" >&2
    exit 1
  fi
  mkdir -p "${dst_dir}"
  cp "${src}" "${dst}"
  echo "Copied ${src} -> ${dst}"
}

build_kernel() {
  cmake --build "${KERNEL_BUILD_DIR}" --target "${KERNEL_BUILD_TARGET}" --parallel "${KERNEL_PARALLEL}"
  copy_kernel
}

show_kernel() {
  python - <<'PY'
import sgl_kernel
print('sgl_kernel package:', sgl_kernel.__file__)
print('common_ops module:', sgl_kernel.common_ops.__file__)
print('has tree op:', hasattr(sgl_kernel, 'tree_speculative_sampling_target_only'))
PY
}

test_kernel() {
  show_kernel
  python -m pytest "${KERNEL_DIR}/tests/speculative/test_speculative_sampling.py" -q
}

bench() {
  show_kernel
  export CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-1.3415851593017578}"
  export CSD_KEY_SELECTION_STRATEGY="${CSD_KEY_SELECTION_STRATEGY:-frequency}"
  export METHOD_SET="${METHOD_SET:-vanilla csd_plain_table}"
  export TASK_FILTER="${TASK_FILTER:-gsm8k}"
  export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_entropy_gate_current_kernel}"
  bash "${SCRIPT_DIR}/run_lighteval_csd_decoding_comparison_dyn.sh"
}

cmd="${1:-bench}"
case "${cmd}" in
  build-kernel)
    build_kernel
    ;;
  copy-kernel)
    copy_kernel
    ;;
  show-kernel)
    show_kernel
    ;;
  test-kernel)
    test_kernel
    ;;
  bench)
    bench
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    echo "Usage: bash $0 {build-kernel|copy-kernel|show-kernel|test-kernel|bench}" >&2
    exit 2
    ;;
esac
