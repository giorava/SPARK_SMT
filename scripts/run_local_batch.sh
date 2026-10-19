#!/bin/bash
# =============================================================================
# run_local_batch.sh — run SPARK batch processing on a single machine.
#
# Forwards all CLI args to src/batch_hpc_fast.py. The "hpc" in the script
# name is a misnomer when run locally — the same script handles both
# multiprocessing (-j N) on a workstation and SLURM array execution.
#
# Examples:
#   ./run_local_batch.sh params.csv -o out/
#   ./run_local_batch.sh params.csv -o out/ -j 8 --save-gt
#   ./run_local_batch.sh params.csv -o out/ -j 4 --auto-normalize
#
# Auto-detects Python in the same order as run_gui.sh.
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
BATCH_SCRIPT="${REPO_ROOT}/src/batch_hpc_fast.py"
BUNDLED_PY="${REPO_ROOT}/python_runtime/bin/python"

if [ -x "$BUNDLED_PY" ]; then
    PYTHON="$BUNDLED_PY"
    echo "Using bundled Python: $PYTHON"
    # See run_gui.sh for context — adds the bundled runtime's library
    # directory to the dynamic linker's search path, mirroring what the
    # .bat does for Windows DLLs. Belt-and-suspenders if conda-unpack
    # wasn't run.
    if [ "$(uname)" = "Darwin" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${REPO_ROOT}/python_runtime/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="${REPO_ROOT}/python_runtime/lib:${LD_LIBRARY_PATH:-}"
    fi
    export PATH="${REPO_ROOT}/python_runtime/bin:${PATH}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
    echo "Using system python3 (no bundled runtime found)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
    echo "Using system python (no bundled runtime found)"
else
    echo "ERROR: no Python interpreter found." >&2
    echo "       Either drop a bundled Python into ${REPO_ROOT}/python_runtime/" >&2
    echo "       or install Python 3.9+ on your system." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <params.csv> [options]"
    echo ""
    echo "Common options:"
    echo "  -o, --outdir DIR        Output directory (default: ./batch_output)"
    echo "  -j, --jobs N            Parallel workers (default: all CPUs)"
    echo "  --rows RANGE            Process specific rows, e.g. \"0-9\" or \"5\""
    echo "  --save-gt               Also save ground-truth TIFF stacks"
    echo "  --bit-depth {8,16}      Output bit depth (default: 8)"
    echo "  --auto-normalize        Two-phase global brightness normalization"
    echo "  --seed N                Base random seed"
    echo ""
    echo "Run with --help for the full option list."
    exit 1
fi

exec "$PYTHON" "$BATCH_SCRIPT" "$@"
