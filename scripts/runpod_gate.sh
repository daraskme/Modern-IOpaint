#!/bin/bash
# RunPod release-gate validation script. Run on a fresh CUDA pod:
#   bash scripts/runpod_gate.sh
# Results are served over HTTP on port 8000 (run.log, results-*.md, DONE/FAILED markers).

mkdir -p /srv && cd /srv && (python3 -m http.server 8000 >/dev/null 2>&1 &)
exec > /srv/run.log 2>&1
set -x
echo "=== POD START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd /root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
git clone --depth 1 https://github.com/daraskme/Modern-IOpaint.git app
cd /root/app
uv venv --python 3.12 .venv
curl -L -o /root/wheel.whl https://github.com/daraskme/Modern-IOpaint/releases/download/v0.1.0b1/modern_iopaint-0.1.0b1-py3-none-any.whl
uv pip install --python .venv /root/wheel.whl
.venv/bin/modern-iopaint setup-gpu || { echo "SETUP_GPU_FAILED"; touch /srv/FAILED; exit 1; }
.venv/bin/python -c "from nunchaku.utils import get_precision; print('precision:', get_precision())"
echo "=== downloading models ==="
.venv/bin/python -c "from modern_iopaint.download import cli_download_model; cli_download_model('lama'); cli_download_model('qwen-image')" || { echo "DOWNLOAD_FAILED"; touch /srv/FAILED; exit 1; }
echo "=== bench quick (real 12GB) ==="
.venv/bin/python scripts/bench_vram.py --quick --precision int4 && .venv/bin/python scripts/bench_vram.py --quick --precision int4 --cap-gib 12
cp benchmarks/results-*.md /srv/ 2>/dev/null
echo "=== smoke_p2 auto profile ==="
.venv/bin/python scripts/smoke_p2.py --skip-edit > /srv/smoke_p2.log 2>&1
tail -5 /srv/smoke_p2.log
echo "=== DONE $(date -u) ==="
touch /srv/DONE
sleep 3600
