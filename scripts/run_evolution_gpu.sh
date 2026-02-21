#!/bin/bash
# Genesis v2.0: WSL2 GPU 진화 루프 실행
# 사용법: wsl -d Ubuntu-24.04 -- bash /mnt/d/00.test/PAPER/EthicaAI/scripts/run_evolution_gpu.sh

set -e
source ~/ethicaai_env/bin/activate
cd /mnt/d/00.test/PAPER/EthicaAI

echo "===================================="
echo "🧬 Genesis v2.0 — GPU Evolution Loop"
echo "===================================="

# GPU 상태 확인
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
echo "📊 GPU Status (Preallocate=false, Fraction=0.80):"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "⚠️ No GPU detected, using CPU"

# JAX 백엔드 확인
echo ""
echo "📦 JAX Info:"
python3 -c "
import jax
print(f'  Backend: {jax.default_backend()}')
print(f'  Devices: {jax.devices()}')
"

echo ""
echo "🚀 Starting Evolution Loop..."
echo "===================================="

# 진화 루프 실행 (로그는 파일 + 터미널 동시 출력)
LOG_DIR="experiments/evolution"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gpu_evolution_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log: $LOG_FILE"
echo ""

python3 -u simulation/genesis/run_evolution.py 2>&1 | tee "$LOG_FILE"
