#!/bin/bash
#SBATCH --job-name=rl-setup
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Prepara o ambiente Python e mede a velocidade de treino no nó.
#
#   sbatch ~/RL/HighwayEnvWithDelay/pcad/setup.slurm
#   (normalmente submetido por pcad/submeter.sh)
#
# 1. instala o uv (gerenciador de Python, binário único) em ~/.local/bin;
# 2. cria ~/RL/venv com Python 3.12 e as versões de requirements-pcad.txt;
# 3. roda os testes do ambiente do experimento;
# 4. mede passos/s com a MESMA carga do experimento: 20 treinos ao mesmo
#    tempo (10 SAC + 10 PPO, 3000 passos cada), para calibrar --time e
#    TIMESTEPS.
#
# Precisa de internet no nó (para baixar uv, Python e pacotes). Se falhar
# nessa etapa, me traga o .err.

set -eu
export LC_ALL=C

REPO="$HOME/RL/HighwayEnvWithDelay"
VENV="$HOME/RL/venv"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/RL/results"

echo "data : $(date -Is)"
echo "job  : ${SLURM_JOB_ID:-?} em $(hostname), fila ${SLURM_JOB_PARTITION:-?}"
echo "CPUs alocadas (nproc): $(nproc)"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
free -g | head -2
echo "python do sistema: $(python3 --version 2>&1 || echo ausente)"

if ! command -v uv >/dev/null; then
  echo; echo "=== instalando uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

echo; echo "=== criando venv ==="
uv venv --clear --python 3.12 "$VENV"
# torch primeiro, do índice só-CPU (o do PyPI traz CUDA, ~2 GB à toa);
# o resto vem do PyPI e aceita o torch 2.15.0+cpu já instalado.
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
uv pip install --python "$VENV/bin/python" -r "$REPO/pcad/requirements-pcad.txt"
"$VENV/bin/python" -c "import torch, highway_env, stable_baselines3 as s; print('ok', torch.__version__, s.__version__)"

export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"

echo; echo "=== testes ==="
"$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_delay_experiment.py

echo; echo "=== velocidade: 10 SAC + 10 PPO ao mesmo tempo, 3000 passos cada ==="
OUT="$HOME/RL/results/setup_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT/logs"
for a in sac ppo; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    "$VENV/bin/python" -W ignore scripts/run_experiment.py train --algo $a --condition C4 --seed $s \
        --timesteps 3000 --eval-every 0 --skip-eval --out-dir "$OUT" > "$OUT/logs/${a}_$s.log" 2>&1 &
  done
done
wait
"$VENV/bin/python" - "$OUT" <<'EOF'
import json, sys, glob
for a in ("sac", "ppo"):
    metas = [json.load(open(f)) for f in glob.glob(f"{sys.argv[1]}/{a}_*/meta.json")]
    # passos EXECUTADOS (o PPO arredonda 3000 para 4096 = 2 rollouts de 2048)
    sps = [m["timesteps_executed"] / m["train_wall_time_s"] for m in metas]
    print(f"{a}: {len(sps)} treinos, {min(sps):.1f}-{max(sps):.1f} passos/s (média {sum(sps)/len(sps):.1f}); "
          f"300k passos ≈ {300000 / (sum(sps)/len(sps)) / 3600:.1f} h")
EOF
echo; echo "=== CONCLUÍDO ==="
