#!/bin/bash
#SBATCH --job-name=rl-exp
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=56G
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Roda, num nó só, todas as combinações ALGOS × CONDS × SEEDS em paralelo
# (1 processo e 1 thread por treino). Pede o nó draco inteiro (32 CPUs =
# 16 núcleos × 2 threads, 64 GB).
#
# Divisão usada (pcad/submeter.sh), 20 treinos por nó:
#   sbatch --export=ALL,CONDS="C1 C2" ~/RL/HighwayEnvWithDelay/pcad/experimento.slurm
#   sbatch --export=ALL,CONDS="C3 C4" ~/RL/HighwayEnvWithDelay/pcad/experimento.slurm
#
# Variáveis (opcionais): ALGOS ("sac ppo"), CONDS ("C1 C2 C3 C4"),
# SEEDS ("0 1 2 3 4"), TIMESTEPS (300000), TRAFFIC (medium),
# EVAL_EPISODES (100). Opções na linha do sbatch (-p, --time, --mem) valem
# sobre as linhas #SBATCH acima.
#
# Memória: SAC ~1 GB por treino com 300k passos (replay buffer), PPO ~0,5 GB.
#
# Pode ser resubmetido: treino com model.zip pronto não é refeito, e
# avaliação com eval_summary.csv pronto também não.

set -u
export LC_ALL=C
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ALGOS="${ALGOS:-${ALGO:-sac ppo}}"
TIMESTEPS="${TIMESTEPS:-300000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONDS="${CONDS:-C1 C2 C3 C4}"
TRAFFIC="${TRAFFIC:-medium}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

REPO="$HOME/RL/HighwayEnvWithDelay"
PY="$HOME/RL/venv/bin/python"
OUT="$HOME/RL/results/exp"
LOGS="$HOME/RL/results/logs"
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

echo "data      : $(date -Is)"
echo "job       : ${SLURM_JOB_ID:-?} em $(hostname), fila ${SLURM_JOB_PARTITION:-?}"
echo "algos=[$ALGOS] conds=[$CONDS] seeds=[$SEEDS] timesteps=$TIMESTEPS traffic=$TRAFFIC"
lscpu | grep -E "Model name|^CPU\(s\)"

run_one() {  # $1=algo $2=condição $3=semente
  local name="$1_$2_seed$3" dir="$OUT/$1_$2_seed$3"
  if [ -f "$dir/eval_summary.csv" ]; then
    echo "$name: já concluído"; return 0
  fi
  if [ -f "$dir/model.zip" ]; then
    "$PY" -W ignore scripts/run_experiment.py eval --run-dir "$dir" \
        --eval-episodes "$EVAL_EPISODES" >> "$LOGS/$name.log" 2>&1
  else
    "$PY" -W ignore scripts/run_experiment.py train --algo "$1" --condition "$2" --seed "$3" \
        --timesteps "$TIMESTEPS" --traffic "$TRAFFIC" --eval-episodes "$EVAL_EPISODES" \
        --out-dir "$OUT" --overwrite >> "$LOGS/$name.log" 2>&1
  fi
  echo "$name: saiu com código $? em $(date -Is)"
}

for a in $ALGOS; do
  for c in $CONDS; do
    for s in $SEEDS; do
      run_one "$a" "$c" "$s" &
    done
  done
done
wait

echo; echo "=== CONCLUÍDO $(date -Is) ==="
for a in $ALGOS; do for c in $CONDS; do for s in $SEEDS; do
  d="${a}_${c}_seed${s}"
  [ -f "$OUT/$d/eval_summary.csv" ] && echo "ok    $d" || echo "FALTA $d"
done; done; done
