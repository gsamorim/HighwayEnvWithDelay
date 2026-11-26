#!/bin/bash
#SBATCH --job-name=rl-exp
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=28G
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Roda, num nó só, todas as combinações ALGOS × CONDS × SEEDS em paralelo
# (1 processo e 1 thread por treino). Fila `shared` (aceita qualquer conta;
# a fila `draco` exige grupo). Padrão: 32 CPUs (um nó inteiro, draco ou bali2)
# e 28 GB (20 treinos de 300k usam ~15 GB); submeter.sh pede 20 CPUs no
# segundo job para caber num nó parcialmente ocupado.
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
# Pode ser resubmetido: o `run_experiment.py run` pula o que já está completo
# (confere metadados, os 7 atrasos e o nº de episódios), só avalia modelos
# prontos e recusa (código 3) pastas de outra configuração. O job termina com
# código 1 se qualquer treino ou avaliação falhar.

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

run_one() {  # $1=algo $2=condição $3=semente; o run_experiment.py decide pular/avaliar/treinar
  "$PY" -W ignore scripts/run_experiment.py run --algo "$1" --condition "$2" --seed "$3" \
      --timesteps "$TIMESTEPS" --traffic "$TRAFFIC" --eval-episodes "$EVAL_EPISODES" \
      --out-dir "$OUT" >> "$LOGS/$1_$2_seed$3.log" 2>&1
}

declare -A PIDS
for a in $ALGOS; do
  for c in $CONDS; do
    for s in $SEEDS; do
      run_one "$a" "$c" "$s" &
      PIDS["${a}_${c}_seed${s}"]=$!
    done
  done
done

# Espera cada treino pelo PID e guarda o código de saída de cada um.
FALHAS=0
for name in "${!PIDS[@]}"; do
  wait "${PIDS[$name]}"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "ok     $name  ($(date +%T))"
  else
    echo "FALHOU $name  código $rc  (ver $LOGS/$name.log)"; FALHAS=$((FALHAS + 1))
  fi
done

echo; echo "=== CONCLUÍDO $(date -Is): ${#PIDS[@]} treinos, $FALHAS falhas ==="
[ "$FALHAS" -eq 0 ] || exit 1
