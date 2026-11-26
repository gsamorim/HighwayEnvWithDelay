#!/usr/bin/env bash
#
# Submete o setup e os dois jobs do experimento, encadeados, e sai.
# Rode no FRONTEND do PCAD:
#
#   bash ~/RL/HighwayEnvWithDelay/pcad/submeter.sh
#
# Os jobs do experimento só começam se o setup terminar sem erro
# (--dependency=afterok). Depois pode fechar o SSH.
#
# Divisão: um nó draco com C1+C2 e outro com C3+C4, cada um rodando
# SAC e PPO × 5 seeds (20 treinos por nó).

set -eu
AQUI=$(cd "$(dirname "$0")" && pwd)
command -v sbatch >/dev/null || { echo "sbatch não encontrado: rode no frontend do PCAD"; exit 1; }

echo "=== fila draco ==="
sinfo -p draco
echo

J0=$(sbatch --parsable "$AQUI/setup.slurm")
J1=$(sbatch --parsable --dependency=afterok:"$J0" --export=ALL,CONDS="C1 C2" "$AQUI/experimento.slurm")
J2=$(sbatch --parsable --dependency=afterok:"$J0" --export=ALL,CONDS="C3 C4" "$AQUI/experimento.slurm")

echo "submetidos:"
echo "  $J0  setup           (~20 min)"
echo "  $J1  experimento C1+C2 (SAC e PPO × 5 seeds) — depois de $J0"
echo "  $J2  experimento C3+C4 (SAC e PPO × 5 seeds) — depois de $J0"
echo "$J0 $J1 $J2" > "$HOME/.rl_jobs"
echo
squeue -u "$USER"
