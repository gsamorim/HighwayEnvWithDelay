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
# Fila shared. Divisão: um nó com C1+C2 (32 CPUs) e outro com C3+C4 (20 CPUs), cada um rodando
# SAC e PPO × 5 seeds (20 treinos por nó).

set -eu
AQUI=$(cd "$(dirname "$0")" && pwd)
command -v sbatch >/dev/null || { echo "sbatch não encontrado: rode no frontend do PCAD"; exit 1; }

echo "=== fila shared (CPUs ocupadas/livres/outras/total) ==="
sinfo -p shared -o "%t %N %C %m"
echo

J0=$(sbatch --parsable "$AQUI/setup.slurm")
J1=$(sbatch --parsable --dependency=afterok:"$J0" --export=ALL,CONDS="C1 C2" "$AQUI/experimento.slurm")
# 20 CPUs no segundo job: cabe num nó com 20 CPUs livres (ex.: draco6 parcialmente ocupado)
J2=$(sbatch --parsable --cpus-per-task=20 --dependency=afterok:"$J0" --export=ALL,CONDS="C3 C4" "$AQUI/experimento.slurm")

echo "submetidos:"
echo "  $J0  setup           (~20 min)"
echo "  $J1  experimento C1+C2 (SAC e PPO × 5 seeds) — depois de $J0"
echo "  $J2  experimento C3+C4 (SAC e PPO × 5 seeds) — depois de $J0"
echo "$J0 $J1 $J2" > "$HOME/.rl_jobs"
echo
squeue -a -u "$USER" -o "%.8i %.10j %.9T %.22r %.20S %N"
