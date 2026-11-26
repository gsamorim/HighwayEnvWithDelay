# Experimentos configurados para o PCAD

Scripts em `pcad/` (roteiro de uso em `pcad/README.md`). Todo treino e toda
avaliação passam por `scripts/run_experiment.py`, que usa o ambiente de
`rl_platform/envs/delay_experiment.py`.

## Jobs

Submetidos juntos por `pcad/submeter.sh`, na fila `draco` (2 × Xeon E5-2640 v2,
16 núcleos / 32 threads, 64 GB por nó; um nó inteiro por job).

| Job | Script | O que roda | Duração estimada* |
|---|---|---|---|
| Preparação | `pcad/setup.slurm` | Cria o Python 3.12 com as versões fixadas, roda os testes do ambiente e mede passos/s com a carga real (10 SAC + 10 PPO ao mesmo tempo, 3 000 passos cada) | ~20 min |
| Experimento 1 | `pcad/experimento.slurm` com `CONDS="C1 C2"` | SAC e PPO × C1, C2 × 5 seeds = 20 treinos em paralelo + avaliação final | ~12 h |
| Experimento 2 | `pcad/experimento.slurm` com `CONDS="C3 C4"` | SAC e PPO × C3, C4 × 5 seeds = 20 treinos em paralelo + avaliação final | ~12 h |

Os experimentos só começam se o setup terminar sem erro.

\* Estimativa grosseira a partir do piloto local; será conferida com os
passos/s medidos pelo setup.

**Total: 40 treinos** (2 algoritmos × 4 condições × 5 seeds).

## Condições

| | Sem atraso no treino | Atraso aleatório no treino |
|---|---|---|
| Só direção | C1 | C2 |
| Direção + aceleração | C3 | C4 |

Atraso aleatório: `k_sorteado ~ U{0..3}` a cada passo,
`k_t = min(k_sorteado, k_{t−1} + 1)`, executa `a_{t−k_t}`.

## Configuração comum

| Item | Valor |
|---|---|
| Ambiente | `racetrack-oval-v1`, highway-env 1.12.1 |
| Reta | 30 m |
| Duração do episódio | 100 s (até 500 passos de 0,2 s) |
| Tráfego | `medium` (`other_vehicles=5`: 1 carro à frente + U{0..4} aleatórios) |
| Recompensa | `lmap(1/(1+4d²) − colisão − 0,3·\|a_direção\| + clip(v/10,0,1), [−1,2]→[0,1]) × na_pista` |
| Observação | grade de ocupação 2×12×12 + v/10 + últimas 6 ações emitidas + k/6 (296 dims só direção, 302 com aceleração; igual com e sem atraso) |
| Algoritmos | SAC e PPO do Stable-Baselines3 2.9.2, hiperparâmetros padrão, `MlpPolicy`, CPU, 1 thread |
| Passos de treino | 300 000 (ajustável com `TIMESTEPS`; o piloto estabilizou antes de 100 000) |
| Seeds de treino | 0, 1, 2, 3, 4 |

## Avaliação

- **Durante o treino:** a cada 50 000 passos, 10 episódios com k=0 (seeds 500 000+), o que gera a curva de aprendizado.
- **Final:** k fixo ∈ {0, 1, 2, 3, 4, 5, 6} (4–6 não vistos no treino), 100 episódios por k, política determinística, seeds 1 000 000 a 1 000 099 (diferentes das de treino).
- **Métricas por episódio:** retorno, duração, chegou aos 100 s, colisão, saída de pista, progresso ao longo da pista (m), voltas (1 volta ≈ 298 m), volta completa, velocidade média.

## Saída (em `~/RL/results/exp/<algo>_<condição>_seed<s>/`)

| Arquivo | Conteúdo |
|---|---|
| `model.zip` | modelo final |
| `meta.json` | configuração, versões, nó, job, tempo de treino, histograma de k no treino |
| `train_episodes.csv` | um registro por episódio de treino |
| `learning_curve.csv` | avaliações periódicas (k=0) |
| `eval_episodes.csv` | avaliação final, um registro por (k, episódio) |
| `eval_summary.csv` | médias por k |

Logs de cada treino: `~/RL/results/logs/<algo>_<condição>_seed<s>.log`.

## Fora dos scripts

- O piloto local (SAC e PPO, C1 e C3, seed 0, 100 000 passos) serviu só para confirmar que os algoritmos aprendem e para medir tempo; não entra no artigo.
- O `scripts/benchmark_delays.py` e os wrappers antigos (`DelayedActionMDPWrapper`, `HighSpeedRewardWrapper`) não são usados.
