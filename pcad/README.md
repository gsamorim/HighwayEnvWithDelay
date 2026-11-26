# Rodando o experimento no PCAD

Regras do PCAD que valem aqui: nada de processamento no frontend (só editar,
copiar e submeter); nada de colar scripts no terminal (o terminal indenta e
quebra `#SBATCH`); arquivos vão por `rsync`.

Fila: `draco` (2 × Xeon E5-2640 v2, 16 núcleos / 32 threads, 64 GB), livre
para qualquer conta. `cei`, `tsubasa`, `cidia`, `saude`, `grace` e `blaise`
são de projetos e exigem permissão do coordenador — não usar.

## 1. Enviar o código (da sua máquina)

```bash
rsync -avP --exclude .git --exclude '__pycache__' --exclude 'results/' \
    <pasta-local>/HighwayEnvWithDelay/ <usuario>@pcad.inf.ufrgs.br:~/RL/HighwayEnvWithDelay/
```

## 2. Submeter tudo (no frontend)

```bash
bash ~/RL/HighwayEnvWithDelay/pcad/submeter.sh
```

Submete três jobs encadeados:

| Job | O que faz | Duração |
|---|---|---|
| `setup.slurm` | cria `~/RL/venv`, roda os testes, mede passos/s com 20 treinos simultâneos | ~20 min |
| `experimento.slurm` C1+C2 | SAC e PPO × 5 seeds (20 treinos) num draco | ~12 h (estimado) |
| `experimento.slurm` C3+C4 | idem para C3+C4 no outro draco | ~12 h (estimado) |

Os dois últimos só começam se o setup terminar sem erro. Pode fechar o SSH.

## 3. Conferir o setup (~20 min depois)

```bash
squeue -u $USER
cat rl-setup_*.out        # me traga esta saída (e o .err se falhar)
```

Se a velocidade medida mostrar que não cabe em 23 h, cancele os dois jobs
(`scancel <id>`; os ids estão em `~/.rl_jobs`) e resubmeta com menos passos:

```bash
sbatch --export=ALL,CONDS="C1 C2",TIMESTEPS=200000 ~/RL/HighwayEnvWithDelay/pcad/experimento.slurm
sbatch --export=ALL,CONDS="C3 C4",TIMESTEPS=200000 ~/RL/HighwayEnvWithDelay/pcad/experimento.slurm
```

Se um job morrer (tempo, nó caiu), basta submeter de novo a mesma linha:
treinos com `model.zip` e avaliações com `eval_summary.csv` são pulados.

## 4. Acompanhar

```bash
squeue -u $USER
tail -n 3 ~/RL/results/logs/sac_C2_seed0.log     # curva a cada 50k passos
ls ~/RL/results/exp/*/eval_summary.csv | wc -l   # avaliações prontas (meta: 40)
```

## 5. Trazer os resultados (da sua máquina)

```bash
rsync -avP <usuario>@pcad.inf.ufrgs.br:~/RL/results/ <pasta-local>/results-pcad/
```
