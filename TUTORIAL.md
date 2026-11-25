# Tutorial: editar config, rodar um grupo de experimentos e gravar o vídeo da melhor run

Este guia assume Windows + PowerShell (terminal integrado do VS Code). Os
tempos de treino abaixo (~2 min por run) foram medidos numa máquina de
referência a ~44-46 steps/seg (SAC, estado aumentado ligado); na sua máquina
pode variar — veja a nota de calibração no passo 3.

## 1. Abrir o projeto no VS Code

1. Abra o VS Code.
2. `File > Open Folder...` e selecione:
   `C:\Users\Uvood\OneDrive\Documentos\Claude Projects\rl_racetrack`
3. Abra um terminal integrado: `Terminal > New Terminal` (abre PowerShell
   nessa mesma pasta).
4. Ative o ambiente Python que você já usa para `scripts/train.py` (venv ou
   conda). Se ainda não tem um:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## 2. Editar o config único (`configs/racetrack_experiment_config.jsonc`)

No VS Code, abra `configs/racetrack_experiment_config.jsonc` (painel
esquerdo, pasta `configs/`). É o arquivo único que controla ambiente, pista,
tráfego e atraso — comentários `//` marcam o que não é válido em cada
ambiente. Hoje ele já vem configurado para:

- `"env": "racetrack_oval"`, `"env_id": "racetrack-oval-v0"`
- `"oval_straight_length_m": 30` (uma das 3 variações escolhidas: 80/30/5)
- `"enable_delay_wrapper": true`, atraso de ação sorteado entre 0 e 3 passos
  no treino (`action_delay_min/max_steps`), conhecido pelo agente
  (`delay_known_to_agent: true`)
- `"traffic": "medium"`

Para o grupo de experimentos deste tutorial, vamos comparar as **3 variações
do comprimento da reta do oval** (80 m, 30 m, 5 m) com o mesmo agente
(SAC + estado aumentado). Não precisa editar o arquivo pra isso — vamos
sobrepor `oval_straight_length_m` direto na linha de comando a cada run (o
arquivo fica como o "padrão de produção"; qualquer flag de CLI sempre
sobrepõe o que está no `.jsonc`). Mas se você quiser fixar uma variação como
novo padrão, é só editar o valor de `"oval_straight_length_m"` no arquivo e
salvar.

## 3. Rodar o grupo de experimentos (~2 min cada)

Os flags abaixo encurtam um treino de verdade (que seria `--total-steps`
bem maior, ~50k+) para uma demonstração rápida: menos passos totais, menos
aquecimento aleatório, avaliação só no final. Rode os 3 comandos em
sequência (cada um ~2 min na máquina de referência; total ~6 min):

```powershell
python scripts/train.py --config configs/racetrack_experiment_config.jsonc `
    --oval-straight-length-m 80 --total-steps 5000 --start-steps 300 `
    --learning-starts 300 --eval-every-steps 2500 --eval-episodes 2 `
    --checkpoint-every-steps 5000 --run-name oval80_demo

python scripts/train.py --config configs/racetrack_experiment_config.jsonc `
    --oval-straight-length-m 30 --total-steps 5000 --start-steps 300 `
    --learning-starts 300 --eval-every-steps 2500 --eval-episodes 2 `
    --checkpoint-every-steps 5000 --run-name oval30_demo

python scripts/train.py --config configs/racetrack_experiment_config.jsonc `
    --oval-straight-length-m 5 --total-steps 5000 --start-steps 300 `
    --learning-starts 300 --eval-every-steps 2500 --eval-episodes 2 `
    --checkpoint-every-steps 5000 --run-name oval5_demo
```

Cada run imprime no terminal uma linha final assim:

```
[eval @ step    5000] return=35.34±7.54 crash_rate=0.00 mean_speed=6.14 (109s elapsed)
```

**Calibração de tempo**: se o seu primeiro run demorar bem mais ou menos que
~2 min, ajuste `--total-steps` proporcionalmente nos próximos dois (ex.: se
demorou 4 min, use `--total-steps 2500` nos restantes).

Isso é uma demonstração, não um treino de verdade — com só 5000 passos o
SAC mal começou a aprender (o estudo de benchmark no próprio `.jsonc`,
chave `_benchmark_protocol_reference`, recomenda orçamentos bem maiores e
múltiplos seeds por configuração para resultados que valham a pena reportar).

## 4. Escolher a melhor run

Compare a linha final `[eval @ step 5000] return=...` impressa por cada um
dos 3 comandos (ou abra os CSVs em `results/logs/oval80_demo.csv` /
`oval30_demo.csv` / `oval5_demo.csv` — coluna `episode_return` ao longo do
treino). Escolha a que tiver maior `return` (e, em caso de empate, menor
`crash_rate`). O checkpoint correspondente está em
`results/checkpoints/<run_name>_final.pt`.

## 5. Gravar o vídeo da melhor run

Troque `oval30_demo` abaixo pelo nome da run vencedora:

```powershell
python scripts/record.py --config configs/racetrack_experiment_config.jsonc `
    --oval-straight-length-m 30 `
    --checkpoint results/checkpoints/oval30_demo_final.pt `
    --episodes 3 --out results/videos/best_run.mp4
```

Pontos importantes:

- **Use sempre `--config` (ou os mesmos flags) na gravação que você usou no
  treino daquela run**, especialmente `--enable-delay-wrapper`. Esse flag
  muda o tamanho da observação; gravar sem ele contra um checkpoint treinado
  com ele dá erro claro (`size mismatch ... torch.Size([256, 80])` vs
  `torch.Size([256, 48])`) em vez de rodar errado silenciosamente — já
  validei esse caso.
- `--oval-straight-length-m` na gravação só muda a geometria da pista
  (não o tamanho da observação), então pode inclusive usar um valor
  diferente do treino de propósito, para testar generalização.
- Mesmo com uma janela do Windows disponível, o `record.py` sempre usa o
  driver de vídeo "dummy" do SDL para renderizar — é esperado não abrir
  nenhuma janela; o vídeo sai direto em `results/videos/best_run.mp4`.

Abra o vídeo final no Explorer/VS Code: `results/videos/best_run.mp4`.

## Referência rápida de flags (todos opcionais, todos sobrepõem o `.jsonc`)

| Flag | O que faz |
|---|---|
| `--env racetrack` \| `racetrack_oval` | troca o registro de ambiente |
| `--env-id racetrack-oval-v1` | troca o id nativo do gym direto |
| `--oval-straight-length-m N` | comprimento da reta do oval (só `env_id` oval) |
| `--oval-obstacle-preset block_lane\|force_decision` | obstáculo nativo do oval |
| `--traffic-speed-min/--traffic-speed-max` | faixa de velocidade do tráfego (precisa dos dois juntos) |
| `--traffic none\|light\|medium\|heavy\|jam` | intensidade do tráfego |
| `--enable-delay-wrapper` | liga o estado aumentado com atraso (RDMDP) |
| `--action-delay-min/max-steps` | faixa do atraso de ação sorteado |
| `--delay-known-to-agent` | agente recebe o tempo restante real de cada comando pendente |
| `--observation-delay-steps` | atraso simples (sem reordenação) na observação |

Agentes disponíveis hoje: **só SAC** (`--agent sac`, default) e um baseline
aleatório (`--agent random`). PPO (e as variantes PPO+estado aumentado)
ainda não existem no repo — ficam para uma próxima versão.
