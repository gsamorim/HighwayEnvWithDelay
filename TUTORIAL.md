# Treine e grave seu agente no Racetrack

Treine um agente por pista, compare as avaliações e gere um vídeo por pista no PowerShell.

## Índice

- [1. Preparo do terminal](#tut-preparo)
- [2. Exemplo completo](#tut-exemplo)
- [3. Configuração](#tut-config)
- [4. Treino](#tut-treino)
- [5. Avaliação](#tut-avaliacao)
- [6. Gravação](#tut-gravacao)
- [7. Agentes](#tut-agentes)
- [Exemplos extras](#tut-extras)
  - [8. Benchmark com atraso](#tut-com-atraso)
  - [9. Ensaio rápido sem atraso](#tut-sem-atraso)

<a name="tut-preparo"></a>

## 1. Preparo do terminal

Abra a pasta `rl_racetrack` no VS Code e use o PowerShell integrado. Se já tem um ambiente preparado, ative-o e use:

```powershell
$python = "python"
```

Para configurar o projeto pela primeira vez:

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
```

Use o mesmo terminal nos próximos passos.

<a name="tut-exemplo"></a>

## 2. Exemplo completo

```powershell
$config = "configs/racetrack_experiment_config.jsonc"
$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$retas = @("80", "30", "5")
$passos = 5000

foreach ($reta in $retas) {
    $nome = "oval${reta}m"
    Write-Host "Treinando $nome. Resultados em $saida"

    & $python scripts/train.py --config $config --agent sac --seed 0 `
        --oval-straight-length-m $reta --run-name $nome `
        --total-steps $passos --start-steps 300 --learning-starts 300 `
        --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
        --log-dir "$saida/logs" --plot-dir "$saida/plots" --checkpoint-dir "$saida/checkpoints"

    if ($LASTEXITCODE -ne 0) { throw "O treino $nome falhou." }
}

foreach ($reta in $retas) {
    $nome = "oval${reta}m"
    $video = "$saida/videos/$nome.mp4"
    Write-Host "Gravando $nome. Vídeo em $video"

    & $python scripts/record.py --config $config --agent sac `
        --oval-straight-length-m $reta --checkpoint "$saida/checkpoints/${nome}_final.pt" `
        --episodes 3 --out $video

    if ($LASTEXITCODE -ne 0) { throw "A gravação $nome falhou." }
}

Invoke-Item "$saida/videos"
```

Cole o bloco uma vez. Ele treina as três pistas, grava três episódios por pista e abre a pasta dos vídeos. Quando o prompt `PS ...>` voltar, o terminal está pronto para outro comando.

<a name="tut-config"></a>

## 3. Configuração

As quatro primeiras linhas definem o arquivo de configuração, a pasta de saída, as retas e o orçamento de treino. O treino começa no `foreach` seguinte.

```powershell
$config = "configs/racetrack_experiment_config.jsonc"
$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$retas = @("80", "30", "5")
$passos = 5000
```

| Linha do exemplo | Exemplo de troca | Efeito |
| --- | --- | --- |
| `$config = "configs/racetrack_experiment_config.jsonc"` | `$config = "configs/meu_experimento.jsonc"` | Usa uma cópia que você preparou do arquivo original. |
| `$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"` | `$saida = "results/experimento/$(Get-Date -Format 'yyyyMMdd-HHmmss')"` | Muda a pasta que reúne os resultados. |
| `$retas = @("80", "30", "5")` | `$retas = @("80", "60", "5")` | Usa 60 m no lugar de 30 m no treino e na gravação. |
| `$passos = 5000` | `$passos = 2500` | Treina por 2.500 passos em cada pista. |

Para editar o [JSONC do experimento](configs/racetrack_experiment_config.jsonc), localize a linha indicada abaixo. `null` usa o padrão do script. Uma opção explícita no PowerShell tem prioridade.

### Pista e tráfego

| Linha atual no JSONC | Exemplo de troca | Efeito |
| --- | --- | --- |
| `"env": "racetrack_oval",` | `"env": "racetrack",` | Troca a entrada do projeto que cria o ambiente. O `env_id` explícito define a pista. |
| `"env_id": "racetrack-oval-v0",` | `"env_id": "racetrack-oval-v1",` | Troca a versão da pista oval. |
| `"oval_straight_length_m": 30,` | `"oval_straight_length_m": 60,` | Muda a reta padrão. No exemplo completo, altere `$retas`. |
| `"oval_obstacle_preset": null,` | `"oval_obstacle_preset": "force_decision",` | Coloca obstáculos nas laterais. `"block_lane"` os coloca na faixa central. |
| `"obstacle_min_clearance_m": null,` | `"obstacle_min_clearance_m": 20.0,` | Pede 20 m de folga na largada. O padrão é 15 m, sujeito ao espaço disponível. |
| `"traffic": "medium",` | `"traffic": "heavy",` | Aumenta o tráfego. Os presets `none`, `light`, `medium`, `heavy` e `jam` usam limites de 0, 2, 5, 10 e 16 veículos. |
| `"traffic_speed_min": null,` | `"traffic_speed_min": 6.0,` | Define o mínimo da velocidade inicial do tráfego, em m/s. |
| `"traffic_speed_max": null,` | `"traffic_speed_max": 9.0,` | Define o máximo. Preencha os dois limites juntos. |

O exemplo usa `"oval_obstacle_preset": null,`. Em teste com a reta de 80 m, `block_lane` gerou observações inválidas para o SAC. Para uma pista comum, use um `env_id` como `racetrack-v0`, deixe o comprimento e o preset do oval em `null` e retire `--oval-straight-length-m $reta` dos dois comandos.

### Aceleração e atraso

| Linha atual no JSONC | Exemplo de troca | Efeito |
| --- | --- | --- |
| `"action_longitudinal": null,` | `"action_longitudinal": false,` | O agente passa a controlar só a direção, a 10 m/s durante a condução normal. |
| `"enable_delay_wrapper": true,` | `"enable_delay_wrapper": false,` | Desliga o atraso e o estado aumentado. |
| `"action_delay_min_steps": 0,` | `"action_delay_min_steps": 2,` | Com máximo 3, sorteia atrasos de 2 a 3 passos por comando. |
| `"action_delay_max_steps": 3,` | `"action_delay_max_steps": 6,` | Com mínimo 0, sorteia atrasos de 0 a 6 passos. |
| `"delay_known_to_agent": true,` | `"delay_known_to_agent": false,` | Oculta os tempos restantes dos comandos pendentes. |
| `"observation_delay_steps": 0,` | `"observation_delay_steps": 2,` | Atrasa a observação em dois passos. |

No PowerShell, `--no-action-longitudinal` equivale a `"action_longitudinal": false,`. Na primeira linha de cada chamada de `scripts/train.py` e `scripts/record.py`, acrescente a opção antes do acento grave final. Use `--action-longitudinal` para ativar a aceleração. A aceleração e o estado aumentado mudam as dimensões do modelo. Mantenha essas opções iguais no treino e na gravação.

Use `0 ≤ mínimo ≤ máximo ≤ 6`. Valores iguais dão atraso fixo. Para testar atraso zero mantendo o estado aumentado, use `"enable_delay_wrapper": true,` e coloque os dois limites em `0`.

### Recompensa

| Linha atual no JSONC | Exemplo de troca | Efeito |
| --- | --- | --- |
| `"high_speed_reward": null,` | `"high_speed_reward": 0.6,` | Muda o peso do bônus de velocidade, por padrão 0,4. |
| `"collision_reward": null,` | `"collision_reward": -8.0,` | Aumenta a penalidade por colisão, por padrão −5. |
| `"action_reward": null,` | `"action_reward": -0.2,` | Aumenta o custo da ação, por padrão −0,1. |
| `"speed_reward_shape": null,` | `"speed_reward_shape": "tanh",` | Usa uma curva em S que penaliza ficar parado ou andar de ré. A curva linear padrão dá bônus zero nesses casos. |
| `"speed_reward_v0_frac": null,` | `"speed_reward_v0_frac": 0.2,` | Na curva `tanh`, muda de 10% para 20% do limite de velocidade o ponto onde o bônus cruza zero. |
| `"speed_reward_steepness_frac": null,` | `"speed_reward_steepness_frac": 0.2,` | Na curva `tanh`, diminui a escala de 0,3 para 0,2 e torna a subida mais íngreme. |

Para ativar `tanh` no PowerShell, substitua a linha com `--learning-starts 300` no comando de treino por estas duas:

```powershell
        --total-steps $passos --start-steps 300 --learning-starts 300 `
        --speed-reward-shape tanh --speed-reward-v0-frac 0.1 --speed-reward-steepness-frac 0.3 `
```

Esses ajustes valem para treino e avaliação. O retorno exibido no vídeo usa a recompensa padrão.

<a name="tut-treino"></a>

## 4. Treino

O primeiro `foreach` treina e salva um modelo por pista.

```powershell
foreach ($reta in $retas) {
    $nome = "oval${reta}m"
    Write-Host "Treinando $nome. Resultados em $saida"

    & $python scripts/train.py --config $config --agent sac --seed 0 `
        --oval-straight-length-m $reta --run-name $nome `
        --total-steps $passos --start-steps 300 --learning-starts 300 `
        --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
        --log-dir "$saida/logs" --plot-dir "$saida/plots" --checkpoint-dir "$saida/checkpoints"

    if ($LASTEXITCODE -ne 0) { throw "O treino $nome falhou." }
}
```

| Trecho do PowerShell | Exemplo de troca | Efeito |
| --- | --- | --- |
| `--agent sac` | `--agent ppo` | Troca o agente. Repita na gravação e use a extensão indicada na [seção 7](#tut-agentes). |
| `--seed 0` | `--seed 1` | Muda a semente de treino. |
| `$nome = "oval${reta}m"` | `$nome = "teste_oval${reta}m"` | Muda o prefixo dos arquivos. Repita a troca no segundo laço. |
| `--total-steps $passos` | `$passos = 2500` no início do bloco | Reduz os passos de treino por pista. |
| `--start-steps 300` | `--start-steps 600` | Aumenta o período inicial de ações aleatórias do SAC caseiro. |
| `--learning-starts 300` | `--learning-starts 600` | Exige mais transições antes de começar as atualizações do SAC. |
| `--checkpoint-every-steps $passos` | `--checkpoint-every-steps 2500` | Salva um checkpoint a cada 2.500 passos. |
| `--log-dir "$saida/logs"` | `--log-dir "$saida/csv"` | Muda a pasta dos CSVs e metadados. |
| `--plot-dir "$saida/plots"` | `--plot-dir "$saida/figuras"` | Muda a pasta dos PNGs. |
| `--checkpoint-dir "$saida/checkpoints"` | `--checkpoint-dir "$saida/modelos"` | Muda a pasta dos modelos. Use a mesma pasta no `--checkpoint` da gravação. |

<a name="tut-avaliacao"></a>

## 5. Avaliação

Esta linha está dentro do primeiro `foreach`.

```powershell
        --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
```

Com `--eval-every-steps $passos`, a avaliação ocorre no fim do treino. Para avaliar também no passo 2.500 de uma execução de 5.000 passos, use `--eval-every-steps 2500`. Troque `--eval-episodes 2` por `--eval-episodes 5` para avaliar cinco episódios.

Saída no terminal:

```text
[eval @ step    5000] return=35.34±7.54 crash_rate=0.00 mean_speed=6.14 (109s elapsed)
```

- `return` mostra o retorno médio e o desvio padrão.
- `crash_rate` mostra a fração dos episódios com colisão. Sair da pista pode terminar o episódio com zero colisões.
- `mean_speed` mostra a velocidade média em m/s.
- `elapsed` mostra o tempo de execução em segundos.

<a name="tut-gravacao"></a>

## 6. Gravação

O segundo `foreach` carrega o modelo final de cada pista e grava três episódios. A última linha abre a pasta dos vídeos.

```powershell
foreach ($reta in $retas) {
    $nome = "oval${reta}m"
    $video = "$saida/videos/$nome.mp4"
    Write-Host "Gravando $nome. Vídeo em $video"

    & $python scripts/record.py --config $config --agent sac `
        --oval-straight-length-m $reta --checkpoint "$saida/checkpoints/${nome}_final.pt" `
        --episodes 3 --out $video

    if ($LASTEXITCODE -ne 0) { throw "A gravação $nome falhou." }
}

Invoke-Item "$saida/videos"
```

A pasta `$saida/videos` recebe `oval80m.mp4`, `oval30m.mp4` e `oval5m.mp4`. Cada vídeo usa as sementes 0, 1 e 2 no exemplo.

| Trecho do PowerShell | Exemplo de troca | Efeito |
| --- | --- | --- |
| `--checkpoint "$saida/checkpoints/${nome}_final.pt"` | `--checkpoint "$saida/checkpoints/${nome}_step5000.pt"` | Usa o checkpoint periódico, se existir. |
| `--episodes 3` | `--episodes 1` | Grava um episódio por pista. |
| `$video = "$saida/videos/$nome.mp4"` | `$video = "$saida/videos/demo_$nome.mp4"` | Adiciona `demo_` aos nomes dos vídeos. |
| `--episodes 3 --out $video` | `--episodes 3 --out $video --seed 123 --fps 10 --gif` | Muda a semente inicial e a taxa de quadros, e gera também um GIF. |

Se o modelo der erro de dimensão ao carregar, confira se `enable_delay_wrapper` e `action_longitudinal` são iguais no treino e na gravação.

<a name="tut-agentes"></a>

## 7. Agentes

| # | Agente | `--agent` | `enable_delay_wrapper` | Modelo final |
| --- | --- | --- | --- | --- |
| 1 | Aleatório | `random` | — | — |
| 2 | SAC do projeto | `sac` | `false` | `.pt` |
| 3 | SAC do projeto | `sac` | `true` | `.pt` |
| 4 | SAC via SB3 | `sac_sb3` | `false` | `.zip` |
| 5 | SAC via SB3 | `sac_sb3` | `true` | `.zip` |
| 6 | PPO via SB3 | `ppo` | `false` | `.zip` |
| 7 | PPO via SB3 | `ppo` | `true` | `.zip` |

No exemplo completo, troque `--agent sac` nas chamadas de treino e gravação. Para PPO ou SAC via SB3, troque também `${nome}_final.pt` por `${nome}_final.zip` no caminho do modelo. Para `random`, retire `--checkpoint` e seu caminho da chamada de gravação.

Se preferir escolher o agente no JSONC, troque `"agent": null,` por, por exemplo, `"agent": "ppo",` e retire `--agent sac` dos dois comandos. Defina `enable_delay_wrapper` no mesmo JSONC de acordo com a tabela.

<a name="tut-extras"></a>

## Exemplos extras

Os dois exemplos usam `racetrack-oval-v1`, reta de 80 m, tráfego médio e direção a 10 m/s. Cada episódio dura até 100 s simulados (500 passos). O script salva modelos, logs, gráficos e o CSV de avaliação em `$saida`. [Código do benchmark](scripts/benchmark_delays.py)

<a name="tut-com-atraso"></a>

### 8. Benchmark com atraso

```powershell
$saida = "results/benchmark/atrasos-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

& $python scripts/benchmark_delays.py --output-dir $saida --duration 100 `
    --total-steps 500000 --seeds 0 1 2 3 4 --train-max-delay 3 `
    --eval-delays 0 1 2 3 4 5 6 --eval-episodes 100

if ($LASTEXITCODE -ne 0) { throw "O benchmark de atraso falhou." }
```

Estas duas linhas definem o treino e a avaliação.

```powershell
    --total-steps 500000 --seeds 0 1 2 3 4 --train-max-delay 3 `
    --eval-delays 0 1 2 3 4 5 6 --eval-episodes 100
```

- `--total-steps 500000 --seeds 0 1 2 3 4` treina por 500.000 passos por variante e semente.
- `--train-max-delay 3` treina SAC sem atraso e SAC com estado aumentado e atrasos uniformes de 0 a 3 passos.
- `--eval-delays 0 1 2 3 4 5 6 --eval-episodes 100` avalia cada política final em 100 episódios por atraso fixo. O SAC de referência usa a observação original durante a avaliação, inclusive com atraso.

O total é de **5 milhões de passos de treino e 7.000 episódios de avaliação final**. Confira `avaliacao.csv` para comparar as políticas.

<a name="tut-sem-atraso"></a>

### 9. Ensaio rápido sem atraso

```powershell
$saida = "results/benchmark/sem-atraso-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

& $python scripts/benchmark_delays.py --output-dir $saida --duration 100 `
    --total-steps 5000 --seeds 0 --train-max-delay 0 `
    --eval-delays 0 --eval-episodes 10

if ($LASTEXITCODE -ne 0) { throw "O ensaio sem atraso falhou." }
```

Trecho que define o ensaio:

```powershell
    --total-steps 5000 --seeds 0 --train-max-delay 0 `
    --eval-delays 0 --eval-episodes 10
```

Treina um SAC por 5.000 passos com a semente 0 e roda dez episódios de avaliação. Os dois atrasos ficam em zero.
