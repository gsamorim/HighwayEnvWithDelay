# Treine e grave seu agente no Racetrack

Este tutorial compara três comprimentos de reta da pista oval. Você treina um agente SAC para cada pista, compara as avaliações no terminal e grava três episódios com o modelo final de cada pista. O exemplo usa Windows e PowerShell.

## 1. Prepare o terminal

No VS Code, abra a pasta `rl_racetrack`, onde estão `scripts/` e `requirements.txt`. Abra **Terminal → New Terminal** e selecione o PowerShell.

Se já tem um ambiente Python com as dependências do projeto, ative-o e defina qual comando vai chamar o Python.

```powershell
$python = "python"
```

Se estiver começando do zero, crie o ambiente e instale as dependências.

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
```

Mantenha o mesmo terminal aberto durante o tutorial. Os próximos passos reutilizam as variáveis definidas nele.

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

Cole o bloco completo uma vez. Primeiro rodam os três treinos. Depois o bloco grava as três pistas automaticamente e abre a pasta de vídeos. Quando o prompt `PS ...>` voltar, o terminal está pronto para outro comando. As seções 3 a 6 mostram onde fica cada parte do bloco. Este exemplo usa `--agent sac` (o SAC implementado do zero neste projeto); a seção 7 lista os outros três agentes disponíveis e como trocar.

## 3. Primeiras quatro linhas do exemplo

Este trecho começa em `$config = ...` e termina em `$passos = 5000`, inclusive. A próxima linha de código, `foreach ($reta in $retas) {`, já inicia o treino.

```powershell
$config = "configs/racetrack_experiment_config.jsonc"
$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$retas = @("80", "30", "5")
$passos = 5000
```

| Linha a editar no PowerShell | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `$config = "configs/racetrack_experiment_config.jsonc"` | `$config = "configs/meu_experimento.jsonc"` | Faz treino e gravação lerem outro JSONC. Crie esse arquivo copiando o original antes de mudar o caminho. |
| `$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"` | `$saida = "results/experimento/$(Get-Date -Format 'yyyyMMdd-HHmmss')"` | Troca a pasta que reúne logs, gráficos, modelos e vídeo. A data e a hora identificam o grupo. |
| `$retas = @("80", "30", "5")` | `$retas = @("80", "60", "5")` | Troca a reta de 30 m pela de 60 m. O treino e a gravação percorrem essa mesma lista. |
| `$passos = 5000` | `$passos = 2500` | Encurta o treino de cada pista. No exemplo, a avaliação e o checkpoint periódico acompanham o novo total. |

Além dessas quatro linhas de PowerShell, o exemplo lê o [arquivo de configuração](configs/racetrack_experiment_config.jsonc). Para as opções abaixo, abra esse arquivo, encontre a chave com `Ctrl+F` e substitua o valor na linha existente. Mantenha as aspas de texto e a vírgula da linha. Uma opção explícita no PowerShell prevalece sobre a chave correspondente no JSONC. Quando uma chave está ausente ou vale `null`, o script usa seu padrão.

### Pista, obstáculos e tráfego

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"env": "racetrack_oval",` | `"env": "racetrack",` | Seleciona a entrada que monta o ambiente no projeto. O `env_id` explícito continua determinando qual pista será usada. |
| `"env_id": "racetrack-oval-v0",` | `"env_id": "racetrack-oval-v1",` | Troca a versão da pista do HighwayEnv. Esta linha fica logo abaixo de `"env": "racetrack_oval",`. |
| `"oval_straight_length_m": 30,` | `"oval_straight_length_m": 60,` | Troca a reta padrão para 60 m. Para trocar uma reta **neste exemplo**, mude a linha `$retas = @("80", "30", "5")` no PowerShell, pois os dois comandos passam o comprimento explicitamente. |
| `"oval_obstacle_preset": null,` | `"oval_obstacle_preset": "force_decision",` | Coloca obstáculos nas laterais do oval padrão. Já `"block_lane"` coloca obstáculos na faixa central. `null` desativa esses cenários. |
| `"obstacle_min_clearance_m": null,` | `"obstacle_min_clearance_m": 20.0,` | Pede 20 m de folga na largada em relação a obstáculos e outros veículos. `null` usa 15 m. A opção `--obstacle-min-clearance-m 20` tem o mesmo efeito. |
| `"traffic": "medium",` | `"traffic": "heavy",` | Aumenta a densidade de tráfego. `none`, `light`, `medium`, `heavy` e `jam` correspondem a limites de 0, 2, 5, 10 e 16 veículos. A quantidade efetiva pode variar a cada episódio. |
| `"traffic_speed_min": null,` | `"traffic_speed_min": 6.0,` | Define a menor velocidade inicial sorteada, em m/s. Preencha também a linha do máximo. |
| `"traffic_speed_max": null,` | `"traffic_speed_max": 9.0,` | Define a maior velocidade inicial sorteada, em m/s. Com as duas linhas em `null`, o tráfego usa a velocidade inicial nativa. |

Para testar outra versão do oval com obstáculos, faça as duas substituições de `env_id` e `oval_obstacle_preset` no JSONC. Os comandos de treino e gravação já leem esse mesmo arquivo. Se usar uma pista comum, escolha um ID como `racetrack-v0`, deixe `oval_straight_length_m` e `oval_obstacle_preset` em `null` e remova `--oval-straight-length-m $reta` dos dois comandos.

Para reproduzir o treino do tutorial, mantenha `"oval_obstacle_preset": null,`. Nos testes com `"block_lane"` e reta de 80 m, a observação inicial continha valores inválidos (`NaN`) para o SAC.

O `SafeSpawnWrapper` corrige largadas fora da faixa e procura uma posição afastada de obstáculos e outros carros. Em uma reta curta ou cheia, a folga pedida pode não caber. O wrapper procura um espaço na própria faixa, e a folga é avaliada no instante da largada.

### Controle da aceleração

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"action_longitudinal": null,` | `"action_longitudinal": false,` | Com `null` ou `true`, o agente controla aceleração e direção. Com `false`, controla apenas a direção e o carro larga e segue a `speed_limit` durante a condução normal (10 m/s neste exemplo). |

Pelo PowerShell, acrescente `--no-action-longitudinal` à linha que começa com `& $python scripts/train.py --config $config --agent sac --seed 0`, antes do acento grave final. Repita a opção na linha que começa com `& $python scripts/record.py --config $config --agent sac`. Use `--action-longitudinal` para ativar a aceleração explicitamente.

Esse ajuste muda a ação de dois valores para um e também muda o tamanho da observação quando o wrapper de atraso está ligado. Treine e grave com a mesma configuração.

### Atraso de ação e de observação

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"enable_delay_wrapper": true,` | `"enable_delay_wrapper": false,` | Desliga a simulação de atraso e retira da observação o histórico de ações, a fila, os tempos restantes e a última ação aplicada. Isso muda o formato de entrada do modelo. **É o toggle do "estado aumentado" citado na seção 7 — funciona do mesmo jeito para qualquer um dos quatro agentes.** |
| `"action_delay_min_steps": 0,` | `"action_delay_min_steps": 2,` | Com o máximo original em 3, passa a sortear atrasos de 2 a 3 passos por comando. |
| `"action_delay_max_steps": 3,` | `"action_delay_max_steps": 6,` | Com o mínimo original em 0, amplia a faixa para 0 a 6 passos. Use mínimo menor ou igual ao máximo, com os dois entre 0 e 6. Valores iguais dão atraso fixo. |
| `"delay_known_to_agent": true,` | `"delay_known_to_agent": false,` | Zera os tempos restantes mostrados ao agente. O atraso e o formato de entrada continuam ativos. |
| `"observation_delay_steps": 0,` | `"observation_delay_steps": 2,` | A observação do ambiente chega dois passos depois. O valor `0` entrega a observação sem atraso adicional. |

O carro mantém o último comando aplicado enquanto espera um novo. Um comando antigo que chega depois que um mais recente já foi aplicado é descartado. A frequência padrão do projeto é de cinco decisões por segundo simulado, então três passos de atraso equivalem a 0,6 s. Para comparar com atraso de ação zero usando o mesmo formato de entrada, mantenha `"enable_delay_wrapper": true` e coloque os dois limites do atraso de ação em `0`.

### Recompensa do treino e da avaliação

Essas linhas ficam sob `Reward shaping` no JSONC. No exemplo completo, `null` mantém os padrões de `scripts/train.py`. Os ajustes da tabela podem ser feitos individualmente, desde que a curva esteja em `"tanh"` para usar os dois parâmetros específicos dela.

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"high_speed_reward": null,` | `"high_speed_reward": 0.6,` | Aumenta de 0,4 para 0,6 o peso do termo de velocidade. Use `0` para desativá-lo. |
| `"collision_reward": null,` | `"collision_reward": -8.0,` | Aumenta a penalidade de colisão. O peso padrão é −5,0. |
| `"action_reward": null,` | `"action_reward": -0.2,` | Aumenta o custo de ações de maior magnitude. O peso padrão é −0,1. |
| `"speed_reward_shape": null,` | `"speed_reward_shape": "tanh",` | Troca a curva linear padrão por uma curva em S. O termo pode ser negativo quando o carro está parado ou muito lento e se aproxima de um teto em alta velocidade. |
| `"speed_reward_v0_frac": null,` | `"speed_reward_v0_frac": 0.2,` | Com `"tanh"`, faz o termo cruzar zero a 20% do limite de velocidade, em vez dos 10% padrão. |
| `"speed_reward_steepness_frac": null,` | `"speed_reward_steepness_frac": 0.2,` | Com `"tanh"`, torna a subida em torno desse ponto mais rápida que no padrão de 0,3. |

O `HighSpeedRewardWrapper` usa a **velocidade com sinal** do carro. Acelerar altera a velocidade ao longo do tempo. Já `action_reward` controla separadamente o custo da ação de acelerar ou esterçar.

- **`linear` (padrão)** usa `coef * clip(speed / speed_limit, 0, 1.2)`. Com o peso padrão positivo, o termo é zero parado ou de ré, aumenta com a velocidade à frente e fica limitado a `1.2 * coef`. Ficar parado na pista não rende bônus de velocidade.
- **`tanh`** usa `coef * tanh((speed - v0) / steepness)`. O termo é negativo abaixo de `v0`, vale zero em `v0` e cresce mais rápido perto desse ponto. Ele se aproxima de `-coef` em marcha a ré e de `+coef` em alta velocidade, com ganho cada vez menor.

`v0` e `steepness` são as frações do limite de velocidade da tabela multiplicadas por `speed_limit`. Com o limite padrão de 10 m/s e as frações padrão de 0,1 e 0,3, o cruzamento ocorre a 1 m/s e a escala da transição é de 3 m/s. Com `coef = 0.4`, o termo `tanh` vale aproximadamente −0,13 quando o carro está parado. Fora da pista, o wrapper zera esse termo nos dois formatos.

**Onde ativar pelo PowerShell.** No comando `scripts/train.py`, localize a linha com `--total-steps $passos --start-steps 300 --learning-starts 300`. Substitua essa linha pelas duas abaixo, preservando o acento grave no fim de cada uma.

```powershell
        --total-steps $passos --start-steps 300 --learning-starts 300 `
        --speed-reward-shape tanh --speed-reward-v0-frac 0.1 --speed-reward-steepness-frac 0.3 `
```

Esse exemplo ativa `tanh` com as frações padrão. No JSONC, a mudança equivalente é trocar `"speed_reward_shape": null,` por `"speed_reward_shape": "tanh",` e deixar as duas frações em `null`. O formato afeta o treino e a avaliação que o treino imprime. O gravador carrega a política treinada e usa a recompensa padrão do ambiente para o retorno exibido no vídeo.

## 4. O laço treina e salva um modelo por pista

Este é o primeiro `foreach ($reta in $retas) {` do exemplo completo. Ele termina na chave `}` antes do segundo `foreach`, que faz as gravações.

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

O laço executa `scripts/train.py` uma vez por reta, com um agente novo a cada rodada. A checagem de `$LASTEXITCODE` interrompe o grupo se o treino falhar. Na tabela, altere o trecho indicado na linha em que ele aparece.

| Trecho atual no PowerShell | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `--agent sac` | `--agent random` | Usa um agente de referência que toma ações aleatórias. Para gravar essa referência, use `--agent random` nos dois comandos e retire `--checkpoint` com seu caminho do gravador, pois o agente aleatório não salva pesos. A seção 7 lista os outros dois agentes (`ppo`, `sac_sb3`) e o detalhe do checkpoint `.zip` deles. |
| `--seed 0` | `--seed 1` | Muda a semente de inicialização do SAC, do primeiro reset e do sorteio dos atrasos. A opção prevalece sobre `"seed": 0,` no JSONC durante o treino. |
| `$nome = "oval${reta}m"` | `$nome = "teste_oval${reta}m"` | `--run-name $nome` usa esse nome, por exemplo `teste_oval80m`. O comando prevalece sobre `"run_name": null,` no JSONC. Faça a mesma troca no laço de gravação para encontrar os arquivos. |
| `--total-steps $passos` | Mantenha o trecho e troque `$passos = 5000` por `$passos = 2500` nas primeiras quatro linhas. | Reduz a quantidade de passos por treino mantendo a avaliação no último passo. O PowerShell prevalece sobre `"total_steps": 50000,` no JSONC. |
| `--start-steps 300` | `--start-steps 600` | Usa ações aleatórias nos primeiros 600 passos em vez de 300. |
| `--learning-starts 300` | `--learning-starts 600` | Exige pelo menos 600 transições guardadas para o SAC atualizar a rede. A atualização também depende de terminar a exploração inicial e ter dados suficientes para o lote. |
| `--checkpoint-every-steps $passos` | `--checkpoint-every-steps 2500` | Salva um checkpoint periódico a cada 2.500 passos. O treinador grava também o modelo `_final.pt` ao terminar (`_final.zip` para `--agent ppo`/`--agent sac_sb3` — ver seção 7). |

A última linha do comando de treino define as subpastas. Se quiser mover todos os resultados juntos, prefira mudar `$saida` no início do exemplo.

| Trecho atual no PowerShell | Exemplo de substituição | Arquivo gerado |
| --- | --- | --- |
| `--log-dir "$saida/logs"` | `--log-dir "$saida/csv"` | Muda a pasta do CSV com métricas de treino e do JSON com as configurações efetivas. |
| `--plot-dir "$saida/plots"` | `--plot-dir "$saida/figuras"` | Muda a pasta dos PNGs com retorno, colisões, velocidade e temperatura de entropia do SAC. |
| `--checkpoint-dir "$saida/checkpoints"` | `--checkpoint-dir "$saida/modelos"` | Muda a pasta dos checkpoints. Nesse caso, troque também `--checkpoint "$saida/checkpoints/${nome}_final.pt"` por `--checkpoint "$saida/modelos/${nome}_final.pt"` na gravação. |

## 5. A avaliação permite comparar as pistas

### Linha de avaliação dentro do treino

Esta linha faz parte do comando `scripts/train.py` mostrado na seção 4.

```powershell
        --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
```

| Trecho atual no PowerShell | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `--eval-every-steps $passos` | `--eval-every-steps 2500` | Com 5.000 passos, avalia também no passo 2.500. Para comparar as pistas ao final, leia a linha do passo 5.000 de cada uma. |
| `--eval-episodes 2` | `--eval-episodes 5` | Roda cinco episódios de avaliação sem atualizar a rede. Essa opção prevalece sobre `"eval_episodes": 5,` no JSONC. |

### Saída da avaliação no terminal

O treino imprime uma linha como esta para cada pista.

```text
[eval @ step    5000] return=35.34±7.54 crash_rate=0.00 mean_speed=6.14 (109s elapsed)
```

- `return` é a recompensa acumulada por episódio, resumida como **média ± desvio padrão**. Compare a média antes do `±` para ver qual pista teve maior retorno nos episódios avaliados.
- `crash_rate` é a fração dos episódios com colisão. Um episódio que termina por saída de pista pode ter zero colisões.
- `mean_speed` é a velocidade média em m/s. Use-a junto com o vídeo para entender como o agente dirigiu.
- `elapsed` é o tempo decorrido desde o início daquele treino, em segundos.

O PNG e o CSV mostram dados de treino. Use as linhas `[eval ...]` para comparar a avaliação das pistas. O exemplo usa uma semente de treino e dois episódios de avaliação por pista. A seção `_benchmark_protocol_reference` no JSONC reúne notas para comparações com mais sementes e episódios.

## 6. Um vídeo por pista

Este é o segundo `foreach` do exemplo completo. Ele começa após os treinos e termina em `Invoke-Item "$saida/videos"`.

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

Para cada reta, `scripts/record.py` carrega o checkpoint `_final.pt` correspondente e reúne três episódios novos em um único MP4. Com a configuração do exemplo, a pasta `$saida/videos` recebe `oval80m.mp4`, `oval30m.mp4` e `oval5m.mp4`. Cada vídeo usa as sementes 0, 1 e 2 para seus três episódios.

| Trecho atual no PowerShell | Onde e como editar | O que muda |
| --- | --- | --- |
| `--config $config` e `--agent sac` | Para trocar o arquivo, mude `$config = ...` nas primeiras quatro linhas. Para trocar o agente, use o mesmo valor de `--agent` no treino e na gravação. | Recria o ambiente e o tipo de agente de cada treino. Ajustes de pista, tráfego e atraso feitos no JSONC já são lidos pelos dois comandos. |
| `--oval-straight-length-m $reta` | Para testar outros comprimentos, mude `$retas = ...` no início do bloco. | Grava cada modelo na mesma geometria usada no seu treino. |
| `--checkpoint "$saida/checkpoints/${nome}_final.pt"` | Troque por `--checkpoint "$saida/checkpoints/${nome}_step5000.pt"` para usar o checkpoint periódico, se gerado. | Muda o arquivo de pesos carregado. Com `--agent ppo`/`--agent sac_sb3` a extensão é `.zip`, não `.pt` (seção 7). |
| `--episodes 3` | `--episodes 1` | Grava um episódio por pista. Esse valor é independente de `--eval-episodes 2` no treino. |
| `$video = "$saida/videos/$nome.mp4"` | `$video = "$saida/videos/demo_$nome.mp4"` | Acrescenta `demo_` a cada nome de vídeo. `$nome` identifica a pista no arquivo. |
| `--episodes 3 --out $video` | `--episodes 3 --out $video --seed 123 --fps 10 --gif` | Começa pela semente 123, salva o MP4 a 10 quadros por segundo em vez de 15 e grava também um GIF. |

O gravador renderiza em segundo plano. A última linha abre a pasta dos MP4s. Se aparecer `size mismatch` ao carregar o modelo, confira `enable_delay_wrapper` e `action_longitudinal`. Treino e gravação precisam usar os mesmos formatos de observação e ação.

## 7. Os quatro agentes e as sete combinações (random, SAC×2, PPO, × estado aumentado)

Além do SAC caseiro usado no exemplo das seções 2 a 6, o projeto tem mais três agentes selecionáveis por `--agent` (ou pela chave `"agent"` do JSONC): um agente de referência que não aprende, e duas implementações que usam a biblioteca **stable-baselines3 (SB3)** em vez de código próprio.

| `--agent` | O que é | Checkpoint |
| --- | --- | --- |
| `sac` (padrão) | SAC implementado do zero neste projeto (Haarnoja et al. 2018/2019) — nunca usou SB3, é uma implementação própria num arquivo só. | `_final.pt` |
| `random` | Agente de referência: toma ações aleatórias, não aprende nada. Não gera checkpoint. | — |
| `ppo` | PPO via `stable-baselines3` (biblioteca pronta, não reimplementado). | `_final.zip` |
| `sac_sb3` | SAC via `stable-baselines3` — um segundo SAC, útil para comparar com o `sac` caseiro. | `_final.zip` |

O "estado aumentado" (`"enable_delay_wrapper": true/false` no JSONC, seção 3) é um ajuste do **ambiente**, não do agente: funciona do mesmo jeito para `sac`, `ppo` e `sac_sb3`, sem precisar trocar mais nada. Isso dá sete combinações com sentido prático (o `random` não aprende, então o estado aumentado não muda o que ele faz):

| # | Agente | Estado aumentado | `--agent` |
| --- | --- | --- | --- |
| 1 | random | — | `random` |
| 2 | SAC (caseiro) | não | `sac` |
| 3 | SAC (caseiro) | sim | `sac` + `"enable_delay_wrapper": true` |
| 4 | SAC (via SB3) | não | `sac_sb3` |
| 5 | SAC (via SB3) | sim | `sac_sb3` + `"enable_delay_wrapper": true` |
| 6 | PPO (via SB3) | não | `ppo` |
| 7 | PPO (via SB3) | sim | `ppo` + `"enable_delay_wrapper": true` |

### Dependência extra: `stable-baselines3`

`--agent ppo` e `--agent sac_sb3` precisam do pacote `stable-baselines3`, já listado em `requirements.txt` (`pip install -r requirements.txt` da seção 1 já instala). Sem esse pacote instalado, `--agent sac` e `--agent random` continuam funcionando normalmente — só `ppo`/`sac_sb3` dão um erro claro pedindo para instalar o pacote, no momento em que o agente é de fato usado (agir, salvar ou carregar), não ao iniciar o script.

### Como rodar o exemplo da seção 2 com PPO ou com SAC-via-SB3

Troque `--agent sac` por `--agent ppo` ou `--agent sac_sb3` nas DUAS linhas (treino e gravação), e troque a extensão do checkpoint de `.pt` para `.zip` na linha de gravação. Nada mais no bloco da seção 2 precisa mudar.

```powershell
& $python scripts/train.py --config $config --agent ppo --seed 0 `
    --oval-straight-length-m $reta --run-name $nome `
    --total-steps $passos --start-steps 300 --learning-starts 300 `
    --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
    --log-dir "$saida/logs" --plot-dir "$saida/plots" --checkpoint-dir "$saida/checkpoints"
```

```powershell
& $python scripts/record.py --config $config --agent ppo `
    --oval-straight-length-m $reta --checkpoint "$saida/checkpoints/${nome}_final.zip" `
    --episodes 3 --out $video
```

Para `--agent sac_sb3`, use exatamente os mesmos dois comandos acima trocando `ppo` por `sac_sb3` — o checkpoint também é `.zip`.

### Hiperparâmetros específicos de cada agente SB3 (opcionais)

PPO e SAC-via-SB3 têm hiperparâmetros próprios, configuráveis por flag de CLI ou pela chave correspondente no JSONC (todas `null` por padrão = usa o padrão do próprio `stable-baselines3`). `--buffer-size`, `--batch-size` e `--learning-starts` (já usados pelo `sac` caseiro) valem também para `sac_sb3`, com o mesmo significado.

| Agente | Flags | Chaves no JSONC |
| --- | --- | --- |
| `ppo` | `--ppo-n-steps`, `--ppo-n-epochs`, `--ppo-learning-rate`, `--ppo-gamma`, `--ppo-gae-lambda`, `--ppo-clip-range`, `--ppo-ent-coef`, `--ppo-vf-coef` | `ppo_n_steps`, `ppo_n_epochs`, `ppo_learning_rate`, `ppo_gamma`, `ppo_gae_lambda`, `ppo_clip_range`, `ppo_ent_coef`, `ppo_vf_coef` |
| `sac_sb3` | `--sac-sb3-learning-rate`, `--sac-sb3-gamma`, `--sac-sb3-tau`, `--sac-sb3-train-freq`, `--sac-sb3-gradient-steps`, `--sac-sb3-ent-coef`, `--sac-sb3-target-update-interval` | `sac_sb3_learning_rate`, `sac_sb3_gamma`, `sac_sb3_tau`, `sac_sb3_train_freq`, `sac_sb3_gradient_steps`, `sac_sb3_ent_coef`, `sac_sb3_target_update_interval` |

PPO e SAC-via-SB3 não usam o mesmo laço de treino do `sac` caseiro — o treino de ambos roda pelo próprio `model.learn()` do `stable-baselines3` (PPO é on-policy e SAC-via-SB3 não expõe seu replay buffer para fora). Na prática isso é transparente: o console imprime `[eval @ step ...]` no mesmo formato, o CSV/PNG de treino têm as mesmas colunas, e `scripts/record.py` carrega qualquer um dos quatro agentes sem diferença de uso — só muda a extensão do arquivo de checkpoint.

## 8. Benchmark de atraso com cinco sementes

Use o PowerShell da seção 1. Este exemplo usa `racetrack-oval-v1`, com **HighwayEnv 1.12.1** como versão de referência. A reta tem 80 m, o tráfego é médio e o agente só controla a direção, com velocidade fixa de 10 m/s durante a condução normal. A duração máxima é **100 s simulados**, ou **500 decisões** a 5 Hz. Colisões ou saídas da pista podem encerrar o episódio antes.

No [script do benchmark](scripts/benchmark_delays.py), `ENV_ID`, `STRAIGHT_LENGTH_M` e `POLICY_HZ` definem a pista, a reta e a frequência. Em `make_benchmark_env()`, `traffic_density="medium"` e `action_longitudinal=False` definem o tráfego e o controle.

### Exemplo completo

```powershell
$saida = "results/benchmark/atrasos-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

& $python scripts/benchmark_delays.py --output-dir $saida --duration 100 `
    --total-steps 500000 --seeds 0 1 2 3 4 --train-max-delay 3 `
    --eval-delays 0 1 2 3 4 5 6 --eval-episodes 100

if ($LASTEXITCODE -ne 0) { throw "O benchmark de atraso falhou." }
```

### Trecho que define o treino

```powershell
    --total-steps 500000 --seeds 0 1 2 3 4 --train-max-delay 3 `
```

| Opção do trecho | O que faz |
| --- | --- |
| `--total-steps 500000` | Treina cada política por 500.000 passos do ambiente. Esse é o orçamento de uma execução, por variante e por semente. |
| `--seeds 0 1 2 3 4` | Usa cinco sementes de treino para cada variante. |
| `--train-max-delay 3` | Treina o SAC de referência **sem atraso** e adiciona o **SAC+atrasos**. Nessa segunda variante, cada comando recebe um atraso uniforme sorteado entre 0, 1, 2 e 3 passos. O agente observa também o estado aumentado, com histórico, fila e tempos restantes. |

As duas variantes usam o mesmo algoritmo SAC. O exemplo compara uma política treinada sem atraso com outra que foi treinada com atraso e tem acesso ao estado aumentado. As cinco sementes por variante produzem **10 políticas finais e 5 milhões de passos de treino** no total.

### Trecho que define a avaliação

```powershell
    --eval-delays 0 1 2 3 4 5 6 --eval-episodes 100
```

| Opção do trecho | O que faz |
| --- | --- |
| `--eval-delays 0 1 2 3 4 5 6` | Aplica sete atrasos **fixos**, um de cada vez, de 0 a 6 passos. A 5 Hz, 6 passos representam 1,2 s simulados. |
| `--eval-episodes 100` | Roda 100 episódios novos para cada política treinada e cada atraso. Usa as sementes de avaliação 100000 a 100099 em todas as condições. São 700 episódios por política e **7.000 no total**. |

O script avalia os pesos finais de cada treino nos sete atrasos, sem atualizá-los. Durante a avaliação com atraso, o SAC de referência sofre o atraso nas ações e continua recebendo a observação original. O SAC+atrasos recebe a observação aumentada em todos os níveis, inclusive no zero. Os avisos `[eval @ step ...]` durante o treino mostram avaliações intermediárias na configuração de treino.

### Trecho que define onde salvar

```powershell
$saida = "results/benchmark/atrasos-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

& $python scripts/benchmark_delays.py --output-dir $saida --duration 100 `
```

`--duration 100` é o tempo máximo de cada episódio, tanto em treino quanto em avaliação. Em `$saida` ficam os modelos finais em `checkpoints`, os logs em `logs`, os gráficos de treino em `plots` e `avaliacao.csv`. Esse CSV tem uma linha por política, semente de treino e atraso de avaliação, com retorno médio, desvio padrão, taxa de colisão, velocidade média e duração média em passos. A pasta de saída precisa ser nova.

## 9. Ensaio rápido do SAC sem atraso

Use a mesma pista, o mesmo controle apenas de direção e o mesmo limite de 100 s por episódio. Este ensaio usa **uma semente**, **5.000 passos de treino** e **10 episódios de avaliação**. Ele permite conferir o fluxo e inspecionar os resultados antes de rodar o benchmark maior.

### Exemplo completo

```powershell
$saida = "results/benchmark/sem-atraso-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

& $python scripts/benchmark_delays.py --output-dir $saida --duration 100 `
    --total-steps 5000 --seeds 0 --train-max-delay 0 `
    --eval-delays 0 --eval-episodes 10

if ($LASTEXITCODE -ne 0) { throw "O ensaio sem atraso falhou." }
```

### Trecho que encurta o ensaio

```powershell
    --total-steps 5000 --seeds 0 --train-max-delay 0 `
    --eval-delays 0 --eval-episodes 10
```

`--train-max-delay 0` roda apenas o SAC de referência, com a observação original e comandos imediatos. `--eval-delays 0` mantém essa condição nos dez episódios de avaliação. A linha `--output-dir $saida --duration 100` direciona os arquivos para a pasta `sem-atraso-...` e mantém a duração do cenário principal.
