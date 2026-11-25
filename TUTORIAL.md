# Treine e grave seu agente no Racetrack

Este tutorial compara três comprimentos de reta da pista oval. Você treina um agente SAC para cada pista, compara as avaliações no terminal e escolhe qual modelo gravar. O exemplo usa Windows e PowerShell.

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

do {
    $escolha = (Read-Host "Qual reta quer gravar? Opções: $($retas -join ', ')").Trim()
} until ($escolha -in $retas)

$nome = "oval${escolha}m"
$video = "$saida/best_run.mp4"

& $python scripts/record.py --config $config --agent sac `
    --oval-straight-length-m $escolha --checkpoint "$saida/checkpoints/${nome}_final.pt" `
    --episodes 3 --out $video

if ($LASTEXITCODE -ne 0) { throw "A gravação falhou." }
Invoke-Item $video
```

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
| `$retas = @("80", "30", "5")` | `$retas = @("80", "60", "5")` | Troca a reta de 30 m pela de 60 m. A pergunta de gravação oferece os valores dessa mesma lista. |
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

Para testar outra versão do oval com obstáculos, faça as duas substituições de `env_id` e `oval_obstacle_preset` no JSONC. Os comandos de treino e gravação já leem esse mesmo arquivo. Se usar uma pista comum, escolha um ID como `racetrack-v0`, deixe `oval_straight_length_m` e `oval_obstacle_preset` em `null` e remova `--oval-straight-length-m $reta` do treino e `--oval-straight-length-m $escolha` da gravação.

Para reproduzir o treino do tutorial, mantenha `"oval_obstacle_preset": null,`. Nos testes com `"block_lane"` e reta de 80 m, a observação inicial continha valores inválidos (`NaN`) para o SAC.

O `SafeSpawnWrapper` corrige largadas fora da faixa e procura uma posição afastada de obstáculos e outros carros. Em uma reta curta ou cheia, a folga pedida pode não caber. O wrapper procura um espaço na própria faixa, e a folga é avaliada no instante da largada.

### Controle da aceleração

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"action_longitudinal": null,` | `"action_longitudinal": false,` | Com `null` ou `true`, o SAC controla aceleração e direção. Com `false`, controla apenas a direção e o carro larga e segue a `speed_limit` durante a condução normal (10 m/s neste exemplo). |

Pelo PowerShell, acrescente `--no-action-longitudinal` à linha que começa com `& $python scripts/train.py --config $config --agent sac --seed 0`, antes do acento grave final. Repita a opção na linha que começa com `& $python scripts/record.py --config $config --agent sac`. Use `--action-longitudinal` para ativar a aceleração explicitamente.

Esse ajuste muda a ação de dois valores para um e também muda o tamanho da observação quando o wrapper de atraso está ligado. Treine e grave com a mesma configuração.

### Atraso de ação e de observação

| Linha atual no JSONC | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `"enable_delay_wrapper": true,` | `"enable_delay_wrapper": false,` | Desliga a simulação de atraso e retira da observação o histórico de ações, a fila, os tempos restantes e a última ação aplicada. Isso muda o formato de entrada do modelo. |
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

Este trecho começa em `foreach ($reta in $retas) {` e vai até a chave `}` imediatamente antes de `do {`. O laço executa `scripts/train.py` uma vez por reta, com um agente novo a cada rodada. A checagem de `$LASTEXITCODE` interrompe o grupo se o treino falhar. Na tabela, altere o trecho indicado na linha em que ele aparece.

| Trecho atual no PowerShell | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `--agent sac` | `--agent random` | Usa um agente de referência que toma ações aleatórias. Para gravar essa referência, use `--agent random` nos dois comandos e retire `--checkpoint` com seu caminho do gravador, pois o agente aleatório não salva pesos. |
| `--seed 0` | `--seed 1` | Muda a semente de inicialização do SAC, do primeiro reset e do sorteio dos atrasos. A opção prevalece sobre `"seed": 0,` no JSONC durante o treino. |
| `$nome = "oval${reta}m"` | `$nome = "teste_oval${reta}m"` | `--run-name $nome` usa esse nome, por exemplo `teste_oval80m`. O comando prevalece sobre `"run_name": null,` no JSONC. Repita o prefixo em `$nome = "teste_oval${escolha}m"` na gravação para encontrar os arquivos. |
| `--total-steps $passos` | Mantenha o trecho e troque `$passos = 5000` por `$passos = 2500` nas primeiras quatro linhas. | Reduz a quantidade de passos por treino mantendo a avaliação no último passo. O PowerShell prevalece sobre `"total_steps": 50000,` no JSONC. |
| `--start-steps 300` | `--start-steps 600` | Usa ações aleatórias nos primeiros 600 passos em vez de 300. |
| `--learning-starts 300` | `--learning-starts 600` | Exige pelo menos 600 transições guardadas para o SAC atualizar a rede. A atualização também depende de terminar a exploração inicial e ter dados suficientes para o lote. |
| `--checkpoint-every-steps $passos` | `--checkpoint-every-steps 2500` | Salva um checkpoint periódico a cada 2.500 passos. O treinador grava também o modelo `_final.pt` ao terminar. |

A última linha do comando de treino define as subpastas. Se quiser mover todos os resultados juntos, prefira mudar `$saida` no início do exemplo.

| Trecho atual no PowerShell | Exemplo de substituição | Arquivo gerado |
| --- | --- | --- |
| `--log-dir "$saida/logs"` | `--log-dir "$saida/csv"` | Muda a pasta do CSV com métricas de treino e do JSON com as configurações efetivas. |
| `--plot-dir "$saida/plots"` | `--plot-dir "$saida/figuras"` | Muda a pasta dos PNGs com retorno, colisões, velocidade e temperatura de entropia do SAC. |
| `--checkpoint-dir "$saida/checkpoints"` | `--checkpoint-dir "$saida/modelos"` | Muda a pasta dos checkpoints. Nesse caso, troque também `--checkpoint "$saida/checkpoints/${nome}_final.pt"` por `--checkpoint "$saida/modelos/${nome}_final.pt"` na gravação. |

## 5. A avaliação ajuda a escolher o que gravar

As opções de avaliação ficam na mesma linha do comando de treino que `--checkpoint-every-steps $passos`. A escolha do modelo aparece depois dos treinos, em `do {` até `} until ($escolha -in $retas)`.

| Trecho atual no PowerShell | Exemplo de substituição | O que muda |
| --- | --- | --- |
| `--eval-every-steps $passos` | `--eval-every-steps 2500` | Com 5.000 passos, avalia também no passo 2.500. Para comparar as pistas ao final, leia a linha do passo 5.000 de cada uma. |
| `--eval-episodes 2` | `--eval-episodes 5` | Roda cinco episódios de avaliação sem atualizar a rede. Essa opção prevalece sobre `"eval_episodes": 5,` no JSONC. |

A avaliação de cada pista aparece no terminal neste formato:

```text
[eval @ step    5000] return=35.34±7.54 crash_rate=0.00 mean_speed=6.14 (109s elapsed)
```

- `return` é a recompensa acumulada por episódio, resumida como **média ± desvio padrão**. Para seguir o critério da demonstração, escolha a pista com a maior média, antes do `±`.
- `crash_rate` é a fração dos episódios com colisão. Use a menor taxa para desempatar. Um episódio que termina por saída de pista pode ter zero colisões.
- `mean_speed` é a velocidade média em m/s. Use-a junto com o vídeo para entender como o agente dirigiu.
- `elapsed` é o tempo decorrido desde o início daquele treino, em segundos.

Ao final dos três treinos, digite a reta escolhida na pergunta do terminal. O PNG e o CSV mostram dados de treino. Para escolher a gravação, compare as linhas `[eval ...]`. Uma semente e dois episódios bastam para percorrer a demonstração. Para comparar desempenho, use mais sementes de treino e episódios de avaliação. A seção `_benchmark_protocol_reference` no JSONC reúne notas para essa comparação e é ignorada pelos scripts.

## 6. A última parte carrega o modelo e abre o vídeo

O trecho da gravação começa em `$nome = "oval${escolha}m"` e termina em `Invoke-Item $video`. `scripts/record.py` usa a resposta do terminal para escolher o comprimento da reta e localizar o checkpoint.

| Trecho atual no PowerShell | Onde e como editar | O que muda |
| --- | --- | --- |
| `--config $config` e `--agent sac` | Para trocar o arquivo, mude `$config = ...` nas primeiras quatro linhas. Para trocar o agente, use o mesmo valor de `--agent` no treino e na gravação. | Recria o ambiente e o tipo de agente do treino escolhido. Ajustes de pista, tráfego e atraso feitos no JSONC já são lidos pelos dois comandos. |
| `--oval-straight-length-m $escolha` | Escolha a reta na pergunta do terminal. Para oferecer outros valores, mude `$retas = ...` no início do bloco. | Usa a mesma geometria que foi usada para treinar o modelo escolhido. |
| `--checkpoint "$saida/checkpoints/${nome}_final.pt"` | Troque por `--checkpoint "$saida/checkpoints/${nome}_step5000.pt"` para usar o checkpoint periódico, se gerado. | Muda o arquivo de pesos carregado. |
| `--episodes 3` | `--episodes 1` | Grava um episódio em vez de três. O gravador começa pela semente do JSONC, `0` neste exemplo, e usa uma semente sucessiva por episódio. |
| `$video = "$saida/best_run.mp4"` | `$video = "$saida/demonstracao.mp4"` | Muda o nome usado por `--out $video` e por `Invoke-Item $video`. |
| `--episodes 3 --out $video` | `--episodes 3 --out $video --seed 123 --fps 10 --gif` | Começa pela semente 123, salva o MP4 a 10 quadros por segundo em vez de 15 e grava também um GIF. |

O gravador renderiza em segundo plano. `Invoke-Item` abre o MP4 no player padrão depois que o arquivo é salvo. Se aparecer `size mismatch` ao carregar o modelo, confira `enable_delay_wrapper`. Treino e gravação precisam usar o mesmo formato de observação.
