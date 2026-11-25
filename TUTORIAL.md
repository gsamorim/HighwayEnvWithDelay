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

## 3. O começo do bloco define o experimento

`$config` aponta para o [arquivo JSONC do experimento](configs/racetrack_experiment_config.jsonc), que aceita comentários. Os scripts leem o arquivo com `--config`. Uma opção explícita no comando tem prioridade sobre a chave correspondente no JSONC. Uma chave omitida ou com valor `null` deixa o script usar seu próprio padrão.

`$saida` cria um caminho com data e hora para reunir os arquivos do grupo. `$retas` lista os três comprimentos a testar e também fornece as opções da pergunta no terminal. `$passos` fixa o mesmo orçamento de treino para as três rodadas.

### Pista, obstáculos e tráfego

| Chave no JSONC | Valor neste exemplo | O que controla |
| --- | --- | --- |
| `env` | `racetrack_oval` | Seleciona a entrada do registro do projeto que constrói o ambiente oval. Para mudar pelo terminal, use `--env`. |
| `env_id` | `racetrack-oval-v0` | Seleciona a versão da pista no HighwayEnv. `--env-id` substitui esse valor. Para usar uma pista comum, combine `"env": "racetrack"` com um ID compatível, como `racetrack-v0`, e deixe as opções exclusivas do oval como `null`. |
| `oval_straight_length_m` | `80`, `30` e `5`, um por rodada | Define o comprimento da reta em metros. O JSONC traz `30` e o laço substitui esse valor com `--oval-straight-length-m`. |
| `oval_obstacle_preset` | `null` | Mantém a pista sem um cenário de obstáculos adicional. Com `--oval-obstacle-preset`, `block_lane` coloca obstáculos na faixa central e `force_decision` coloca obstáculos nas faixas laterais do oval padrão. |
| `traffic` | `medium` | Controla a intensidade do tráfego. Os níveis `none`, `light`, `medium`, `heavy` e `jam` configuram limites de 0, 2, 5, 10 e 16 outros veículos. Use `--traffic` com um desses nomes ou um inteiro. A quantidade efetiva pode variar a cada episódio. |
| `traffic_speed_min` e `traffic_speed_max` | `null` para ambas | Mantêm a velocidade inicial nativa dos carros do tráfego. Use `--traffic-speed-min` e `--traffic-speed-max` juntas para sortear velocidades iniciais dentro de uma faixa em m/s. |

O `SafeSpawnWrapper` corrige a posição inicial do carro do agente quando o HighwayEnv o coloca fora da faixa. A correção permite usar as retas de 5 m e 30 m. Com tráfego, ainda pode haver colisão logo após a largada.

### Atraso de ação e de observação

| Chave no JSONC | Valor neste exemplo | O que controla |
| --- | --- | --- |
| `enable_delay_wrapper` | `true` | Ativa a simulação de atraso e acrescenta à observação o histórico de comandos, a fila de pendências, os tempos restantes e a última ação aplicada. `--enable-delay-wrapper` liga essa opção pelo terminal. |
| `action_delay_min_steps` e `action_delay_max_steps` | `0` e `3` | Sorteiam para cada comando um atraso inteiro de 0 a 3 passos de decisão. Use `--action-delay-min-steps` e `--action-delay-max-steps` para ajustar a faixa. O limite atual é 6 passos. Valores iguais tornam o atraso fixo. |
| `delay_known_to_agent` | `true` | Informa ao agente quantos passos faltam para cada comando pendente chegar. `--delay-known-to-agent` liga essa informação e `false` no JSONC oculta os tempos. |
| `observation_delay_steps` | `0` | Entrega a observação da pista sem atraso adicional. `--observation-delay-steps` define um atraso fixo para essa observação. |

O carro mantém o último comando aplicado enquanto espera um novo. Um comando antigo que chega depois que um mais recente já foi aplicado é descartado. A frequência padrão do projeto é de cinco decisões por segundo simulado, então três passos de atraso equivalem a 0,6 s. Para comparar com atraso de ação zero usando o mesmo formato de entrada, mantenha `"enable_delay_wrapper": true` e coloque os dois limites do atraso de ação em `0`.

## 4. O laço treina e salva um modelo por pista

`foreach` executa `scripts/train.py` uma vez para cada reta. Cada execução começa com um agente novo. `$nome` vira `oval80m`, `oval30m` ou `oval5m` e identifica os arquivos da rodada. A checagem de `$LASTEXITCODE` interrompe o grupo se algum comando falhar.

| Opção | Valor neste exemplo | O que controla |
| --- | --- | --- |
| `--agent` | `sac` | Usa o agente SAC (Soft Actor-Critic), que aprende a comandar aceleração e direção. `--agent random` serve como referência de ações aleatórias. |
| `--seed` / `seed` | `0` | Define a semente usada para iniciar o treino, incluindo a inicialização do SAC, o primeiro reset do ambiente e o sorteio dos atrasos. |
| `--run-name` / `run_name` | `oval80m`, `oval30m` ou `oval5m` | Dá nome à execução. Com `"run_name": null` no JSONC e sem essa opção, o script monta um nome automaticamente. |
| `--total-steps` / `total_steps` | `5000` | Define quantas ações a rodada envia ao ambiente. Substitui os `50000` do JSONC. Para uma demonstração mais curta, ajuste `$passos` para `2500` antes de rodar o bloco. |
| `--start-steps` | `300` | Usa ações aleatórias nos primeiros 300 passos para explorar e coletar experiência. |
| `--learning-starts` | `300` | Exige pelo menos 300 transições guardadas para o SAC atualizar a rede. O treinador inicia as atualizações depois da exploração inicial, quando também há dados suficientes para o lote. |
| `--checkpoint-every-steps` | `$passos` | Salva um checkpoint periódico no último passo do exemplo. O treinador salva também um arquivo `_final.pt` ao terminar. |

Para cada execução, o script grava os resultados nas subpastas abaixo. Os nomes ilustram a rodada de 80 m.

| Opção | Arquivo dentro de `$saida` | Conteúdo |
| --- | --- | --- |
| `--log-dir` | `logs/oval80m.csv` e `logs/oval80m.meta.json` | Métricas dos episódios de treino e configurações efetivas da rodada. |
| `--plot-dir` | `plots/oval80m.png` | Curvas de treino com retorno, colisões, velocidade e temperatura de entropia do SAC. |
| `--checkpoint-dir` | `checkpoints/oval80m_step5000.pt` e `checkpoints/oval80m_final.pt` | Checkpoint periódico e modelo final. A gravação usa o arquivo final. |

## 5. A avaliação ajuda a escolher o que gravar

| Opção | Valor neste exemplo | O que controla |
| --- | --- | --- |
| `--eval-every-steps` | `$passos` | Roda uma avaliação a cada intervalo de passos. Com o intervalo igual ao total, há uma avaliação no fim de cada treino. |
| `--eval-episodes` / `eval_episodes` | `2` | Roda dois episódios de avaliação com a política treinada, sem atualizar a rede. Substitui os cinco episódios do JSONC. |

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

`scripts/record.py` usa a resposta do terminal para definir o comprimento da reta e localizar o checkpoint correspondente.

| Opção | O que faz neste exemplo |
| --- | --- |
| `--config`, `--agent` e `--oval-straight-length-m` | Recriam o ambiente e o agente com o mesmo JSONC, o mesmo tipo de agente e a mesma reta do treino escolhido. Se você sobrepuser tráfego, obstáculos ou atraso no treino, repita esses ajustes na gravação. |
| `--checkpoint` | Carrega `checkpoints/<nome>_final.pt` do grupo atual. |
| `--episodes 3` | Grava três episódios novos do agente. O gravador começa pela semente do JSONC, `0` neste exemplo, e usa uma semente sucessiva para cada episódio. Você pode trocá-la com `--seed`. |
| `--out` | Salva o vídeo em `$saida/best_run.mp4`. `--fps` define os quadros por segundo do arquivo, 15 por padrão, e `--gif` também salva uma cópia em GIF. |

O gravador renderiza em segundo plano. `Invoke-Item` abre o MP4 no player padrão depois que o arquivo é salvo. Se aparecer `size mismatch` ao carregar o modelo, confira `enable_delay_wrapper`. Treino e gravação precisam usar o mesmo formato de observação.
