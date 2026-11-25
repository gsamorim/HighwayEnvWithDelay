# Treine e grave seu agente no Racetrack

Compare três versões da pista oval, escolha o melhor resultado da demonstração e grave um MP4 com o agente treinado. O roteiro usa Windows e PowerShell. Cada rodada treina um agente SAC do zero, com a mesma configuração de tráfego e atraso.

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

## 2. Rode as três variações da pista

A [configuração do experimento](configs/racetrack_experiment_config.jsonc) vem com uma reta de **30 m**, tráfego médio e atraso de ação sorteado entre **0 e 3 passos**. O agente recebe o tempo restante dos comandos pendentes.

O bloco abaixo testa retas de **80 m, 30 m e 5 m**. A opção `--oval-straight-length-m` muda a pista para cada rodada, e o arquivo de configuração segue com seus valores salvos. O treino imprime a avaliação no fim de cada rodada e guarda os resultados em uma pasta com data e hora.

```powershell
$config = "configs/racetrack_experiment_config.jsonc"
$saida = "results/tutorial/$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$passos = 5000

foreach ($reta in 80, 30, 5) {
    $nome = "oval$($reta)m"
    Write-Host "Treinando $nome. Resultados em $saida"

    & $python scripts/train.py --config $config --agent sac --seed 0 `
        --oval-straight-length-m $reta --run-name $nome `
        --total-steps $passos --start-steps 300 --learning-starts 300 `
        --eval-every-steps $passos --eval-episodes 2 --checkpoint-every-steps $passos `
        --log-dir "$saida/logs" --plot-dir "$saida/plots" --checkpoint-dir "$saida/checkpoints"

    if ($LASTEXITCODE -ne 0) { throw "O treino $nome falhou." }
}
```

O tempo varia com a máquina. Para encurtar a demonstração, defina `$passos = 2500` **antes de iniciar o grupo**. Assim as três rodadas têm o mesmo orçamento e a avaliação acontece no último passo de cada uma.

**Largada nas retas curtas.** O `SafeSpawnWrapper` corrige a posição inicial do carro do agente se o HighwayEnv o colocar fora da faixa. Com isso, as retas de 5 m e 30 m voltam a fazer parte do roteiro. Com tráfego, ainda pode haver colisão logo após a largada.

## 3. Escolha o agente para gravar

Logo abaixo de cada `Treinando oval…`, procure uma linha de avaliação como a do exemplo.

```text
[eval @ step    5000] return=35.34±7.54 crash_rate=0.00 mean_speed=6.14 (109s elapsed)
```

Compare o número **antes do `±`** em `return`, que é o retorno médio da avaliação. Escolha a maior média. Em caso de empate, use a menor `crash_rate`, a fração dos episódios em que houve colisão. Anote o comprimento da reta correspondente.

Os arquivos abaixo ficam dentro de `$saida`. Aqui, `oval80m` corresponde à reta de 80 m.

| Arquivo | Para que serve |
| --- | --- |
| `checkpoints/oval80m_final.pt` | Modelo treinado usado na gravação. |
| `logs/oval80m.csv` | Métricas por episódio **durante o treino**. |
| `logs/oval80m.meta.json` | Argumentos usados na rodada. |
| `plots/oval80m.png` | Curvas de treino. |

Com uma semente e dois episódios de avaliação, o resultado serve para escolher um vídeo da demonstração. Para comparar desempenho com mais confiança, repita cada configuração com várias sementes e mais episódios de avaliação.

## 4. Grave o MP4

Na primeira linha, ajuste `80` para o comprimento que você escolheu. A mesma variável seleciona a geometria da pista e o arquivo do modelo.

```powershell
$reta = 80
$nome = "oval$($reta)m"

& $python scripts/record.py --config $config --agent sac `
    --oval-straight-length-m $reta `
    --checkpoint "$saida/checkpoints/${nome}_final.pt" `
    --episodes 3 --out "$saida/best_run.mp4"

if ($LASTEXITCODE -ne 0) { throw "A gravação falhou." }
Invoke-Item "$saida/best_run.mp4"
```

O gravador gera os quadros em segundo plano, salva o MP4 e imprime `Saved video:` com o caminho. A última linha abre o arquivo no player padrão do Windows.

Para reproduzir a configuração da pista, use o mesmo `--config` e repita qualquer opção de pista, tráfego ou atraso que você tenha alterado no treino. Se aparecer `size mismatch` ao carregar o modelo, confira especialmente `enable_delay_wrapper`. Essa opção altera o tamanho da entrada do agente e precisa coincidir entre treino e gravação.

## Opções úteis para outros experimentos

As opções na linha de comando têm prioridade sobre o arquivo JSONC.

| Quero mudar | Opção |
| --- | --- |
| Variante da pista | Use `--env racetrack_oval` e um `--env-id` oval. Ao trocar para uma pista comum, escolha também um `env_id` compatível e deixe as opções do oval como `null` no JSONC. |
| Comprimento da reta ou obstáculo | `--oval-straight-length-m` ou `--oval-obstacle-preset` com `block_lane` ou `force_decision`. |
| Intensidade do tráfego | `--traffic` com `none`, `light`, `medium`, `heavy` ou `jam`. |
| Velocidade inicial do tráfego | `--traffic-speed-min` e `--traffic-speed-max` juntos, em m/s. |
| Atraso de ações | Com `enable_delay_wrapper` ligado, ajuste `--action-delay-min-steps` e `--action-delay-max-steps` entre 0 e 6. Use 0 nos dois para manter a mesma entrada do agente com atraso de ação zero. |
| Tempo de entrega visível para o agente | `--delay-known-to-agent` mostra o tempo restante dos comandos pendentes. Para ocultá-lo, use `"delay_known_to_agent": false` no JSONC. |
| Atraso na observação | `--observation-delay-steps` define quantos passos a observação leva para chegar ao agente. |
| Agente de referência | `--agent random` usa ações aleatórias. O agente treinável neste roteiro é o SAC. |

Para mudar o padrão das próximas rodadas, edite o [arquivo de configuração](configs/racetrack_experiment_config.jsonc). Para ver todas as opções, rode `& $python scripts/train.py --help` ou `& $python scripts/record.py --help`.
