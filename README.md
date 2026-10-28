# Pipeline meteorológico INMET + MongoDB Atlas

Pipeline Python para importar histórico CSV do INMET, atualizar observações recentes, treinar um modelo diário e publicar previsões para o frontend no MongoDB Atlas.

## Estrutura

- `scripts/importar_historico.py`: carga histórica inicial.
- `main.py`: atualização recorrente, adequado ao Render Cron Job.
- `src/transform.py`: normalização de colunas, datas e agregação diária.
- `src/mongo_repository.py`: conexão, índices únicos e upsert em lote.
- `src/ml_forecaster.py`: treinamento e previsão Random Forest com validação MAE.
- `src/forecast_client.py`: leitura da previsão oficial de 10 dias do INMET.
- `render.yaml`: agendamento diário às 06:00 UTC.

## 1. Preparar o ambiente local

Python 3.11 ou 3.12 recomendado.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite `.env` e informe `MONGODB_URI`, banco, coleções e o código correto da estação automática. Nunca coloque a URI diretamente em um script.

Coloque o CSV em `data/historico_inmet.csv`, ou altere `HISTORICAL_CSV`.

## 2. Confirmar o CSV

Os nomes das colunas variam entre arquivos do INMET. O normalizador reconhece alguns nomes comuns, mas não todos os formatos. Antes de importar, se houver erro de coluna, confira o cabeçalho do CSV e acrescente os nomes correspondentes em `ALIASES` em `src/transform.py`.

A data/hora da estação automática do INMET costuma ser UTC. Confirme essa convenção no arquivo baixado antes de usar os dados em análises locais. O código armazena timestamps horários como UTC e agrega os dias pelo calendário de `America/Sao_Paulo`.

## 3. Importar o histórico dos últimos 10 anos

```powershell
python -m scripts.importar_historico
```

Para informar outro arquivo:

```powershell
python -m scripts.importar_historico --csv "C:\caminho\historico.csv"
```

Para importar todas as datas contidas no CSV:

```powershell
python -m scripts.importar_historico --all
```

A importação é idempotente para a chave estação + data/hora: executar novamente atualiza o documento correspondente, em vez de criar duplicata.

## 4. Testar a atualização recente

```powershell
python main.py
```

Se a API retornar erro, confirme o código da estação, a conectividade e o formato da resposta. O endpoint usado é `https://apitempo.inmet.gov.br/estacao/dados/{data_inicio}/{data_fim}/{codigo_estacao}`. A disponibilidade e o esquema da API podem mudar; valide uma resposta antes de agendar produção.

## 5. Previsões para o frontend

Cada execução de `main.py` grava um documento por estação na coleção definida por `MONGODB_PREDICTIONS_COLLECTION` (por padrão, `previsoes`). O documento contém:

- `modelo_ml.previsoes`: previsão diária de temperatura, umidade e precipitação para os próximos dias.
- `modelo_ml.previsoes[].chuva_provavel`: indicação diária derivada de `precipitacao_total_mm >= 0.1`.
- `chuva_proximas_horas.previsoes`: temperatura, umidade, probabilidade de chuva e indicação booleana para cada uma das próximas 24 horas.
- `modelo_ml.validacao.mae_por_variavel`: erro absoluto médio calculado em uma janela histórica de validação.
- `previsao_oficial.previsao_10dias`: resposta de previsão oficial do INMET, quando disponível.
- `gerado_em` e `status_previsao_oficial`: metadados para o frontend saber quando atualizar a tela.
- `status_observacoes` e `observacoes_recebidas`: indicam se a execução recebeu dados novos ou usou o histórico salvo.

O frontend pode buscar `{ estacao_codigo: "A707" }` e ordenar ou exibir `modelo_ml.previsoes` por `data_local`. Se o INMET estiver indisponível, o documento continua sendo atualizado com o ML e o status oficial fica `indisponivel`.

MAE significa *Mean Absolute Error* (erro absoluto médio). Ele é a média de `abs(valor_real - valor_previsto)` em dados que o modelo não usou para treinar. Um MAE de `1.479` em temperatura significa erro médio de aproximadamente `1.479 °C`; quanto menor, melhor. Ele não é uma garantia para o futuro, apenas uma medida de desempenho histórico.

O primeiro modelo implementado é Random Forest, escolhido como baseline operacional. SARIMAX e XGBoost devem ser adicionados depois de uma comparação temporal justa usando as métricas de validação, sem misturar dados futuros no treinamento.

## 6. Configurar o MongoDB Atlas

- Copie a connection string em Atlas > Connect > Drivers.
- Crie um usuário de banco com permissões de leitura/escrita apenas no banco do projeto.
- Em Network Access, autorize a origem necessária para o ambiente de execução. Evite liberar acesso irrestrito quando puder usar uma lista de IPs adequada.
- Não envie `.env` ao GitHub. Guarde `MONGODB_URI` como secret no Render.
- O script cria índices únicos nas coleções configuradas. Se a coleção já tiver duplicatas na combinação estação + data/hora, a criação do índice falhará; remova ou consolide as duplicatas antes.

## 7. Fluxo de produção limpo: treino fora da rotina diária

A rotina de produção não deve treinar modelos a cada execução. O padrão correto é:

1. Coletar e armazenar novos dados no MongoDB.
2. Publicar a previsão diária usando o modelo já treinado.
3. Treinar/validar modelos novos em uma rotina separada, manual ou agendada para manutenção.
4. Salvar o melhor modelo em um artefato persistido.
5. Atualizar o artefato do ambiente em produção somente quando a nova versão for validada.

Esse padrão evita timeouts no Render, reduz custo computacional e mantém a coleta de dados operacional e previsível.

## 8. Comparar Random Forest, XGBoost e SARIMAX em produção

Para testar a melhor opção, execute a comparação fora do cron:

```powershell
python -m scripts.comparar_modelos
```

O script compara:

- Random Forest
- XGBoost
- SARIMAX/SARIMA

usando os mesmos dados históricos e uma janela de validação temporal. Ele calcula MAE por variável e imprime o ranking dos modelos. O vencedor deve ser salvo como artefato final e usado pelo refresh diário.

O fluxo recomendado é:

```text
historico MongoDB
    -> comparar modelos (offline)
    -> escolher melhor modelo
    -> salvar artifact (.joblib)
    -> refresh diário usa artifact
    -> publica previsao no MongoDB
```

A comparação deve usar apenas dados anteriores à janela de validação. Não misture futuro com treino; isso distorce a métrica de precisão e cria falsa confiança.

## Limitações e validações necessárias

- A API pública pode limitar períodos ou mudar o esquema. Teste com a estação real antes da implantação.
- O CSV precisa ter dados horários para produzir resumos diários coerentes.
- Precipitação diária é calculada pela soma dos valores horários válidos. Verifique a definição da variável no arquivo original para evitar somar valores que representem acumulados em janelas sobrepostas.
- Dias com observações faltantes podem ter totais subestimados. `quantidade_registros_horarios` ajuda a identificar dias incompletos.
- Este pipeline organiza observações meteorológicas, não gera previsões de machine learning ainda.
