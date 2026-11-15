# Guia de Execução em Três Níveis

O projeto disponibiliza três interfaces de execução complementares, adequadas tanto para depuração granular quanto para execuções completas e reproduzíveis.

---

## 1. Nível 1: Scripts Granulares Individuais (`scripts/`)

Cada script em `scripts/` é um wrapper leve que delega a lógica ao pacote central:

```powershell
# 1. Download do dataset
python scripts/download_data.py --config configs/default.yaml

# 2. Auditoria do arquivo bruto
python scripts/audit_data.py --config configs/default.yaml

# 3. Preparação dos dados (limpeza, deduplicação, divisão 70/15/15)
python scripts/prepare_data.py --config configs/default.yaml

# 4. Treinamento dos baselines (Majoritário e TF-IDF)
python scripts/train_baselines.py --config configs/default.yaml

# 5. Fine-tuning do Transformer BERTimbau
python scripts/train_transformer.py --config configs/default.yaml

# 6. Avaliação completa e geração de predições no conjunto de teste
python scripts/evaluate_models.py --config configs/default.yaml

# 7. Análise de erros e heurística de negações
python scripts/analyze_errors.py --config configs/default.yaml

# 8. Consolidação do pacote de evidências e manifesto
python scripts/build_evidence.py --config configs/default.yaml
```

---

## 2. Nível 2: Execução por Fase via CLI (`sentiment_transformer.cli`)

Permite executar uma fase completa do pipeline mantendo a modularidade:

```powershell
# Fase 1: Data (Download, Auditoria, Limpeza, Deduplicação e Divisão)
python -m sentiment_transformer.cli phase data --config configs/default.yaml

# Fase 2: Baselines (Majoritário e TF-IDF + Regressão Logística)
python -m sentiment_transformer.cli phase baselines --config configs/default.yaml

# Fase 3: Transformer (Tokenização, DataLoader e Fine-Tuning)
python -m sentiment_transformer.cli phase transformer --config configs/default.yaml

# Fase 4: Evaluation (Métricas de Teste, Matrizes de Confusão e Curvas)
python -m sentiment_transformer.cli phase evaluation --config configs/default.yaml

# Fase 5: Evidence (Análise de Erros, Relatório Comparativo e Manifesto)
python -m sentiment_transformer.cli phase evidence --config configs/default.yaml
```

---

## 3. Nível 3: Pipeline Completo e Isolado

Executa todas as 5 fases em sequência, criando um diretório de *run* imutável com data, hora e configuração em `artifacts/runs/<run_id>/`:

### 3.1 Smoke Test (Validação Rápida em Poucos Minutos)
```powershell
python -m sentiment_transformer.cli run --config configs/smoke.yaml
```

### 3.2 Experimento Principal Padrão (10.000 avaliações balanceadas)
```powershell
python -m sentiment_transformer.cli run --config configs/default.yaml
```

### 3.3 Experimento Ampliado
```powershell
python -m sentiment_transformer.cli run --config configs/full.yaml
```

### 3.4 Wrapper de Execução Direta
```powershell
python scripts/run_experiment.py --config configs/default.yaml
```

---

## 4. Monitoramento e Logs em Tempo Real

Durante a execução do treinamento, validação e inferência do Transformer, o pipeline emite mensagens periódicas com métricas dinâmicas para acompanhamento contínuo (inclusive em execuções longas em CPU):

- **Treinamento**: Progresso percentual por época, passos concluídos, perda média acumulada (*running loss*), taxa de aprendizado atual (*learning rate*), velocidade (*samples/s*), tempo decorrido e estimativa de tempo restante (*ETA*).
- **Validação**: Progresso em lote com percentual a cada intervalo.
- **Inferência**: Acompanhamento por lote, contagem de amostras processadas, taxa de throughput e tempo decorrido.
