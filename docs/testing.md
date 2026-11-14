# Guia de Testes e Garantia de Qualidade

Este documento detalha a suíte de testes unitários, testes de integração, testes de invariantes e os critérios de cobertura de código.

---

## 1. Estrutura da Suíte de Testes

```text
tests/
├── conftest.py             # Fixtures compartilhadas (temp_dir, configs, DataFrames de teste)
├── fixtures/
│   └── sample_reviews.csv  # Base mínima com casos de teste representativos
├── unit/                   # Testes unitários de funções e classes individuais
│   ├── test_config.py
│   ├── test_paths.py
│   ├── test_reproducibility.py
│   ├── test_data_validation.py
│   ├── test_data_preprocessing.py
│   ├── test_data_deduplication.py
│   ├── test_data_splitting.py
│   ├── test_data_audit.py
│   ├── test_baselines.py
│   ├── test_device.py
│   ├── test_modeling.py
│   ├── test_metrics.py
│   ├── test_error_analysis.py
│   ├── test_evidence.py
│   ├── test_download.py
│   ├── test_plots.py
│   ├── test_logging.py
│   ├── test_trainer.py
│   └── test_orchestrator.py
└── integration/            # Testes de integração de pipelines e fases
    ├── test_data_phase.py
    ├── test_baselines_phase.py
    ├── test_pipeline.py
    ├── test_cli.py
    └── test_invariants.py
```

---

## 2. Invariantes Científicos Testados

A suíte de testes verifica automaticamente os seguintes invariantes obrigatórios:
1. **Exclusão Efetiva da Nota 3**: Nenhuma avaliação com nota 3 está presente nas partições de treino, validação ou teste;
2. **Restrição de Rótulos**: Todos os rótulos de destino estão estritamente contidos no conjunto $\{0, 1\}$;
3. **Presença Bipopulacional**: Ambas as classes estão representadas em todas as partições geradas;
4. **Isolamento de Identificadores (No Leakage)**: Interseção vazia de `review_id` entre as três partições;
5. **Isolamento de Texto Normalizado (No Text Leakage)**: Interseção vazia de textos normalizados entre as três partições;
6. **Proporções de Amostragem**: Respeito às proporções estratificadas configuradas (70% treino, 15% validação, 15% teste);
7. **Isolamento do Teste**: O conjunto de teste é mantido intocado durante o treinamento e seleção de checkpoints.

---

## 3. Execução dos Testes e Cobertura

### 3.1 Executando Todos os Testes
```powershell
python -m pytest tests/
```

### 3.2 Executando com Relatório de Cobertura
```powershell
python -m pytest --cov=sentiment_transformer --cov-report=term-missing tests/
```

### 3.3 Verificação Estática de Formatação e Tipagem
```powershell
# Formatação com Black
black --check src tests scripts

# Linting com Ruff
ruff check src tests scripts

# Tipagem estática com Mypy
mypy src
```
