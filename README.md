# Classificação Binária de Sentimentos com BERTimbau, PyTorch e Pipeline Científico Reproduzível

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-yellow.svg)](https://huggingface.co/docs/transformers/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-brightgreen.svg)](https://github.com/astral-sh/ruff)

Projeto completo de Engenharia de Software e experimento científico para **Classificação Binária de Sentimentos em Avaliações de Produtos em Português** utilizando o modelo Transformer encoder-only **BERTimbau Base** (`neuralmind/bert-base-portuguese-cased`), PyTorch, baselines clássicos e pipeline científico com geração de evidências completas para publicação acadêmica.

---

## 1. Pergunta de Pesquisa e Hipótese

### Pergunta de Pesquisa
> **Em que medida o fine-tuning do BERTimbau melhora a classificação de avaliações negativas e positivas em comparação com um baseline majoritário e um baseline clássico baseado em TF-IDF com regressão logística?**

### Hipótese Científica
> **O BERTimbau ajustado deverá superar o baseline majoritário e apresentar ganho sobre TF-IDF em F1 macro, especialmente em avaliações com negação, dependência contextual e combinação entre título e corpo do texto.**

---

## 2. Mapa da Documentação

A documentação detalhada do projeto está organizada no diretório [`docs/`](docs/) e vinculada abaixo:

| Documento | Descrição |
| :--- | :--- |
| [`docs/architecture.md`](docs/architecture.md) | Arquitetura de software, fluxo de dados, organização de módulos e princípios de engenharia. |
| [`docs/dataset.md`](docs/dataset.md) | Especificação do B2W-Reviews01, download, composição textual, exclusão de nota 3 e deduplicação. |
| [`docs/setup_windows.md`](docs/setup_windows.md) | Guia completo de instalação local no Windows com PowerShell (CPU e GPU NVIDIA com CUDA). |
| [`docs/execution.md`](docs/execution.md) | Guia de execução nos três níveis: scripts individuais, fases via CLI e pipeline completo. |
| [`docs/experiments.md`](docs/experiments.md) | Desenho experimental, perfis de configuração (`smoke`, `default`, `full`) e protocolo de teste. |
| [`docs/evidence.md`](docs/evidence.md) | Estrutura e catálogo de todos os artefatos de evidência gerados em `artifacts/runs/<run_id>/`. |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Diretrizes de reprodutibilidade, controle de sementes, determinismo e hashes SHA-256. |
| [`docs/testing.md`](docs/testing.md) | Guia de testes unitários, testes de integração, testes de invariantes e cobertura de código. |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Resolução de problemas comuns (erros de download, OOM de GPU, caminhos no Windows). |
| [`docs/article_support.md`](docs/article_support.md) | Mapeamento direto das evidências geradas para as seções de um artigo científico. |

---

## 3. Instalação no Windows com PowerShell

Consulte [`docs/setup_windows.md`](docs/setup_windows.md) para obter instruções detalhadas.

```powershell
# 1. Criar e ativar ambiente virtual
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependências de execução e desenvolvimento
pip install -r requirements-dev.txt
pip install -e .
```

*Nota sobre aceleração por GPU*: Caso disponha de GPU NVIDIA, instale o build apropriado do PyTorch para sua versão do CUDA (detalhado em [`docs/setup_windows.md`](docs/setup_windows.md)). O projeto seleciona GPU automaticamente quando disponível e realiza fallback seguro para CPU caso contrário.

---

## 4. Três Níveis de Execução

Consulte [`docs/execution.md`](docs/execution.md) para todos os parâmetros suportados.

### Nível 1: Scripts Granulares Individuais (`scripts/`)
```powershell
python scripts/download_data.py --config configs/default.yaml
python scripts/audit_data.py --config configs/default.yaml
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train_baselines.py --config configs/default.yaml
python scripts/train_transformer.py --config configs/default.yaml
python scripts/evaluate_models.py --config configs/default.yaml
python scripts/analyze_errors.py --config configs/default.yaml
python scripts/build_evidence.py --config configs/default.yaml
```

### Nível 2: Execução por Fase via CLI do Pacote
```powershell
python -m sentiment_transformer.cli phase data --config configs/default.yaml
python -m sentiment_transformer.cli phase baselines --config configs/default.yaml
python -m sentiment_transformer.cli phase transformer --config configs/default.yaml
python -m sentiment_transformer.cli phase evaluation --config configs/default.yaml
python -m sentiment_transformer.cli phase evidence --config configs/default.yaml
```

### Nível 3: Pipeline Completo com Isolamento de Run
```powershell
# Smoke test rápido (200 amostras, 1 época) para validação de integridade
python -m sentiment_transformer.cli run --config configs/smoke.yaml

# Experimento científico principal padrão (10.000 amostras balanceadas, 3 épocas)
python -m sentiment_transformer.cli run --config configs/default.yaml
```

---

## 5. Estrutura do Repositório

```text
projeto-5-transformers/
├── README.md                  # Este documento (índice central)
├── pyproject.toml             # Configuração central de build, ferramentas e testes
├── requirements.txt           # Dependências de execução
├── requirements-dev.txt       # Dependências de desenvolvimento e testes
├── Makefile                   # Alvos comuns de automação
├── .editorconfig              # Padrões de formatação de arquivos
├── .pre-commit-config.yaml    # Hooks de pré-commit
├── configs/                   # Arquivos de configuração YAML
│   ├── smoke.yaml             # Configuração reduzida para testes rápidos
│   ├── default.yaml           # Configuração do experimento científico principal
│   └── full.yaml              # Configuração ampliada
├── data/                      # Diretórios de dados (isolados e ignorados pelo git)
│   ├── raw/                   # Dataset bruto baixado (B2W-Reviews01.csv)
│   ├── interim/               # Dados intermediários
│   └── processed/             # Partições estratificadas (train.csv, val.csv, test.csv)
├── docs/                      # Documentação técnica e científica completa (10 docs)
├── notebooks/                 # Notebook exploratório secundário
│   └── exploratory_analysis.ipynb
├── scripts/                   # Scripts executáveis (Nível 1 de execução)
├── src/                       # Código-fonte principal do pacote Python
│   └── sentiment_transformer/
│       ├── constants.py
│       ├── exceptions.py
│       ├── paths.py
│       ├── logging_config.py
│       ├── reproducibility.py
│       ├── config.py
│       ├── data/              # Download, auditoria, composição e divisão
│       ├── baselines/         # Baseline Majoritário e TF-IDF + Logistic Regression
│       ├── modeling/          # PyTorch Dataset, Device, Tokenizer e Trainer
│       ├── evaluation/        # Métricas, matrizes, curvas e análise de erros
│       ├── evidence/          # Hashes, metadados de ambiente e manifesto
│       └── pipeline/          # Orquestrador ponta a ponta e fases
├── tests/                     # Suíte de testes automatizados (unit, integration, invariants)
└── artifacts/                 # Saída imutável de execuções
    └── runs/                  # Diretórios isolados por timestamp e run_id
```

---

## 6. Qualidade, Testes e Cobertura

O projeto adota padrões rigorosos de Engenharia de Software configurados no `pyproject.toml`:

```powershell
# Execução da suíte completa de testes com relatório de cobertura
python -m pytest tests/

# Verificação de formatação e linting
black --check src tests scripts
ruff check src tests scripts

# Verificação estática de tipos
mypy src
```

---

## 7. Licença

Este projeto é disponibilizado sob a licença [MIT](LICENSE).
