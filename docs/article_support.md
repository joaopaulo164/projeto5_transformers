# Mapeamento de Evidências para Seções do Artigo Científico

Este documento detalha como os artefatos gerados pelo projeto `sentiment_transformer` devem ser aproveitados na escrita de um artigo científico acadêmico ou relatório técnico.

---

## 1. Mapeamento Direto por Seção

### 1.1 Materiais e Métodos (Metodologia)
- **Descrição do Corpus e Aquisição**: Consultar `docs/dataset.md` e os arquivos `dataset_audit.json` e `dataset_statistics.json` gerados na pasta de run.
- **Justificativa da Exclusão da Nota 3**: Citar a justificativa metodológica padronizada presente em `dataset_statistics.json` e `docs/dataset.md`.
- **Estratégia de Composição Textual**: Detalhar a concatenação $\text{título} + \text{ [SEP] } + \text{corpo}$ descrita em `docs/dataset.md`.
- **Deduplicação e Particionamento**: Referenciar as estatísticas de `deduplication_stats` e `split_statistics.json` (divisão estratificada 70/15/15 sem sobreposição).
- **Ambiente Computacional e Reprodutibilidade**: Extrair dados de CPU, GPU, versão do PyTorch e commits de `environment.json` e `data_fingerprints.json`.

### 1.2 Modelos e Protocolo Experimental
- **Modelos de Linha de Base**: Descrever o Baseline Majoritário e o pipeline TF-IDF com Regressão Logística conforme `docs/experiments.md`.
- **Arquitetura BERTimbau Base**: Detalhar o checkpoint `neuralmind/bert-base-portuguese-cased`, hiperparâmetros (taxa de aprendizado, épocas, otimizador AdamW com decaimento linear e warmup) obtidos de `config_resolved.yaml`.
- **Critério de Seleção**: Ressaltar que a escolha de checkpoint baseou-se estritamente no F1 macro de validação, mantendo o teste isolado até a avaliação final.

### 1.3 Resultados Quantitativos
- **Tabela Comparativa Consolidada**: Inserir diretamente a tabela Markdown gerada em `model_comparison.md` (ou importar os valores de `model_comparison.csv`).
- **Curvas de Aprendizado**: Incluir a figura `training_curves.png` ilustrando a convergência de perda e evolução do F1 macro ao longo das épocas.
- **Matrizes de Confusão**: Apresentar os gráficos `confusion_matrix_majority.png`, `confusion_matrix_tfidf.png` e `confusion_matrix_transformer.png`.

### 1.4 Análise Qualitativa e Discussão de Erros
- **Análise Sistemática de Erros**: Utilizar os dados de `error_analysis.csv` para discutir:
  1. *Falsos Positivos e Falsos Negativos de Alta Confiança*;
  2. *Impacto de Negações e Inversões de Polaridade*;
  3. *Conflitos entre Título e Corpo*;
  4. *Avaliações com Polaridade Dividida entre Produto e Entrega*.

### 1.5 Ameaças à Validade e Limitações
- Rótulos fracos derivados de estrelas;
- Limites de determinismo em GPU;
- Truncamento em 128 tokens para avaliações muito extensas (documentado em `dataset_statistics.json`).
