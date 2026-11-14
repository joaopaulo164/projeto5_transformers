# Catálogo e Estrutura do Pacote de Evidências

Cada execução do pipeline completo (`sentiment-transformer run`) gera um diretório isolado e imutável em:

```text
artifacts/runs/<run_id>/
```

Onde `<run_id>` possui o formato `YYYYMMDD_HHMMSS_<nome_configuracao>`, evitando qualquer sobreposição ou colisão.

---

## 1. Descrição dos Artefatos Gerados

| Arquivo | Formato | Descrição Científica |
| :--- | :---: | :--- |
| `config_resolved.yaml` | YAML | Configuração experimental integral e resolvida utilizada na execução. |
| `execution.log` | Texto | Registro cronológico detalhado de todos os eventos e logs da execução. |
| `environment.json` | JSON | Especificação de hardware (CPU, GPU, RAM), SO, Python e versões de pacotes. |
| `data_fingerprints.json` | JSON | Hashes criptográficos SHA-256 do arquivo original e de cada partição gerada. |
| `dataset_audit.json` | JSON | Estatísticas detalhadas de auditoria da base de dados bruta (distribuição, nulos, etc.). |
| `dataset_statistics.json` | JSON | Estatísticas de pré-processamento, contagem de notas 3 excluídas e comprimentos. |
| `split_statistics.json` | JSON | Proporções e distribuição de classes nas partições de treino, validação e teste. |
| `baseline_majority_metrics.json` | JSON | Métricas quantitativas completas do Baseline Majoritário. |
| `baseline_tfidf_metrics.json` | JSON | Métricas quantitativas completas do Baseline TF-IDF + Regressão Logística. |
| `transformer_metrics.json` | JSON | Métricas quantitativas completas do Transformer BERTimbau Fine-Tuned. |
| `classification_report_majority.csv` | CSV | Relatório detalhado por classe (precisão, recall, F1, suporte) do Majoritário. |
| `classification_report_tfidf.csv` | CSV | Relatório detalhado por classe do TF-IDF. |
| `classification_report_transformer.csv` | CSV | Relatório detalhado por classe do BERTimbau. |
| `confusion_matrix_majority.csv` | CSV | Matriz de confusão numérica do Baseline Majoritário. |
| `confusion_matrix_majority.png` | PNG | Imagem renderizada em alta resolução (300 DPI) da matriz de confusão do Majoritário. |
| `confusion_matrix_tfidf.csv` | CSV | Matriz de confusão numérica do TF-IDF. |
| `confusion_matrix_tfidf.png` | PNG | Imagem renderizada da matriz de confusão do TF-IDF. |
| `confusion_matrix_transformer.csv` | CSV | Matriz de confusão numérica do BERTimbau. |
| `confusion_matrix_transformer.png` | PNG | Imagem renderizada da matriz de confusão do BERTimbau. |
| `training_history.csv` | CSV | Histórico época a época de perda de treino, perda de validação e métricas. |
| `training_curves.png` | PNG | Gráficos da evolução da curva de perda e evolução do F1 macro ao longo das épocas. |
| `predictions_test_majority.csv` | CSV | Predições individuais do teste pelo Baseline Majoritário com probabilidades. |
| `predictions_test_tfidf.csv` | CSV | Predições individuais do teste pelo TF-IDF com probabilidades. |
| `predictions_test_transformer.csv` | CSV | Predições individuais do teste pelo BERTimbau com probabilidades e textos. |
| `error_analysis.csv` | CSV | Casos de erro e acerto sistematicamente selecionados com marcação de negação. |
| `model_comparison.csv` | CSV | Tabela comparativa consolidada de todas as métricas dos três modelos. |
| `model_comparison.md` | MD | Tabela comparativa formatada em Markdown pronta para inclusão em relatórios e artigos. |
| `run_manifest.json` | JSON | Manifesto final de proveniência com metadados, durações de fase e integridade. |
| `best_checkpoint/` | Diretório | Pesos do melhor modelo BERTimbau segundo F1 macro de validação e config do tokenizer. |

---

## 2. Utilização dos Artefatos para Artigos Científicos

- **Seção de Materiais e Métodos**: Utilize `environment.json`, `data_fingerprints.json` e `config_resolved.yaml`.
- **Seção de Resultados Quantitativos**: Insira a tabela de `model_comparison.md`, os gráficos de `training_curves.png` e as matrizes `confusion_matrix_*.png`.
- **Seção de Análise de Erros e Discussão**: Extraia exemplos qualitativos diretamente de `error_analysis.csv`.
