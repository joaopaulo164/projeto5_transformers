# Desenho Experimental e Protocolo Científico

Este documento detalha o desenho experimental, os perfis de configuração, os modelos comparados e o protocolo de isolamento do conjunto de teste.

---

## 1. Pergunta de Pesquisa e Hipótese

### Pergunta de Pesquisa
> "Em que medida o fine-tuning do BERTimbau melhora a classificação de avaliações negativas e positivas em comparação com um baseline majoritário e um baseline clássico baseado em TF-IDF com regressão logística?"

### Hipótese
> "O BERTimbau ajustado deverá superar o baseline majoritário e apresentar ganho sobre TF-IDF em F1 macro, especialmente em avaliações com negação, dependência contextual e combinação entre título e corpo do texto."

---

## 2. Modelos Comparados

1. **Baseline Majoritário**:
   - `DummyClassifier(strategy="most_frequent")`
   - Estabelece o limiar trivial inferior de desempenho.
2. **Baseline Clássico (TF-IDF + Regressão Logística)**:
   - `TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)`
   - `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`
   - Estabelece o padrão clássico baseado em frequência de n-gramas e linearidade.
3. **Transformer Principal (BERTimbau Base Fine-Tuned)**:
   - Checkpoint: `neuralmind/bert-base-portuguese-cased`
   - Arquitetura *encoder-only* pré-treinada em corpus amplo em língua portuguesa.
   - Fine-tuning com cabeça de classificação binária (`AutoModelForSequenceClassification`).

---

## 3. Perfis de Configuração

| Perâmetro | `configs/smoke.yaml` | `configs/default.yaml` | `configs/full.yaml` |
| :--- | :---: | :---: | :---: |
| **Objetivo** | Validação de integridade do pipeline | Experimento científico principal | Execução ampliada |
| **Amostras Totais** | 200 | 10.000 | 20.000 (ou todo o conjunto) |
| **Balanceamento** | 100 Neg / 100 Pos | 5.000 Neg / 5.000 Pos | 10.000 Neg / 10.000 Pos |
| **Comprimento Máx (`max_length`)** | 64 tokens | 128 tokens | 128 tokens |
| **Épocas** | 1 | 3 | 3 |
| **Tamanho do Lote (`batch_size`)** | 4 | 8 | 16 |
| **Taxa de Aprendizado (`lr`)** | $3 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ |
| **Early Stopping (Paciência)** | 1 época | 2 épocas | 2 épocas |

---

## 4. Protocolo Experimental Estrito

1. **Divisão Estratificada 70/15/15**: 70% treino, 15% validação e 15% teste.
2. **Isolamento Absoluto do Conjunto de Teste**:
   - O conjunto de teste **nunca** é utilizado durante o treinamento, escolha de hiperparâmetros, cálculo de vocabulário TF-IDF ou critério de *early stopping*.
   - A seleção do melhor checkpoint do modelo Transformer baseia-se exclusivamente no **F1 macro obtido no conjunto de validação**.
   - O conjunto de teste é avaliado apenas uma única vez, na fase final de avaliação (`run_evaluation_phase`).
3. **Métrica Primária**: F1 Macro ($F_1 = \frac{F_{1,\text{neg}} + F_{1,\text{pos}}}{2}$).
