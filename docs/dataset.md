# Especificação e Metodologia do Dataset B2W-Reviews01

Este documento detalha as características do dataset **B2W-Reviews01**, os procedimentos de aquisição, as regras de composição textual, a justificativa da exclusão de notas intermediárias e a prevenção de vazamento de dados (*data leakage*).

---

## 1. Origem e Características Gerais

- **Nome**: B2W-Reviews01 (Corpus de Avaliações de Produtos de Comércio Eletrônico no Brasil)
- **Fonte Oficial**: Repositório oficial Americanas Tech / B2W no GitHub (`https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/master/B2W-Reviews01.csv` - a organização anterior `b2wads` foi migrada para `americanas-tech`).
- **Idioma**: Português do Brasil (pt-BR)
- **Escala de Avaliação**: 1 a 5 estrelas (`overall_rating`)

---

## 2. Aquisição: Download Automático e Fallback Manual

### 2.1 Download Automático
O download é executado por streaming com timeout configurável e escrita prévia em arquivo temporário antes da substituição atômica no destino final `data/raw/B2W-Reviews01.csv`.
São validados:
- Código de resposta HTTP (200 OK);
- Tamanho mínimo configurável (ex: 10 MB para arquivo oficial);
- Ausência de conteúdo HTML (típico de páginas de erro ou redirecionamento);
- Cálculo do hash criptográfico SHA-256.

### 2.2 Fallback Manual
Caso o ambiente não disponha de acesso à internet:
1. Baixe o arquivo `B2W-Reviews01.csv` diretamente da fonte oficial ou repositório arquivado.
2. Salve o arquivo no caminho exato:
   ```powershell
   data/raw/B2W-Reviews01.csv
   ```
3. O pipeline validará automaticamente a estrutura, colunas e hash do arquivo local antes de iniciar o processamento.

---

## 3. Definição da Tarefa e Rotulagem Binária

A tarefa do experimento é a **classificação binária de sentimentos**:
- **Notas 1 e 2**: Classe **Negativa**, rótulo inteiro `0`.
- **Nota 3**: **Excluída** do experimento principal.
- **Notas 4 e 5**: Classe **Positiva**, rótulo inteiro `1`.

### 3.1 Justificativa Metodológica da Exclusão da Nota 3

> "A polaridade foi definida a partir das avaliações numéricas, considerando notas 1 e 2 como negativas e notas 4 e 5 como positivas. Avaliações com nota 3 foram excluídas porque a posição intermediária na escala não garante neutralidade textual. Esses registros podem expressar opiniões mistas, polaridade moderada ou avaliações divergentes entre produto, entrega e experiência de compra. Como as estrelas funcionam como rótulos fracos do conteúdo textual, a exclusão da classe intermediária reduz ruído de rotulagem e permite avaliar, neste primeiro experimento, casos de polaridade mais claramente definida."

A exclusão é contabilizada nos relatórios de auditoria e manifestos (`dataset_audit.json` e `dataset_statistics.json`).

---

## 4. Composição da Entrada Textual

Para capturar tanto a síntese expressa no título quanto o detalhamento do corpo da avaliação, a entrada do modelo é composta seguindo regras estritas:

1. **Preservação de Conteúdo**: O título e o corpo originais são preservados (com normalização Unicode NFKC e colapso de espaços repetidos).
2. **Composição Completa**: Quando título e corpo existirem:
   $$\text{texto\_combinado} = \text{título} + \text{ [sep\_token] } + \text{corpo}$$
   O token separador real é consultado dinamicamente no tokenizer (`tokenizer.sep_token`).
3. **Ausência de Título**: Quando o título estiver vazio, utiliza-se apenas o corpo.
4. **Ausência de Corpo**: Registros sem corpo textual ou contendo apenas espaços são descartados e contabilizados.

---

## 5. Deduplicação e Prevenção de Vazamento (Data Leakage)

1. **Chave Normalizada de Duplicata**: Gera-se uma representação canônica em minúsculas, com espaços colapsados e Unicode normalizado para agrupar textos semanticamente idênticos.
2. **Resolução de Conflitos de Rótulo**: Se dois registros tiverem o mesmo texto normalizado mas notas conflitantes (uma negativa e uma positiva), o grupo inteiro é removido do experimento, eliminando ambiguidades no padrão-ouro.
3. **Divisão Prévia à Amostragem**: A deduplicação ocorre rigorosamente antes da divisão em treino, validação e teste.
4. **Testes de Invariantes**: O pipeline valida automaticamente que não há sobreposição de `review_id` nem de texto normalizado entre as partições de treino (70%), validação (15%) e teste (15%).
5. **Identificadores Únicos Determinísticos**: Como o dataset bruto B2W-Reviews01 não possui uma coluna de chave primária exclusiva por avaliação (as colunas `submission_date` e `reviewer_id` podem se repetir entre avaliações distintas), o carregador gera identificadores determinísticos únicos e estáveis (`review_0000000`, `review_0000001`, ...), garantindo rastreabilidade sem colisões de IDs.
