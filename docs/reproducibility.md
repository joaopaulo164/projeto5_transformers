# Diretrizes de Reprodutibilidade e Determinismo

A reprodutibilidade científica é assegurada através de mecanismos estruturados em nível de software, dados e algoritmos.

---

## 1. Controle de Sementes Aleatórias (Seeds)

O módulo `sentiment_transformer.reproducibility.set_seed` fixa de maneira centralizada:
- Semente nativa do interpretador Python (`random.seed`);
- Semente do gerador numérico NumPy (`np.random.seed`);
- Semente do gerador de tensores PyTorch (`torch.manual_seed` e `torch.cuda.manual_seed_all`);
- Variável de ambiente `PYTHONHASHSEED` para fixar a ordem de hashing de strings e dicionários;
- Flags `torch.backends.cudnn.deterministic = True` e `torch.backends.cudnn.benchmark = False`.

---

## 2. Hashes Criptográficos e Linhagem de Dados

- O arquivo bruto e cada arquivo particionado (`train.csv`, `val.csv`, `test.csv`, `all_processed.csv`) recebem um hash SHA-256 no momento de sua criação.
- O manifesto `run_manifest.json` registra todos os hashes calculados, permitindo auditar se os dados foram alterados entre execuções.

---

## 3. Limitações de Determinismo em GPU (CUDA)

Embora todas as sementes e flags de determinismo sejam ativadas, pequenas variações de ponto flutuante podem ocorrer em GPUs devido a:
- Operações atômicas assíncronas em CUDA (`atomicAdd`);
- Algoritmos de convolução/atenção otimizados em diferentes versões de arquitetura ou drivers NVIDIA;
- Operações de paralelização não determinísticas de baixo nível.

Para máxima reprodutibilidade, registre sempre os metadados gerados em `environment.json` (incluindo modelo da GPU e versão do driver).

---

## 4. Registro de Controle de Versão (Git)

O manifesto captura automaticamente:
- Hash do commit atual (`commit_hash`);
- Nome da branch ativa (`branch`);
- Indicador se há modificações locais não commitadas (`is_dirty`).
