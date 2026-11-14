# Guia de Resolução de Problemas (Troubleshooting)

Este documento reúne soluções para os problemas e erros mais comuns em ambientes de execução local no Windows.

---

## 1. Problemas de Aquisição de Dados

### 1.1 Falha de Conexão ou Download Interrompido
- **Sintoma**: Mensagem `DataDownloadError: Download failed with HTTP status...` ou timeout.
- **Causa**: Falha de rede, bloqueio corporativo ou indisponibilidade temporária do repositório remoto.
- **Solução**: Faça o download manual de `B2W-Reviews01.csv` e coloque o arquivo diretamente em `data/raw/B2W-Reviews01.csv`. Em seguida, execute `python scripts/prepare_data.py`.

### 1.2 Arquivo Baixado é uma Página HTML
- **Sintoma**: Mensagem `Downloaded content is HTML rather than CSV data`.
- **Causa**: Download salvou uma página web de erro (ex: 404 do GitHub) com extensão CSV.
- **Solução**: Execute o download forçado (`python scripts/download_data.py --force`) ou obtenha o arquivo bruto real.

---

## 2. Problemas com CUDA e Memória de GPU

### 2.1 CUDA Out of Memory (OOM)
- **Sintoma**: `torch.cuda.OutOfMemoryError: CUDA out of memory`.
- **Causa**: O tamanho do lote (`batch_size`) ou o comprimento de sequência (`max_length`) excede a VRAM disponível da GPU.
- **Solução**:
  1. No arquivo de configuração (`configs/default.yaml`), reduza `batch_size` de `8` para `4` ou `2`.
  2. Aumente `gradient_accumulation_steps` proporcionalmente (ex: `gradient_accumulation_steps: 4`) para manter o tamanho de lote efetivo.
  3. Reduza `max_length` de `128` para `64` se necessário.

### 2.2 CUDA Indisponível (Execução em CPU)
- **Sintoma**: `CUDA available: False`.
- **Causa**: A máquina não possui GPU NVIDIA ou o PyTorch instalado é a versão exclusiva para CPU.
- **Solução**: O pipeline continuará funcionando normalmente em CPU com fallback automático. Para utilizar GPU, instale o build do PyTorch compatível com sua versão de CUDA conforme descrito em `docs/setup_windows.md`.

---

## 3. Permissões e Execução de Scripts no Windows PowerShell

### 3.1 Script de Ativação do Ambiente Bloqueado
- **Sintoma**: `cannot be loaded because running scripts is disabled on this system`.
- **Solução**: Execute no PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```

---

## 4. Reexecução Segura do Pipeline

Todas as fases e comandos individuais são idempotentes e reexecutáveis com segurança. Caso uma execução seja interrompida por desligamento ou falta de energia, basta reexecutar o comando correspondente sem risco de corrupção do dataset bruto original.
