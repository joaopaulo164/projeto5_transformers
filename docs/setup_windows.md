# Guia de Configuração e Instalação no Windows com PowerShell

Este guia orienta o provisionamento do ambiente local de execução no sistema operacional Windows utilizando PowerShell.

---

## 1. Pré-requisitos do Sistema

- **Sistema Operacional**: Windows 10 ou Windows 11 (64-bit);
- **Python**: Versão 3.10 ou superior instalada e no PATH;
- **Terminal**: Windows PowerShell ou PowerShell 7+;
- **Git** (opcional, para controle de versão);
- **GPU NVIDIA com CUDA** (opcional; caso ausente, o pipeline executa em CPU automaticamente).

---

## 2. Criação e Ativação do Ambiente Virtual

Abra o PowerShell na raiz do projeto:

```powershell
# Verificar versão do Python
python --version

# Criar ambiente virtual isolado (.venv)
python -m venv .venv

# Configurar política de execução do PowerShell se necessário
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Ativar o ambiente virtual
.\.venv\Scripts\Activate.ps1
```

---

## 3. Instalação das Dependências

### 3.1 Instalação Básica (Runtime)
```powershell
pip install -r requirements.txt
pip install -e .
```

> **Importante**: O comando `pip install -e .` instala o pacote em modo editável. Caso execute `python -m sentiment_transformer.cli` sem essa etapa, o Python gerará `ModuleNotFoundError: No module named 'sentiment_transformer'` porque o código-fonte reside em `src/sentiment_transformer/` e não está no `sys.path` do `.venv`. Alternativamente, pode-se definir `$env:PYTHONPATH = "src"`.

### 3.2 Instalação para Desenvolvimento e Testes
```powershell
pip install -r requirements-dev.txt
pip install -e .
```

---

## 4. Configuração de Suporte a GPU (CUDA) ou CPU

### 4.1 Fallback Automático para CPU
Por padrão, se não houver placa de vídeo NVIDIA compatível ou se os drivers CUDA não estiverem instalados, o PyTorch usará CPU automaticamente para todos os treinamentos e avaliações. O *smoke test* é projetado para rodar em CPU em poucos segundos.

### 4.2 Habilitando Aceleração por GPU (CUDA)
Caso possua GPU NVIDIA compatível, consulte a matriz de compatibilidade do PyTorch oficial e instale o build correspondente à versão do driver:

```powershell
# Exemplo para PyTorch com CUDA 12.1 (ajuste conforme seu driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 Verificação de Hardware e CUDA
Execute o comando abaixo no PowerShell para inspecionar os recursos detectados:

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Disponível: {torch.cuda.is_available()} | Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 5. Verificação da Instalação do Pacote

Verifique se a CLI está acessível e respondendo corretamente:

```powershell
python -m sentiment_transformer.cli --help
```
