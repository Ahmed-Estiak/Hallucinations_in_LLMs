param(
    [string]$PythonPath = "$env:LOCALAPPDATA\Programs\Python\Python312-clean\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python 3.12 not found at $PythonPath. Install Python 3.12 first."
}

& $PythonPath -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install `
    python-dotenv `
    openai `
    google-genai `
    pandas `
    sentence-transformers `
    transformers `
    datasets `
    accelerate `
    peft `
    sentencepiece `
    protobuf
& .\.venv\Scripts\python.exe -m pip install --no-deps FlagEmbedding
& .\.venv\Scripts\python.exe -m pip install --force-reinstall fsspec==2026.2.0

& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
