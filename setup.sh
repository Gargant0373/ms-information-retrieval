#!/usr/bin/env zsh
set -euo pipefail

VENV_DIR=".venv"
MODEL="llama3.2:3b"
OLLAMA_PORT=11434
OLLAMA_URL="http://localhost:${OLLAMA_PORT}"

OS="$(uname)"

echo "======================================"
echo "IR Project Environment Setup"
echo "Detected OS: $OS"
echo "======================================"

# ------------------------------------------------
# Check Python
# ------------------------------------------------
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found." >&2
  exit 1
fi

# ------------------------------------------------
# Check Java (PyTerrier)
# ------------------------------------------------
if ! command -v java &>/dev/null; then
  echo "Warning: Java not found. PyTerrier requires Java."
fi

# ------------------------------------------------
# Virtual Environment
# ------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "Python environment ready."

# ------------------------------------------------
# Install Ollama (Mac)
# ------------------------------------------------
install_ollama_mac() {
  if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama via Homebrew..."
    
    if ! command -v brew &>/dev/null; then
      echo "Installing Homebrew..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install ollama
  fi
}

# ------------------------------------------------
# Start Ollama Server
# ------------------------------------------------
start_ollama_native() {
  if ! pgrep -f "ollama serve" >/dev/null; then
    echo "Starting Ollama server..."
    ollama serve >/dev/null 2>&1 &
  fi
}

# ------------------------------------------------
# Docker Setup (Linux fallback)
# ------------------------------------------------
start_ollama_docker() {

  if ! command -v docker &>/dev/null; then
    echo "Error: Docker required but not installed." >&2
    exit 1
  fi

  if ! docker info &>/dev/null; then
    echo "Docker daemon not running." >&2
    exit 1
  fi

  if ! docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "Starting Ollama Docker container..."

    docker run -d \
      -v ollama:/root/.ollama \
      -p 11434:11434 \
      --name ollama \
      ollama/ollama
  fi
}

# ------------------------------------------------
# OS-specific logic
# ------------------------------------------------
if [[ "$OS" == "Darwin" ]]; then
  echo "Using native Ollama (Metal acceleration)"
  install_ollama_mac
  start_ollama_native
else
  echo "Using Docker-based Ollama"
  start_ollama_docker
fi

# ------------------------------------------------
# Wait for API
# ------------------------------------------------
echo "Waiting for Ollama API..."

for i in {1..30}; do
  if curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
    echo "Ollama API ready."
    break
  fi
  sleep 2
done

# ------------------------------------------------
# Pull Model
# ------------------------------------------------
echo "Ensuring model '$MODEL' exists..."

if [[ "$OS" == "Darwin" ]]; then
  ollama pull "$MODEL"
else
  docker exec ollama ollama pull "$MODEL"
fi

# ------------------------------------------------
# Run Validation
# ------------------------------------------------
echo ""
echo "======================================"
echo "Running Ollama validation tests..."
echo "======================================"

set +e
OLLAMA_MODEL="$MODEL" python llama_test.py
TEST_EXIT=$?
set -e

if [[ $TEST_EXIT -ne 0 ]]; then
  echo ""
  echo "Ollama validation FAILED."
  echo "Check container logs or run:"
  echo "   python llama_test.py"
  exit 1
fi

echo ""
echo "Ollama validation successful."

echo ""
echo "======================================"
echo "SETUP COMPLETE"
echo "======================================"
echo ""
echo "Activate environment:"
echo ""
echo "source $VENV_DIR/bin/activate"
echo ""