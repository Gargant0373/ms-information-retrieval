"""
config.py
~~~~~~~~~
Central configuration for the IR project pipeline.

All hardcoded values and environment-sensitive settings live here.
Override Ollama settings with environment variables:
    OLLAMA_URL      – base URL of the Ollama server  (default: http://localhost:11434)
    OLLAMA_MODEL    – model tag to use               (default: llama3.2:3b)
    OLLAMA_TIMEOUT  – HTTP timeout in seconds        (default: 60)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR    = str(PROJECT_ROOT / "indices" / "vaswani")
OUTPUT_DIR   = str(PROJECT_ROOT / "output")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "vaswani"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

BM25_MODEL = "BM25"

# Pseudo-relevance feedback (classical QE)
PRF_FB_DOCS  = 3
PRF_FB_TERMS = 10

# LLM synonym & paraphrase generation
LLM_SYNONYMS_N = 5

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_METRICS    = {"map", "ndcg", "ndcg_cut_10", "P_5", "P_10", "recall_10", "recall_100"}
DISPLAY_METRICS = ["map", "ndcg", "ndcg_cut_10", "P_5", "P_10", "recall_10", "recall_100"]

# ---------------------------------------------------------------------------
# LLM / Ollama
# ---------------------------------------------------------------------------

OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# llama-cpp-python fallback (used when Ollama's Metal backend is broken)
# ---------------------------------------------------------------------------

def find_gguf_model(model_name: str = None) -> str:
    """
    Locate the GGUF blob already downloaded by Ollama.

    Reads the Ollama manifest for *model_name* and returns the absolute path
    to the model blob file, or an empty string if not found.
    """
    import json
    from pathlib import Path

    if model_name is None:
        model_name = OLLAMA_MODEL
    parts = model_name.split(":")
    name, tag = parts[0], (parts[1] if len(parts) > 1 else "latest")

    manifest_path = (
        Path.home() / ".ollama" / "models" / "manifests"
        / "registry.ollama.ai" / "library" / name / tag
    )
    if not manifest_path.exists():
        return ""

    manifest = json.loads(manifest_path.read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            blob = Path.home() / ".ollama" / "models" / "blobs" / digest
            if blob.exists():
                return str(blob)
    return ""
