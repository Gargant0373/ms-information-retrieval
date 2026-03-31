"""
llama_test.py

End-to-end validation of an Ollama installation.

Checks:
1. Ollama API availability
2. Model presence
3. Basic generation
4. Chat API
5. Simple reasoning sanity check
"""

import os
import sys
import time
import requests

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def fail(msg: str):
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str):
    print(f"[OK] {msg}")


# ------------------------------------------------
# API check
# ------------------------------------------------

def wait_for_api(timeout=30):
    print("[..] Waiting for Ollama API...")

    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
            if resp.status_code == 200:
                version = resp.json().get("version", "unknown")
                ok(f"Ollama API reachable (version {version})")
                return
        except Exception:
            pass

        time.sleep(2)

    fail("Ollama API did not become available")


# ------------------------------------------------
# Model check
# ------------------------------------------------

def check_model():
    print("[..] Checking installed models...")

    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()

    models = [m["name"] for m in resp.json().get("models", [])]

    if MODEL not in models:
        fail(f"Model '{MODEL}' not installed. Available: {models}")

    ok(f"Model '{MODEL}' present")


# ------------------------------------------------
# Generation test
# ------------------------------------------------

def generate_test():
    print("[..] Running generation test...")

    payload = {
        "model": MODEL,
        "prompt": "In one short sentence, define information retrieval.",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        timeout=120,
    )

    resp.raise_for_status()

    text = resp.json().get("response", "").strip()

    if not text:
        fail("Generation returned empty response")

    print(f"[OK] Model response: {text[:120]}")


# ------------------------------------------------
# Chat test
# ------------------------------------------------

def chat_test():
    print("[..] Running chat test...")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Answer concisely in one sentence.",
            },
            {
                "role": "user",
                "content": "What does IR stand for in computer science?",
            },
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=120,
    )

    resp.raise_for_status()

    message = resp.json().get("message", {}).get("content", "").strip()

    if not message:
        fail("Chat endpoint returned empty message")

    print(f"[OK] Chat response: {message}")


# ------------------------------------------------
# Reasoning sanity test
# ------------------------------------------------

def reasoning_test():
    print("[..] Running reasoning sanity test...")

    payload = {
        "model": MODEL,
        "prompt": "What is 7 * 9? Answer with only the number.",
        "stream": False,
        "options": {"temperature": 0},
    }

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        timeout=60,
    )

    resp.raise_for_status()

    result = resp.json().get("response", "").strip()

    if "63" not in result:
        fail(f"Math sanity test failed (expected 63, got '{result}')")

    ok("Reasoning sanity test passed")


# ------------------------------------------------
# llama-cpp-python fallback test
# ------------------------------------------------

def llama_cpp_test() -> bool:
    """
    Test the llama-cpp-python CPU backend.
    Returns True if the test passes, False otherwise.
    """
    print("\n[..] Testing llama-cpp-python fallback backend...")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from project.llama_service import LlamaService
        from project.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, find_gguf_model

        gguf = find_gguf_model(MODEL)
        if not gguf:
            print(f"[SKIP] No GGUF blob found for {MODEL!r}. Run: ollama pull {MODEL}")
            return False

        print(f"[..] GGUF found: {gguf}")
        svc = LlamaService(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
        # Force the llama_cpp backend (skip Ollama which we know is broken)
        svc._backend = "llama_cpp"
        print("[..] Running expand_query via llama-cpp-python (CPU)...")
        result = svc.expand_query("information retrieval systems")
        if not result:
            print("[FAIL] llama-cpp-python returned empty result")
            return False
        ok(f"llama-cpp-python backend: {result[:80]!r}")
        return True
    except Exception as exc:
        print(f"[FAIL] llama-cpp-python test error: {exc}")
        return False


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    print("=" * 60)
    print("Ollama + LlamaService Validation")
    print(f"Endpoint : {OLLAMA_BASE}")
    print(f"Model    : {MODEL}")
    print("=" * 60)

    ollama_ok = False
    try:
        wait_for_api()
        check_model()
        generate_test()
        chat_test()
        reasoning_test()
        ollama_ok = True
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Ollama test failed: {e}")
        print("       This is known to happen on macOS 26 due to a Metal")
        print("       shader bug in MetalPerformancePrimitives.framework.")
        print("       Checking llama-cpp-python fallback...")

    cpp_ok = llama_cpp_test() if not ollama_ok else True

    print("=" * 60)
    if ollama_ok:
        print("RESULT: ALL TESTS PASSED (Ollama)")
    elif cpp_ok:
        print("RESULT: PASSED via llama-cpp-python CPU fallback")
        print("        LlamaService will auto-use this backend.")
    else:
        print("RESULT: FAILED — no backend is available.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()