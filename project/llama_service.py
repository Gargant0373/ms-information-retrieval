"""
llama_service.py
~~~~~~~~~~~~~~~~
A lightweight wrapper around the local Ollama API for LLM-assisted
query expansion / rewriting in an Information Retrieval pipeline.

Quick start
-----------
    from project.llama_service import LlamaService

    svc = LlamaService()                    # uses defaults
    expanded = svc.expand_query("what is blaphsemy")
    # → "blasphemy definition meaning religion"

Usage with PyTerrier (util.evaluate_all)
-----------------------------------------
    from project.llama_service import LlamaService
    from util import evaluate_all

    svc = LlamaService()
    print(evaluate_all(svc.expand_query))

Environment variables
---------------------
    OLLAMA_URL    – base URL of the Ollama server  (default: http://localhost:11434)
    OLLAMA_MODEL  – model tag to use               (default: llama3.2:3b)
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (can be overridden via env vars or constructor args)
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_DEFAULT_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Prompt template for query expansion / rewriting
_EXPANSION_SYSTEM = (
    "You are an expert Information Retrieval assistant. "
    "Your job is to rewrite or expand user search queries so they work "
    "better with a keyword-based search engine (BM25). "
    "Return ONLY the rewritten query — no explanation, no punctuation, "
    "no quotes, no bullet points."
)

_EXPANSION_USER_TMPL = (
    "Rewrite the following search query to improve retrieval performance. "
    "Add relevant synonyms, related terms, or clarifications. "
    "Keep the result concise (5–15 words).\n\n"
    "Original query: {query}\n\n"
    "Rewritten query:"
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LlamaService:
    """
    Thin client for the Ollama REST API, tailored for IR query expansion.

    Parameters
    ----------
    base_url : str
        Base URL of the Ollama server (default: ``http://localhost:11434``).
    model : str
        Ollama model tag (default: ``llama3.2:3b``).
    timeout : int
        HTTP timeout in seconds for generation requests (default: 60).
    temperature : float
        Sampling temperature — 0 gives deterministic output (default: 0.0).
    num_predict : int
        Maximum number of tokens to generate (default: 64).
    system_prompt : str | None
        Override the default system prompt for ``expand_query``.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: int = 60,
        temperature: float = 0.0,
        num_predict: int = 64,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.system_prompt = system_prompt or _EXPANSION_SYSTEM
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama API is reachable."""
        try:
            r = self._session.get(f"{self.base_url}/api/version", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: int = 30, poll: float = 2.0) -> None:
        """
        Block until the Ollama API is reachable or *timeout* seconds elapse.

        Raises
        ------
        TimeoutError
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_available():
                return
            time.sleep(poll)
        raise TimeoutError(
            f"Ollama API at {self.base_url} did not become available "
            f"within {timeout} s."
        )

    def list_models(self) -> list[str]:
        """Return a list of locally available model tags."""
        r = self._session.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Send a single-turn chat request and return the assistant reply.

        Parameters
        ----------
        user_message : str
            The user-turn content.
        system_message : str | None
            Optional system prompt override.
        temperature : float | None
            Override instance-level temperature.
        num_predict : int | None
            Override instance-level num_predict.

        Returns
        -------
        str
            The model's reply (stripped).
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": num_predict if num_predict is not None else self.num_predict,
            },
        }

        r = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Send a raw generation request and return the response text.

        Parameters
        ----------
        prompt : str
            The full prompt string.
        system : str | None
            Optional system prompt.
        temperature : float | None
            Override instance-level temperature.
        num_predict : int | None
            Override instance-level num_predict.

        Returns
        -------
        str
            The model's response (stripped).
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": num_predict if num_predict is not None else self.num_predict,
            },
        }
        if system:
            payload["system"] = system

        r = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    # ------------------------------------------------------------------
    # Query expansion / rewriting
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> str:
        """
        Rewrite/expand a search query using the LLM.

        This is the main entry-point for the IR pipeline and is designed
        to be passed directly to ``util.evaluate_all``:

            evaluate_all(svc.expand_query)

        The method falls back to the original query on any error, so it
        is safe to use in batch evaluation.

        Parameters
        ----------
        query : str
            The original search query.

        Returns
        -------
        str
            The rewritten query (falls back to *query* on failure).
        """
        try:
            user_msg = _EXPANSION_USER_TMPL.format(query=query)
            expanded = self.chat(
                user_message=user_msg,
                system_message=self.system_prompt,
            )
            # Strip any residual punctuation / quotes the model may add
            expanded = _clean_query(expanded)
            if not expanded:
                return query
            logger.debug("expand_query: %r → %r", query, expanded)
            return expanded
        except Exception as exc:  # noqa: BLE001
            logger.warning("expand_query failed for %r: %s", query, exc)
            return query

    def expand_query_multi(
        self,
        query: str,
        n: int = 3,
        separator: str = " ",
    ) -> str:
        """
        Generate *n* alternative query formulations and concatenate them.

        Useful for pseudo-relevance feedback style expansion where you
        want high recall.

        Parameters
        ----------
        query : str
            The original search query.
        n : int
            Number of alternative formulations to generate.
        separator : str
            String used to join the alternatives (default: single space).

        Returns
        -------
        str
            All formulations joined by *separator*, or *query* on failure.
        """
        system = (
            "You are an expert Information Retrieval assistant. "
            f"Generate exactly {n} alternative search queries for the given "
            "query. Output only the queries, one per line, no numbering, "
            "no explanation."
        )
        user_msg = f"Original query: {query}\n\nAlternative queries:"

        try:
            raw = self.chat(user_message=user_msg, system_message=system)
            lines = [_clean_query(l) for l in raw.splitlines() if l.strip()]
            alternatives = [l for l in lines if l][:n]
            if not alternatives:
                return query
            combined = separator.join([query] + alternatives)
            logger.debug("expand_query_multi: %r → %r", query, combined)
            return combined
        except Exception as exc:  # noqa: BLE001
            logger.warning("expand_query_multi failed for %r: %s", query, exc)
            return query

    def rewrite_with_context(
        self,
        query: str,
        context: str,
        instruction: Optional[str] = None,
    ) -> str:
        """
        Rewrite a query given additional domain context.

        Parameters
        ----------
        query : str
            The original search query.
        context : str
            Background context (e.g. a retrieved document snippet,
            a topic description, or a domain hint).
        instruction : str | None
            Optional extra instruction appended to the system prompt.

        Returns
        -------
        str
            The rewritten query, or *query* on failure.
        """
        system = (
            "You are an expert Information Retrieval assistant. "
            "Given a search query and additional context, rewrite the query "
            "to better capture the information need. "
            "Return ONLY the rewritten query — no explanation, no quotes."
        )
        if instruction:
            system += f" {instruction}"

        user_msg = (
            f"Context: {context}\n\n"
            f"Original query: {query}\n\n"
            "Rewritten query:"
        )

        try:
            rewritten = self.chat(user_message=user_msg, system_message=system)
            rewritten = _clean_query(rewritten)
            return rewritten or query
        except Exception as exc:  # noqa: BLE001
            logger.warning("rewrite_with_context failed for %r: %s", query, exc)
            return query

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LlamaService(model={self.model!r}, base_url={self.base_url!r})"
        )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _clean_query(text: str) -> str:
    """Strip surrounding quotes, bullet symbols, and excess whitespace."""
    text = text.strip()
    # Remove leading bullet / numbering artefacts
    text = re.sub(r"^[\-\*\d\.\)]+\s*", "", text)
    # Remove wrapping quotes
    text = text.strip("\"'`")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
