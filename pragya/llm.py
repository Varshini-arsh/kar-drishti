"""The local language model used by Pragya's agents.

Rules (from the brief):
  * Runtime: Ollama (default) or any OpenAI-compatible local server (LM Studio, llama.cpp).
  * Default model qwen3.5:4b with thinking OFF; fallback qwen2.5:3b;
    qwen3.5:9b only when at least 8 GB of memory is free.
  * Temperature 0 and a fixed seed, so the same question gives the same answer.
  * Never download a model. Only local addresses, unless an admin setting allows a
    "Department server" (an OpenAI-compatible address on the office network).
  * At most 2 model calls per question (one to plan, one to write): see Budget.

FakeLLM gives fixed, repeatable replies for the tests (no model needed).
"""
from __future__ import annotations

import os
import re

import requests

from kar_drishti.llm import LLMError, LocalLLM, parse_json  # noqa: F401  (re-exported)

DEFAULT_MODEL = os.getenv("PRAGYA_MODEL", "qwen3.5:4b")
FALLBACK_MODELS = ["qwen2.5:3b"]
BIG_MODEL = "qwen3.5:9b"
BIG_MODEL_MIN_FREE_GB = 8.0
SEED = 42
TEMPERATURE = 0.0
MAX_CALLS_PER_QUESTION = 2
# Admin switch: allow a non-local "Department server" address (off by default).
ALLOW_DEPARTMENT_SERVER = os.getenv("PRAGYA_ALLOW_DEPARTMENT_SERVER", "0") == "1"

# Optional cloud runtime: Groq (OpenAI-compatible, needs internet and an API key).
# Off unless the user picks it in Settings > Model or sets KD_LLM_PROVIDER=groq / GROQ_API_KEY.
# With Groq, the question and the facts needed to answer it are sent to Groq: use it with demo data.
GROQ_URL = "https://api.groq.com/openai"
GROQ_DEFAULT_MODEL = os.getenv("PRAGYA_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_PREFERRED = [GROQ_DEFAULT_MODEL, "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]


def groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def choose_groq_model(models: list[str], wanted: str | None = None) -> tuple[str | None, str]:
    if not models:
        return None, "Groq did not list any models. Check the API key and the internet connection."
    if wanted in models:
        return wanted, f"Using {wanted} on Groq (cloud)."
    pick = next((m for m in GROQ_PREFERRED if m in models), None) or next(
        (m for m in models if "llama" in m and "guard" not in m), models[0])
    return pick, f"Using {pick} on Groq (cloud)."


def _installed(name: str, models: list[str]) -> str | None:
    """Exact installed name for `name` ('qwen2.5:3b' also matches 'qwen2.5:3b-instruct-q4_K_M')."""
    if name in models:
        return name
    return next((m for m in models if m.startswith(name)), None)


def choose_model(models: list[str], free_gb: float | None, wanted: str | None = None) -> tuple[str | None, str]:
    """Pick the model to use from the INSTALLED ones. Returns (model or None, plain-English note).
    Never suggests downloading anything automatically."""
    if not models:
        return None, "No local models found. Install one yourself, e.g. in PowerShell: ollama pull qwen2.5:3b"
    if wanted:
        hit = _installed(wanted, models)
        if hit and wanted.startswith(BIG_MODEL) and (free_gb is None or free_gb < BIG_MODEL_MIN_FREE_GB):
            note = (f"{BIG_MODEL} needs at least {BIG_MODEL_MIN_FREE_GB:.0f} GB of free memory "
                    f"({'unknown' if free_gb is None else f'{free_gb:.1f} GB'} free now), so a smaller model is used.")
            m, _ = choose_model(models, free_gb)
            return m, note
        if hit:
            return hit, f"Using {hit}."
    for name in [DEFAULT_MODEL] + FALLBACK_MODELS:
        hit = _installed(name, models)
        if hit:
            return hit, f"Using {hit}." if name == DEFAULT_MODEL else \
                f"{DEFAULT_MODEL} is not installed, so the fallback {hit} is used."
    small = [m for m in models if m.startswith("qwen") and not m.startswith(BIG_MODEL)]
    m = small[0] if small else models[0]
    return m, f"Neither {DEFAULT_MODEL} nor {FALLBACK_MODELS[0]} is installed; using {m}."


def uses_thinking_switch(model: str) -> bool:
    """Qwen3-family models can 'think' before answering; Pragya turns that off for speed."""
    return bool(re.match(r"qwen3", model or ""))


class Budget:
    """Counts model calls for one question. Agents ask before each call."""

    def __init__(self, limit: int = MAX_CALLS_PER_QUESTION):
        self.limit, self.used, self.log = limit, 0, []

    @property
    def left(self) -> int:
        return self.limit - self.used

    def take(self, purpose: str) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        self.log.append(purpose)
        return True


class PragyaLLM(LocalLLM):
    """LocalLLM with temperature 0, a fixed seed and thinking switched off."""

    def __init__(self, provider: str = "ollama", base_url: str = "http://localhost:11434",
                 model: str = DEFAULT_MODEL, timeout: int = 240, department_server: bool = False,
                 api_key: str | None = None):
        is_groq = provider == "groq"
        super().__init__(provider, GROQ_URL if is_groq else base_url, model, 60 if is_groq else timeout,
                         allow_remote=is_groq or (department_server and ALLOW_DEPARTMENT_SERVER))
        self.api_key = (api_key or groq_key()) if is_groq else (api_key or "")

    @property
    def where(self) -> str:
        return "Groq (cloud)" if self.provider == "groq" else "the local model"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def list_models(self) -> list[str]:
        if self.provider != "groq":
            return super().list_models()
        if not self.api_key:
            return []
        try:
            r = requests.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=8)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []) if m.get("active", True)
                          and not any(w in m["id"] for w in ("whisper", "tts", "guard", "prompt-")))
        except Exception:
            return []

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = TEMPERATURE,
             purpose: str = "") -> str:
        try:
            if self.provider == "ollama":
                body = {"model": self.model, "messages": messages, "stream": False,
                        "options": {"temperature": temperature, "seed": SEED, "num_ctx": 8192}}
                if uses_thinking_switch(self.model):
                    body["think"] = False
                if json_mode:
                    body["format"] = "json"
                r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
                if r.status_code == 400 and "think" in r.text.lower():       # older Ollama: no 'think' option
                    body.pop("think", None)
                    r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
                r.raise_for_status()
                return strip_thinking(r.json()["message"]["content"])
            body = {"model": self.model, "messages": messages, "temperature": temperature, "seed": SEED}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=body, headers=self._headers(),
                              timeout=self.timeout)
            if r.status_code == 400 and (json_mode or "seed" in r.text.lower()):   # option not supported: retry plain
                body.pop("response_format", None)
                body.pop("seed", None)
                r = requests.post(f"{self.base_url}/v1/chat/completions", json=body, headers=self._headers(),
                                  timeout=self.timeout)
            if self.provider == "groq" and r.status_code == 401:
                raise LLMError("Groq did not accept the API key. Check it in Settings > Model.")
            if self.provider == "groq" and r.status_code == 429:
                raise LLMError("Groq's free-tier limit was reached. Wait a minute and ask again.")
            r.raise_for_status()
            return strip_thinking(r.json()["choices"][0]["message"]["content"])
        except requests.exceptions.ConnectionError as e:
            if self.provider == "groq":
                raise LLMError("Cannot reach Groq. Groq needs an internet connection.") from e
            raise LLMError(f"Cannot reach the local model at {self.base_url}. Is Ollama running?") from e
        except requests.exceptions.Timeout as e:
            raise LLMError(f"{self.where.capitalize()} took too long to answer. Try a shorter question.") from e
        except (KeyError, ValueError, requests.HTTPError) as e:
            raise LLMError(f"Unexpected reply from {self.where}: {e}") from e

    def chat_json(self, messages: list[dict], purpose: str = "") -> dict:
        return parse_json(self.chat(messages, json_mode=True, purpose=purpose))


def strip_thinking(text: str) -> str:
    """Remove any <think>...</think> block a model may still produce."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class FakeLLM:
    """Repeatable stand-in for the real model, used by tests and the self-test.

    replies: {purpose: reply or callable(messages) -> reply}. Purposes used by Pragya:
      "plan"  -> JSON text choosing agents
      "sql"   -> a SQL query in a ```sql block
      "write" -> the answer text (default: repeat the facts it was given, which always passes the checker)
    """

    def __init__(self, replies: dict | None = None, model: str = "fake-model"):
        self.replies, self.model, self.calls = dict(replies or {}), model, []

    def chat(self, messages, json_mode=False, temperature=0.0, purpose=""):
        self.calls.append(purpose)
        r = self.replies.get(purpose)
        if callable(r):
            r = r(messages)
        if r is None and purpose == "write":
            facts = messages[-1]["content"].split("FACTS:", 1)[-1]
            lines = [ln.strip("- ").strip() for ln in facts.splitlines() if ln.strip()][:5]
            r = "\n".join(lines)
        if r is None:
            raise LLMError(f"FakeLLM has no reply for '{purpose}'")
        return r

    def chat_json(self, messages, purpose=""):
        return parse_json(self.chat(messages, json_mode=True, purpose=purpose))
