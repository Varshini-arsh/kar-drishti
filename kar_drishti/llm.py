"""
Client for a LOCAL language model. Supports:
  * Ollama            (default, http://localhost:11434)
  * OpenAI-compatible local servers such as LM Studio or llama.cpp server
    (set provider="openai", base_url="http://localhost:1234")
No data ever leaves the machine: only localhost URLs are allowed.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests

import config

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class LLMError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, provider: str = config.LLM_PROVIDER, base_url: str = config.LLM_BASE_URL,
                 model: str = config.LLM_MODEL, timeout: int = config.LLM_TIMEOUT_SEC,
                 allow_remote: bool = False):
        host = urlparse(base_url).hostname or ""
        if host not in LOCAL_HOSTS and not allow_remote:
            raise LLMError(f"Refusing non-local LLM address '{base_url}'. Kar Drishti only talks to a "
                           "model running on this computer (or set allow_remote=True for a server on "
                           "your own office network).")
        self.provider, self.base_url, self.model, self.timeout = provider, base_url.rstrip("/"), model, timeout

    # ------------------------------------------------------------------ status
    def list_models(self) -> list[str]:
        try:
            if self.provider == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=4)
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
            r = requests.get(f"{self.base_url}/v1/models", timeout=4)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            return []

    def status(self) -> tuple[bool, str]:
        models = self.list_models()
        if not models:
            return False, (f"No local model server found at {self.base_url}. Start Ollama "
                           f"(it runs in the background after install) and run: ollama pull {self.model}")
        if self.model not in models and not any(m.startswith(self.model) for m in models):
            return False, (f"Model '{self.model}' is not installed. Installed: {', '.join(models)}. "
                           f"Run: ollama pull {self.model}  (or pick an installed model in the sidebar)")
        return True, f"Connected to local model '{self.model}' via {self.provider}."

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float = config.LLM_TEMPERATURE) -> str:
        try:
            if self.provider == "ollama":
                body = {"model": self.model, "messages": messages, "stream": False,
                        "options": {"temperature": temperature, "num_ctx": 8192}}
                if json_mode:
                    body["format"] = "json"
                r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["message"]["content"]
            body = {"model": self.model, "messages": messages, "temperature": temperature}
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Cannot reach the local model at {self.base_url}. Is Ollama running?") from e
        except requests.exceptions.Timeout as e:
            raise LLMError("The local model took too long to answer. Try a smaller model "
                           "(e.g. qwen2.5:3b) or a shorter question.") from e
        except (KeyError, ValueError, requests.HTTPError) as e:
            raise LLMError(f"Unexpected reply from the local model server: {e}") from e

    def chat_json(self, messages: list[dict]) -> dict:
        text = self.chat(messages, json_mode=True)
        return parse_json(text)


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise LLMError(f"Model did not return valid JSON: {text[:200]}")
