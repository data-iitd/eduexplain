"""
OpenAI-compatible vLLM endpoint client. This module never loads model weights.
Start one vLLM server per model and point the pipeline at their /v1 endpoints.
"""
import json
import re
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    name: str
    base_url: str
    api_key: str = "EMPTY"
    model: Optional[str] = None
    timeout: int = 600


def sleep_level_for(base_url):
    """vLLM sleep level for a server: VLLM_SLEEP_LEVELS="url=2,..." (default 1). Level 1 keeps the weights in pinned
    CPU RAM (fast wake); level 2 drops them and reloads from disk on wake, for a model too big to park in RAM
    (VideoExplain's Llama-3.1-70B)."""
    for item in os.getenv("VLLM_SLEEP_LEVELS", "").split(","):
        url, _, level = item.strip().rpartition("=")
        if url and url.rstrip("/") == base_url.rstrip("/"):
            return int(level)
    return 1


class VLLMClient:
    """Small dependency-free client for vLLM's OpenAI-compatible chat API."""
    def __init__(self, name, base_url=None, model=None, api_key=None, timeout=None):
        env_prefix = name.upper().replace("-", "_").replace(".", "_")
        self.name = name
        self.base_url = (base_url or os.getenv(f"{env_prefix}_BASE_URL")
                         or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.getenv(f"{env_prefix}_MODEL") or name
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.timeout = int(timeout or os.getenv("VLLM_TIMEOUT", "600"))
        self.last_new_tokens = 0
        self.last_prompt_tail = None
        self.last_usage = {}
        self.swap_peer = None   # another VLLMClient sharing this GPU (see share_gpu)
        self.sleep_level = sleep_level_for(self.base_url)

    # ------------------------------------------------------------ GPU sharing
    # Both models live on one GPU: the idle server is put to sleep (weights offloaded to CPU RAM, KV cache freed)
    # before the other one runs. Needs `vllm serve --enable-sleep-mode` and VLLM_SERVER_DEV_MODE=1 (serve_models.sh).
    def _admin(self, path, method="POST"):
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        req = urllib.request.Request(f"{root}{path}", data=b"" if method == "POST" else None, method=method,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def is_sleeping(self) -> bool:
        return json.loads(self._admin("/is_sleeping", "GET")).get("is_sleeping", False)

    def sleep(self):
        if not self.is_sleeping():
            self._admin(f"/sleep?level={self.sleep_level}")

    def wake(self):
        if not self.is_sleeping():
            return
        if self.sleep_level == 2:
            # level 2 discarded the weights (no pinned CPU copy): restore the buffers, reload from disk, then the KV cache
            self._admin("/wake_up?tags=weights")
            self._admin_json("/collective_rpc", {"method": "reload_weights"})
            self._admin("/wake_up?tags=kv_cache")
        else:
            self._admin("/wake_up")

    def _admin_json(self, path, payload):
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        req = urllib.request.Request(f"{root}{path}", data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=max(self.timeout, 900)) as response:
            return response.read().decode("utf-8")

    def _extra_peers(self):
        """Other sleepable vLLM servers on this GPU (VLLM_GPU_PEERS=url1,url2,...), e.g. VideoExplain's gemma :8002."""
        urls = [u.strip().rstrip("/") for u in os.getenv("VLLM_GPU_PEERS", "").split(",") if u.strip()]
        known = {self.base_url} | ({self.swap_peer.base_url} if self.swap_peer else set())
        return [VLLMClient(f"peer{k}", u, "peer", self.api_key, 120) for k, u in enumerate(urls) if u not in known]

    def _make_room(self):
        extra = self._extra_peers()
        if self.swap_peer is None and not extra:
            return
        for peer in extra:   # best effort: a peer that is down simply holds no GPU memory
            try:
                peer.sleep()
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                pass
        if self.swap_peer is None:
            try:
                self.wake()
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                pass
            return
        try:
            self.swap_peer.sleep()
            self.wake()
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                # e.g. wake_up OOM: generating on a still-sleeping engine would hang, so fail loudly instead
                detail = exc.read().decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(f"GPU swap failed on {self.base_url} (HTTP {exc.code}): {detail}") from exc
            self._disable_swap(exc)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            self._disable_swap(exc)

    def _disable_swap(self, exc):
        print(f"  ! GPU swap disabled ({self.base_url}): sleep/wake endpoints unavailable: {exc}", flush=True)
        self.swap_peer.swap_peer = self.swap_peer = None

    def count_tokens(self, messages):
        """Exact prompt length (chat template applied) via vLLM's /tokenize endpoint."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        req = urllib.request.Request(f"{root}/tokenize", method="POST",
                                     data=json.dumps({"model": self.model, "messages": messages}).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return int(json.loads(response.read().decode("utf-8"))["count"])

    def generate(self, messages, max_new_tokens=2048, thinking=False,
                 temperature=0.0, **overrides) -> str:
        # vLLM's OpenAI-compatible API uses max_tokens on broad versions and
        # supports chat_template_kwargs on current versions for Qwen thinking mode.
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature or 0.0),
        }
        for key in ("top_p", "top_k", "repetition_penalty", "presence_penalty",
                    "frequency_penalty", "stop"):
            if overrides.get(key) is not None:
                body[key] = overrides[key]
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        self._make_room()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # Long fix-round prompts can exceed the context window with a fixed output budget: retry once with
            # exactly the room that is left (vLLM reports the window and the prompt size in the error).
            # ("prompt contains at least N" is only a bound, so the exact prompt size comes from /tokenize)
            m = re.search(r"maximum context length is (\d+)", detail)
            if exc.code == 400 and m and not overrides.get("_retried"):
                room = int(m.group(1)) - self.count_tokens(messages) - 64
                if room >= 800:
                    return self.generate(messages, max_new_tokens=room, thinking=thinking, temperature=temperature,
                                         _retried=True, **{k: v for k, v in overrides.items() if k != "_retried"})
            raise RuntimeError(f"vLLM endpoint {self.base_url} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach vLLM endpoint {self.base_url}. Is the server running and reachable? {exc}"
            ) from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM response contains no choices: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = data.get("usage") or {}
        self.last_new_tokens = int(usage.get("completion_tokens", 0))
        self.last_finish_reason = choices[0].get("finish_reason")   # "length": cut off at max_tokens
        self.last_usage = usage
        self.last_prompt_tail = None
        return str(content)


def share_gpu(a: VLLMClient, b: VLLMClient):
    """Make two clients swap one GPU: whichever is about to generate puts the other's server to sleep first."""
    if a.base_url != b.base_url:
        a.swap_peer, b.swap_peer = b, a


def client_from_env(role: str, model: str, base_url=None) -> VLLMClient:
    """Construct a client from role-specific environment variables."""
    prefix = "VLLM_GENERATOR" if role == "generator" else "VLLM_VERIFIER"
    return VLLMClient(
        name=model,
        base_url=base_url or os.getenv(f"{prefix}_BASE_URL"),
        model=os.getenv(f"{prefix}_MODEL", model),
        api_key=os.getenv(f"{prefix}_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY")),
        timeout=os.getenv("VLLM_TIMEOUT", "600"),
    )
