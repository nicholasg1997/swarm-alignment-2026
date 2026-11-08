"""Model backends: a scripted replay backend for offline runs, OpenRouter through an
OpenAI-compatible client, and Adaption through its Responses API."""
import os
from pathlib import Path
from typing import Optional, Any, Callable, Dict, List, Union

import yaml

from .troy import BaseModelBackend, GenerationResult, Usage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_STOP = '{"action":"STOP"}'


class ScriptedBackend(BaseModelBackend):
    """Replays a fixed list of replies; an Exception item is raised instead.
    Once the script is exhausted every reply is STOP."""

    def __init__(self, script: List[Union[str, Exception]],
                 offer_script: List[str] = ()) -> None:
        super().__init__(model_id="scripted")
        self._script = list(script)
        self._offers = list(offer_script)

    @property
    def provider(self) -> str:
        return "scripted"

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        last = messages[-1]["content"] if messages else ""
        if "This is NOT a turn" in last:  # board offer (world.py): own script
            item = self._offers.pop(0) if self._offers else (
                '{"read": true}' if '"read"' in last else '{"message": null}')
        else:
            item = self._script.pop(0) if self._script else _STOP
        if isinstance(item, Exception):
            raise item
        prompt_chars = len(system) + sum(len(m["content"]) for m in messages)
        usage = Usage(input_tokens=prompt_chars // 4, output_tokens=len(item) // 4, calls=1)
        self._record(usage)
        return GenerationResult(text=item, usage=usage)


class OpenRouterBackend(BaseModelBackend):
    """OpenAI-compatible chat backend for OpenRouter. Unlike Troy's OpenAIBackend it
    captures OpenRouter's `reasoning` field, which the forced-action retry needs."""

    def __init__(self, model_id: str, temperature: float = 0.7,
                 request_timeout_s: float = 120, client: Any = None,
                 reasoning_effort: Optional[str] = None, **kw: Any) -> None:
        super().__init__(model_id=model_id, temperature=temperature, **kw)
        self.base_url, self.api_key_env = OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"
        self.request_timeout_s = request_timeout_s
        #: pinned per model config so it is a fixed setting, not the provider's default
        self.reasoning_effort = reasoning_effort
        self._client = client

    @property
    def provider(self) -> str:
        return "openrouter"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        extra = ({"extra_body": {"reasoning": {"effort": self.reasoning_effort}}}
                 if self.reasoning_effort else {})
        resp = self._get_client().chat.completions.create(
            model=self.model_id, messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens, temperature=self.temperature,
            timeout=self.request_timeout_s, **extra)
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        usage = Usage(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                      output_tokens=getattr(u, "completion_tokens", 0) or 0,
                      reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0, calls=1)
        self._record(usage)
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        return GenerationResult(text=msg.content or "", usage=usage, reasoning_text=reasoning,
                                raw={"finish_reason": getattr(choice, "finish_reason", None)})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Field access that works on SDK objects and plain dicts."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class AdaptionBackend(BaseModelBackend):
    """Adaption Responses API (`adaption` SDK). The system prompt goes in `instructions`,
    the chat in `input`. Open models return their reasoning as `reasoning_text` parts."""

    def __init__(self, model_id: str, temperature: float = 0.7,
                 request_timeout_s: float = 120, max_retries: int = 4,
                 client: Any = None, **kw: Any) -> None:
        super().__init__(model_id=model_id, temperature=temperature, **kw)
        self.api_key_env = "ADAPTION_API_KEY"
        self.request_timeout_s, self.max_retries = request_timeout_s, max_retries
        self.reasoning_effort = None  # the API exposes no effort setting
        self._client = client

    @property
    def provider(self) -> str:
        return "adaption"

    def _get_client(self):
        if self._client is None:
            from adaption import Adaption
            if not os.environ.get(self.api_key_env):
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = Adaption(timeout=self.request_timeout_s, max_retries=self.max_retries)
        return self._client

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        resp = self._get_client().responses.create(
            model=self.model_id, instructions=system, input=list(messages),
            max_output_tokens=max_tokens, temperature=self.temperature,
            agentic_memory=False, store=False, previous_response_id=None,
            tools=[], tool_choice="none")
        provider_usage = _get(resp, "usage")
        u = provider_usage.to_dict() if hasattr(provider_usage, "to_dict") else dict(provider_usage or {})
        usage = Usage(input_tokens=u.get("input_tokens") or 0,
                      cached_input_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens") or 0,
                      output_tokens=u.get("output_tokens") or 0,
                      reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
                      calls=1)
        self._record(usage)
        status = _get(resp, "status")
        if _get(resp, "error") or status not in ("completed", "incomplete"):
            raise RuntimeError(f"Adaption response failed: status={status!r}, error={_get(resp, 'error')!r}")
        thoughts = []
        for item in _get(resp, "output") or []:
            if _get(item, "type") == "reasoning":
                for part in list(_get(item, "content") or []) + list(_get(item, "summary") or []):
                    if _get(part, "text"):
                        thoughts.append(_get(part, "text"))
        reason = _get(_get(resp, "incomplete_details"), "reason")
        finish = ("length" if reason == "max_output_tokens"
                  else "stop" if status == "completed" else status)
        return GenerationResult(text=_get(resp, "output_text") or "", usage=usage,
                                reasoning_text="\n".join(thoughts) or None,
                                raw={"finish_reason": finish, "cost": u.get("cost"),
                                     "response_id": _get(resp, "id"),
                                     "provider_model": _get(resp, "model"),
                                     "provider_status": status})


def load_dotenv(path: Union[str, Path]) -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_model_config(path: Union[str, Path]) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_backend_factory(model_cfg: Dict[str, Any]) -> Callable[[str], BaseModelBackend]:
    """Returns agent_id -> fresh backend for that agent."""
    kind = model_cfg.get("backend")
    if kind == "mix":  # a different model per agent: {"agents": {a1: <yaml>}, "default": <yaml>}
        subs = {a: load_model_config(path) for a, path in (model_cfg.get("agents") or {}).items()}
        default = load_model_config(model_cfg["default"]) if model_cfg.get("default") else None

        def mixed(agent_id: str) -> BaseModelBackend:
            sub = subs.get(agent_id, default)
            if sub is None:
                raise ValueError(f"mix config has no model for {agent_id} and no default")
            b = make_backend_factory(sub)(agent_id)
            b.model_name = sub.get("name", sub.get("model_id"))
            return b
        return mixed
    if kind == "scripted":
        scripts = model_cfg.get("scripts", {})
        default = model_cfg.get("default_script", [])
        offers = model_cfg.get("offer_scripts", {})
        return lambda agent_id: ScriptedBackend(scripts.get(agent_id, default),
                                                offers.get(agent_id, []))
    if kind == "openrouter":
        def openrouter(agent_id: str) -> BaseModelBackend:
            b = OpenRouterBackend(model_cfg["model_id"],
                                  temperature=model_cfg.get("temperature", 0.7),
                                  request_timeout_s=model_cfg.get("request_timeout_s", 120),
                                  reasoning_effort=model_cfg.get("reasoning_effort"))
            b.max_output_tokens = int(model_cfg.get("max_tokens", 2048))
            return b
        return openrouter
    if kind == "adaption":
        def adaption(agent_id: str) -> BaseModelBackend:
            b = AdaptionBackend(model_cfg["model_id"],
                                temperature=model_cfg.get("temperature", 0.7),
                                request_timeout_s=model_cfg.get("request_timeout_s", 120),
                                max_retries=model_cfg.get("max_retries", 4))
            b.max_output_tokens = int(model_cfg.get("max_tokens", 2048))
            return b
        return adaption
    raise ValueError(f"unknown backend {kind!r}")
