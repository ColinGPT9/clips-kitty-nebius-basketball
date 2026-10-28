"""OpenAI-compatible chat completions: OpenRouter, xAI, Meta, and any future
provider that speaks it (a new one is a ProviderSpec, not new code)."""

import json
import re

from llm.base import ChatTurn, ToolCall
from llm.providers import usage
from llm.providers.base import LLMError, ModelInfo, ProviderSpec
from llm.providers.http import send

# How long an answer may run, as the Anthropic adapter has always said and
# Ollama's num_predict does locally. Without it a model stuck repeating itself
# runs to the provider's own limit: one OpenRouter request of a two-hour
# stream's scoring was still going after twenty minutes. Room enough for a
# reasoning model's thinking and the answer (gpt-oss used ~3,300 on a chunk).
GENERATE_MAX_TOKENS = 8192
CHAT_MAX_TOKENS = 4096

# The response format each (provider, model) turned out to accept. A model
# that refuses a strict schema is asked for plain JSON next time straight away,
# instead of failing the same way on every chunk of a long video.
_working_format: dict[tuple[str, str], int] = {}


def _formats(json_mode: bool, schema: dict | None) -> list[dict | None]:
    formats: list[dict | None] = []
    if schema:
        formats.append({"type": "json_schema",
                        "json_schema": {"name": "clips_kitty", "strict": True, "schema": schema}})
    if json_mode or schema:
        formats.append({"type": "json_object"})
    formats.append(None)
    return formats


def generate(spec: ProviderSpec, key: str, model: str, prompt: str, *,
             json_mode: bool = False, schema: dict | None = None) -> str:
    formats = _formats(json_mode, schema)
    start = _working_format.get((spec.id, model), 0) if len(formats) > 1 else 0
    last: LLMError | None = None
    for index in range(min(start, len(formats) - 1), len(formats)):
        body: dict = {"model": model, "max_tokens": GENERATE_MAX_TOKENS,
                      "messages": [{"role": "user", "content": prompt}]}
        if formats[index]:
            body["response_format"] = formats[index]
        try:
            data = send(spec, key, "POST", "/chat/completions", json_body=spec.body_extras(body))
            usage.record(spec.id, model, data)
        except LLMError as e:
            # Only a refusal of the format is worth another try with a looser
            # one. The prompt asks for the same JSON either way, and every
            # caller parses defensively.
            if formats[index] is None or e.kind not in ("rejected", "model_unavailable"):
                raise
            last = e
            continue
        if formats[index] and formats[index].get("type") == "json_schema" and index + 1 < len(formats) \
                and _choice(spec, data).get("finish_reason") == "length":
            # Held to the schema, the model ran to the output limit: stuck
            # where the schema and what it wanted to write disagree, writing
            # whitespace (measured: 28,000 spaces in one answer). A looser
            # format next, now and for this model from then on.
            last = LLMError("bad_response", f"The {spec.label} model ran out of room in the strict format.")
            _working_format[(spec.id, model)] = index + 1
            continue
        _working_format[(spec.id, model)] = index
        return _text(spec, data)
    raise last or LLMError("rejected", f"{spec.label} refused every way of asking for this answer.")


def chat(spec: ProviderSpec, key: str, model: str, messages: list[dict], tools: list[dict]) -> ChatTurn:
    body: dict = {"model": model, "max_tokens": CHAT_MAX_TOKENS, "messages": [_message(m) for m in messages]}
    if tools:
        body["tools"] = tools
    data = send(spec, key, "POST", "/chat/completions", json_body=spec.body_extras(body))
    usage.record(spec.id, model, data)
    message = _choice(spec, data).get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        calls.append(ToolCall(
            id=str(call.get("id") or fn.get("name") or ""),
            name=str(fn.get("name") or ""),
            arguments=_arguments(fn.get("arguments")),
        ))
    return ChatTurn(text=_without_thinking(_content(message.get("content"))), tool_calls=calls)


def list_models(spec: ProviderSpec, key: str) -> list[ModelInfo]:
    data = send(spec, key, "GET", spec.models_path)
    entries = data.get("data") if isinstance(data.get("data"), list) else data.get("models") or []
    models = [m for m in (spec.model_filter(e) for e in entries if isinstance(e, dict)) if m]
    return sorted(models, key=lambda m: m.name.lower())


def check_key(spec: ProviderSpec, key: str) -> str:
    return check_key_generic(spec, key, list_models)


def check_key_generic(spec: ProviderSpec, key: str, lister) -> str:
    """Shared by every adapter: the provider's own key check where it has one,
    otherwise listing the models, which needs a working key everywhere."""
    if spec.key_check_path:
        body = send(spec, key, "GET", spec.key_check_path)
        if not spec.key_check_ok(body):
            raise LLMError("invalid_key", f"{spec.label} says this key is blocked or disabled.")
        return f"{spec.label} accepted the key."
    count = len(lister(spec, key))
    return f"{spec.label} accepted the key: {count} model(s) available."


# ---- shapes -------------------------------------------------------------


def _message(m: dict) -> dict:
    role = m.get("role")
    if role == "assistant":
        out: dict = {"role": "assistant", "content": m.get("content") or None}
        if m.get("tool_calls"):
            out["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
                for c in m["tool_calls"]
            ]
        return out
    if role == "tool":
        return {"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": str(m.get("content") or "")}
    return {"role": role, "content": str(m.get("content") or "")}


def _choice(spec: ProviderSpec, data: dict) -> dict:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise LLMError("bad_response", f"{spec.label} sent back an answer Clips Kitty couldn't read.")
    return choices[0]


# A reasoning model served without a reasoning parser (Nemotron on some vLLM
# hosts) writes its thinking into the answer: "<think>...</think>{...}", or,
# when the chat template opened the tag itself, "...</think>{...}". The
# parsers take the first "{" to the last "}", so a brace in the thinking
# would lose the whole answer. Thinking that the host sends apart, in
# "reasoning_content", is never read.
_THOUGHT = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def _without_thinking(text: str) -> str:
    text = _THOUGHT.sub("", text, count=1)
    if "</think>" in text and not text.lstrip().startswith(("{", "[")):
        # Thinking whose opening tag the model's template already sent, so
        # only "</think>" arrives, then the JSON. Anything else keeps its
        # "</think>": an answer that starts as JSON, where it is quoted text (a
        # hook, a chat line), or a reply in words that mentions the tag.
        tail = text.partition("</think>")[2]
        if tail.lstrip().startswith(("{", "[", "```")):
            text = tail
    if text.lstrip().startswith("<think>"):
        return ""  # still thinking when it stopped: no answer, which _text says
    return text


def _text(spec: ProviderSpec, data: dict) -> str:
    choice = _choice(spec, data)
    text = _without_thinking(_content((choice.get("message") or {}).get("content")))
    if not text.strip():
        if choice.get("finish_reason") == "length":
            raise LLMError("bad_response", f"The {spec.label} model ran out of room before answering. "
                                           "A model with a larger output limit will do better.")
        raise LLMError("bad_response", f"{spec.label} sent back an empty answer.")
    return text


def _content(content) -> str:
    if isinstance(content, list):  # some providers answer in parts
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}
