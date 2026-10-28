"""Nebius Token Factory: open models, NVIDIA Nemotron among them, on the
user's own Nebius key.

It speaks OpenAI-compatible chat completions (adapters/chat_completions.py),
so this is a ProviderSpec and a model filter, nothing more. Checked against
docs.tokenfactory.nebius.com in October 2026:

- One global address, https://api.tokenfactory.nebius.com/v1. The regional
  ones (api.tokenfactory.us-central1.nebius.com) can stop working when a
  model moves region, Nebius says, so they aren't used.
- `GET /models?verbose=true` lists every model the key can call, with its
  context and its price in USD per token, the way OpenRouter lists them.
- Models come and go (Nebius retires some every few months, and doesn't
  reroute their requests), so none is written here as a given: the list is
  the live one, and `preferred` only puts one first when the list has it.
"""

from llm.providers.base import ModelInfo, ProviderSpec

SITE = "https://tokenfactory.nebius.com"

# What can't pick clips: embedding, rerank and guard models, and image or
# video generators. Image2text models (Nemotron 3.5 Super VL) still answer in
# text, and the app never sends a cloud model the picture, so they stay.
_NOT_TEXT = ("embedding", "rerank", "guard", "flux", "stable-diffusion", "sdxl", "whisper")


def model_info(entry: dict) -> ModelInfo | None:
    model_id = str(entry.get("id") or "")
    lowered = model_id.lower()
    if not model_id or any(word in lowered for word in _NOT_TEXT):
        return None
    modality = str((entry.get("architecture") or {}).get("modality") or "")
    if "->" in modality and "text" not in modality.split("->", 1)[1]:
        return None
    pricing = entry.get("pricing") or {}
    vendor = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    return ModelInfo(
        id=model_id,
        name=str(entry.get("name") or model_id),
        context=_int(entry.get("context_length")),
        json_schema=True,  # response_format json_schema and json_object, per the docs
        tools=True,
        vendor=vendor,
        price=_price(pricing),
        pricing=_lines(pricing),
    )


# Nebius' price fields, as it names them (strings, USD per token), and how each is shown.
_FIELDS = (
    ("prompt", "Input", 1_000_000, "per 1M tokens"),
    ("input_cache_read", "Cached input", 1_000_000, "per 1M tokens"),
    ("completion", "Output", 1_000_000, "per 1M tokens"),
    ("request", "Per request", 1, "per request"),
)


def _amount(pricing: dict, field: str) -> float | None:
    try:
        value = float(pricing[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _price(pricing: dict) -> tuple[float, float] | None:
    prompt, completion = _amount(pricing, "prompt"), _amount(pricing, "completion")
    if prompt is None or completion is None:
        return None
    return round(prompt * 1_000_000, 4), round(completion * 1_000_000, 4)


def _lines(pricing: dict) -> tuple[tuple[str, float, str, bool], ...]:
    lines = []
    for field, label, scale, unit in _FIELDS:
        value = _amount(pricing, field)
        if value:
            lines.append((label, round(value * scale, 6), unit, False))
    return tuple(lines)


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


SPEC = ProviderSpec(
    id="nebius",
    label="Nebius Token Factory",
    adapter="chat_completions",
    base_url="https://api.tokenfactory.nebius.com/v1",
    key_label="Nebius Token Factory API key",
    key_url=f"{SITE}/project/api-keys",
    pricing_url="https://nebius.com/token-factory/prices",
    privacy="Transcripts and prompts are sent to Nebius Token Factory with your key.",
    models_path="/models?verbose=true",
    model_filter=model_info,
    tagline="Open models on Nebius, including NVIDIA Nemotron.",
    # Nemotron 3.5 Lightning: NVIDIA's open model, the cheapest Nemotron Nebius
    # lists ($0.06 / $0.24 per million tokens, October 2026) and among its
    # fastest. Every other NVIDIA model follows it in the list.
    preferred={"text": "nvidia/Nemotron-3_5-Lightning", "text_family": "nvidia/"},
)
