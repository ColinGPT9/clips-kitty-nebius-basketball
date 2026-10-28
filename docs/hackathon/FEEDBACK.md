# Feedback on Nebius Token Factory and NVIDIA Nemotron

This is the feedback the Devpost form asks for. Each point says where it came
from. Part 1 is what we found while reading the documentation and probing the
API before writing the provider, on 2026-10-28, with no key. Part 2 is what
the live runs showed. Nothing here is a guess. Where something was not
measured, it says so.

## Part 1: documentation and API

**1. The model-list error has a different shape from the chat error.**
Without a key, or with a bad one, `GET /v1/models` answers
`{"detail": "Couldn't authenticate. Reason: Unable authenticate"}`.
`POST /v1/chat/completions` answers
`{"error": {"type": "invalid_request_error", "code": "unauthenticated", …}, "detail": "…"}`.
A client that reads the OpenAI-style `error.message`, as most do, gets nothing
from the models endpoint. One shape for every endpoint would help. Clips Kitty
checks a key by listing the models, so its error comes from the status code alone.

**2. The `response_format` description contradicts its own schema.** The
OpenAPI spec's description of `response_format` says "Only {'type': 'json_object'}
or {'type': 'text'} is supported". The ResponseFormat schema next to it, and
the structured-output guide, both document `json_schema`. We built for
`json_schema` with a fallback to `json_object`. The "JSON mode" tag that the guide
says to look for on model cards doesn't appear in the `use_cases` of any model
in the public catalogue, Nemotron's included.

**3. Reasoning control for Nemotron 3 and 3.5 isn't documented on Nebius.**
The chat schema has `reasoning_effort` and returns `reasoning_content`. The
only Nemotron page that explains turning thinking on or off is the
cookbook's, and it covers the retired Llama-3.1-Nemotron-Ultra-253B
("detailed thinking on/off"). For a JSON-producing app the questions that
matter are these. Is thinking returned apart from `content` for every Nemotron
model? Does a strict `json_schema` constrain the thinking or only the answer?
Does `reasoning_effort` change Nemotron's behaviour? A short Nemotron-on-Nebius
page would answer all three. Our provider handles each case (thinking apart,
thinking inline in `<think>` tags, or none) because we could not tell which
one we would get.

**4. The model lifecycle moves fast for a production app.** Two of the five
NVIDIA models in today's catalogue (`NVIDIA-Nemotron-3-Nano-30B-A3B`,
`nemotron-3-super-120b-a12b`) are retired on 2026-11-09. That is twelve
days after we started, and the deprecation notice says requests are not
rerouted. That is why Clips Kitty never hard-codes a Nebius model. It lists
the live models and only *prefers* `nvidia/Nemotron-3_5-Lightning` when the
list has it. A `deprecation_date` or `replaced_by` field in
`/v1/models?verbose=true` would let an app warn its users before a model
disappears.

**5. "Fast" flavours are documented but not offered.** The inference
overview says to append `-fast` to a model name for the Fast flavour. The
public catalogue (`/api/public/models_info`) has no `-fast` model today, and
the June 2026 notice retired several. The overview should say so.

**6. Regional base URLs.** The cookbook's Nemotron Super and Ultra pages
recommend `api.tokenfactory.us-central1.nebius.com`. The serverless page warns
that a regional URL "can stop working if the endpoint's processing region
changes". We use the global `api.tokenfactory.nebius.com`. The cookbook
should match that advice.

**7. Things that worked well.**

- Being OpenAI-compatible meant Clips Kitty needed no new wire format.
  Nebius is one entry in its existing provider catalogue.
- `GET /v1/models?verbose=true` returns context length and price per token in
  the same shape OpenRouter uses, so the settings card could show price and
  context with no special code.
- The public catalogue at `/api/public/models_info` needs no key. The benchmark
  reads current prices from it to compute cost.
- The `x-ratelimit-*` headers and `Retry-After` are documented clearly, and
  so are the error codes (`credit_balance_exhausted`, `rate_limit_exceeded`,
  `slow_down`, `model_not_found`). Each maps onto a message Clips Kitty already
  shows creators.
- `usage` comes back on every response, with `cached_tokens` and
  `reasoning_tokens`.

## Part 2: live runs

These will be added from the first runs with a key, and only with what was
measured: `nebius_check.py --probe`, then the benchmark replays.

- Which of the three response formats each Nemotron model accepted, and
  whether the answer parsed: _not yet measured_.
- Where the thinking came back (apart, inline, or none): _not yet measured_.
- Latency and tokens per Basketball step, compared with Gemma: _not yet
  measured_. See [BENCHMARKS.md](BENCHMARKS.md).
- Rate limits hit on a full game, if any: _not yet measured_.
- Onboarding: account, $1 trial credit, card requirement, time to the first
  successful call: _to be noted by Colin when he creates the key_.

## NVIDIA tools used

- **NVIDIA Nemotron** through Nebius Token Factory: language-model reasoning
  in the Basketball pipeline (scoring commentary chunks and moment windows,
  reranking finalists, writing titles and descriptions).
- Nemotron 3 Nano (`nemotron-3-nano:4b`) through Ollama was already in the app,
  as a local option. Its earlier note in `KNOWN-ISSUES.md` is that it "works,
  but found far fewer moments than the others", and that it returned empty clip
  lists with reasoning off.
