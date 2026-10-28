"""Nebius Token Factory: NVIDIA Nemotron (and Nebius' other open models) on
the user's own Nebius key, under Settings → AI → Direct provider API.

It speaks OpenAI-compatible chat completions, so it is a catalogue entry and
a model filter. These check what is particular to it, against the shapes in
docs.tokenfactory.nebius.com: the verbose model list, its error bodies, a
reasoning model's thinking in the answer, and the usage every answer names.
No test touches the network.
"""

from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from analysis.highlights import CLIPS_SCHEMA, _parse_clips_json
from core.state import StateDB
from llm.base import generate_json
from llm.providers import http, keys, usage
from llm.providers.base import LLMError
from llm.providers.catalog import PROVIDERS
from llm.providers.cloud_backend import CloudBackend
from llm.registry import create_backend
from llm.spec import parse_spec
from server import ai_api

KEY = "v1." + "0a1b2c3d" * 6
HOST = "api.tokenfactory.nebius.com"
NEMOTRON = "nvidia/Nemotron-3_5-Lightning"

# GET /v1/models?verbose=true, as the API reference shows it (RichModel).
MODELS = {"object": "list", "data": [
    {"id": NEMOTRON, "name": "Nemotron 3.5 Lightning", "context_length": 1048576,
     "architecture": {"modality": "text->text", "tokenizer": "Other"},
     "pricing": {"prompt": "0.00000006", "input_cache_read": "0.00000003", "completion": "0.00000024",
                 "image": "0", "request": "0"}, "object": "model", "owned_by": "system"},
    {"id": "nvidia/Nemotron-3-Ultra-550b-a55b", "name": "Nemotron 3 Ultra", "context_length": 1048576,
     "architecture": {"modality": "text->text"},
     "pricing": {"prompt": "0.000001", "completion": "0.000003"}},
    {"id": "nvidia/Nemotron-3.5-Super-VL", "name": "Nemotron 3.5 Super VL", "context_length": 1048576,
     "architecture": {"modality": "text+image->text"},
     "pricing": {"prompt": "0.0000003", "completion": "0.0000009"}},
    {"id": "google/gemma-3-27b-it", "name": "Gemma-3-27b-it", "context_length": 110000,
     "architecture": {"modality": "text+image->text"},
     "pricing": {"prompt": "0.0000001", "completion": "0.0000003"}},
    {"id": "Qwen/Qwen3-Embedding-8B", "name": "Qwen3-Embedding-8B",
     "architecture": {"modality": "text->embedding"}, "pricing": {"prompt": "0.00000001"}},
    {"id": "black-forest-labs/flux-dev", "architecture": {"modality": "text->image"}},
]}

CLIPS = '{"clips": [{"start": 12, "end": 41, "score": 88, "engagement": 80, "trending": false, ' \
        '"hook": "Costello from deep", "reason": "go-ahead three, crowd erupts"}]}'


def answer(content, usage_row=None, **message):
    body = {"id": "chatcmpl-1", "object": "chat.completion", "model": NEMOTRON, "service_tier": "default",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **message},
                         "finish_reason": "stop"}]}
    body["usage"] = usage_row if usage_row is not None else {
        "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens_details": {"reasoning_tokens": 240}}
    return body


class Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class Env:
    def __init__(self, tmp_path, monkeypatch):
        self.data = tmp_path / "data"
        self.data.mkdir()
        self.settings = tmp_path / "settings.yaml"
        self.settings.write_text("model: gemma:7b\n", encoding="utf-8")
        self.config = {"llm": {"backend": "ollama/gemma:7b", "data_dir": str(self.data),
                               "ollama_host": "http://127.0.0.1:1"}}
        self.requests: list[dict] = []
        self.replies: list[Response] = []  # chat replies, in order; then a plain answer
        self.waits: list[float] = []
        monkeypatch.setattr(http, "transport", self.transport)
        monkeypatch.setattr(http, "sleep", self.waits.append)
        usage.reset()
        app = FastAPI()
        ai_api.install(app, config=self.config, db=lambda: StateDB(tmp_path / "state.db"),
                       data_dir=self.data, settings_path=self.settings)
        self.client = TestClient(app, base_url="http://127.0.0.1")

    def transport(self, method, url, **kw):
        self.requests.append({"method": method, "url": url, **kw})
        assert urlsplit(url).hostname == HOST
        if kw["headers"].get("Authorization") != f"Bearer {KEY}":
            # /v1/models answers a bad key with only "detail" (a live probe, October 2026).
            return Response(401, {"detail": "Couldn't authenticate. Reason: Unable authenticate"})
        if urlsplit(url).path == "/v1/models":
            return Response(200, MODELS)
        if urlsplit(url).path == "/v1/chat/completions":
            return self.replies.pop(0) if self.replies else Response(200, answer(CLIPS))
        raise AssertionError(url)

    def save_key(self):
        saved = self.client.put("/ai/providers/nebius/key", json={"api_key": KEY})
        assert saved.status_code == 200, saved.text
        return saved

    def backend(self, model=NEMOTRON):
        return CloudBackend(PROVIDERS["nebius"], model, self.data)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def test_nebius_is_a_direct_provider_beside_the_others():
    spec = PROVIDERS["nebius"]
    assert (spec.tier, spec.adapter, spec.auth) == (3, "chat_completions", "bearer")
    assert spec.base_url == "https://api.tokenfactory.nebius.com/v1"
    assert spec.key_url == "https://tokenfactory.nebius.com/project/api-keys"
    assert spec.pricing_url.startswith("https://nebius.com/")
    assert spec.tagline and "Nemotron" in spec.tagline
    assert spec.public()["recommended"] is False and spec.stt == {}
    assert list(PROVIDERS)[:1] == ["openrouter"]  # OpenRouter stays the recommended cloud path


def test_a_nebius_spec_names_the_provider_and_keeps_the_model_whole(env):
    assert parse_spec(f"nebius/{NEMOTRON}") == ("nebius", NEMOTRON)
    backend = create_backend({"backend": f"nebius/{NEMOTRON}", "data_dir": str(env.data)})
    assert isinstance(backend, CloudBackend)
    assert backend.name == f"nebius/{NEMOTRON}" and backend.model == NEMOTRON
    assert backend.sees_images() is False  # the picture never goes to a cloud model


def test_without_a_key_it_says_where_to_add_one(env):
    with pytest.raises(LLMError) as e:
        env.backend().generate("pick clips", json_mode=True)
    assert e.value.kind == "not_configured" and "Settings → AI" in e.value.message
    assert env.requests == []


def test_the_key_is_checked_listing_models_and_never_shown(env):
    saved = env.save_key()
    assert env.requests[0]["url"] == "https://api.tokenfactory.nebius.com/v1/models?verbose=true"
    row = next(p for p in saved.json()["providers"] if p["id"] == "nebius")
    assert row["has_key"] is True and KEY not in saved.text
    assert keys.load_key(env.data, "nebius") == KEY


def test_a_key_nebius_refuses_is_not_kept(env):
    refused = env.client.put("/ai/providers/nebius/key", json={"api_key": "v1.wrong-key-123456789"})
    assert refused.status_code == 400 and "didn't accept the API key" in refused.json()["detail"]
    assert not keys.has_key(env.data, "nebius")


def test_only_models_that_answer_in_text_are_offered_with_their_price(env):
    env.save_key()
    models = env.client.get("/ai/providers/nebius/models").json()["models"]
    ids = [m["id"] for m in models]
    assert "Qwen/Qwen3-Embedding-8B" not in ids and "black-forest-labs/flux-dev" not in ids
    assert set(ids) == {NEMOTRON, "nvidia/Nemotron-3-Ultra-550b-a55b", "nvidia/Nemotron-3.5-Super-VL",
                        "google/gemma-3-27b-it"}
    lightning = next(m for m in models if m["id"] == NEMOTRON)
    assert lightning["context"] == 1048576 and lightning["vendor"] == "nvidia"
    assert lightning["price"] == {"input": 0.06, "output": 0.24}
    assert [line["label"] for line in lightning["pricing"]] == ["Input", "Cached input", "Output"]


def test_the_preferred_nemotron_survives_the_filter_and_says_why_it_is_preferred(env):
    # Whether today's Nebius list still has it is checked live, by
    # scripts/hackathon/nebius_check.py; here, that the filter keeps it.
    preferred = PROVIDERS["nebius"].preferred
    assert preferred["text"] == "nvidia/Nemotron-3_5-Lightning" and preferred["text_family"] == "nvidia/"
    assert "Nemotron" in preferred["text_why"] and preferred["text_family_label"] == "NVIDIA Nemotron"
    env.save_key()
    ids = [m["id"] for m in env.client.get("/ai/providers/nebius/models").json()["models"]]
    assert preferred["text"] in ids


def test_test_connection_checks_the_chosen_model_without_spending_tokens(env):
    env.save_key()
    ok = env.client.post("/ai/providers/nebius/test", json={"model": NEMOTRON}).json()
    gone = env.client.post("/ai/providers/nebius/test",
                           json={"model": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"}).json()
    assert ok["ok"] is True and NEMOTRON in ok["message"]
    assert gone == {"ok": False, "kind": "model_unavailable",
                    "message": "Nebius Token Factory doesn't list nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 "
                               "for your key."}
    assert not any(r["url"].endswith("/chat/completions") for r in env.requests)


def test_choosing_nemotron_makes_it_the_model_every_job_uses(env):
    refused = env.client.post("/ai/activate", json={"provider": "nebius", "model": NEMOTRON})
    assert refused.status_code == 400 and "Nebius Token Factory API key" in refused.json()["detail"]
    env.save_key()
    chosen = env.client.post("/ai/activate", json={"provider": "nebius", "model": NEMOTRON}).json()
    assert chosen["active"] == {"provider": "nebius", "model": NEMOTRON, "local": False}
    assert f"model: nebius/{NEMOTRON}" in env.settings.read_text(encoding="utf-8")
    assert env.config["llm"]["backend"] == f"nebius/{NEMOTRON}"


def test_a_clip_request_is_held_to_the_schema_and_billed_usage_is_counted(env):
    env.save_key()
    raw = generate_json(env.backend(), "pick the basketball moments", CLIPS_SCHEMA)
    sent = env.requests[-1]
    assert sent["url"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert sent["json"]["model"] == NEMOTRON
    assert sent["json"]["response_format"]["type"] == "json_schema"
    assert sent["json"]["response_format"]["json_schema"]["schema"] == CLIPS_SCHEMA
    assert KEY not in str(sent["json"])
    assert [c.hook for c in _parse_clips_json(raw)] == ["Costello from deep"]
    assert usage.totals() == {f"nebius/{NEMOTRON}": {
        "calls": 1, "prompt_tokens": 1200, "completion_tokens": 300, "cached_tokens": 200,
        "reasoning_tokens": 240}}
    assert usage.summary(usage.totals()) == (f"nebius/{NEMOTRON}: 1 call(s), 1,200 tokens in, 300 out "
                                             "(240 reasoning)")


@pytest.mark.parametrize("content", [
    '<think>The crowd noise at {12s} suggests a three.</think>\n' + CLIPS,
    'Costello is open, the score goes {58-57}.</think>' + CLIPS,  # the template opened <think> itself
])
def test_thinking_written_into_the_answer_is_left_out(env, content):
    env.save_key()
    env.replies = [Response(200, answer(content))]
    raw = env.backend().generate("pick clips", json_mode=True, schema=CLIPS_SCHEMA)
    assert raw == CLIPS
    assert len(_parse_clips_json(raw)) == 1


@pytest.mark.parametrize("content", [
    '{"clips": [{"start": 1, "end": 20, "score": 80, "hook": "chat spams </think> lol", "reason": "r"}]}',
    'In this model\'s output, </think> closes the thinking.',
])
def test_an_answer_that_only_mentions_the_closing_tag_is_kept_whole(env, content):
    env.save_key()
    env.replies = [Response(200, answer(content))]
    assert env.backend().generate("pick clips") == content


def test_usage_a_provider_reports_oddly_never_fails_the_answer(env):
    env.save_key()
    env.replies = [Response(200, answer(CLIPS, usage_row={"prompt_tokens": float("inf"), "completion_tokens": None}))]
    assert env.backend().generate("pick clips") == CLIPS
    assert usage.totals()[f"nebius/{NEMOTRON}"]["prompt_tokens"] == 0


def test_thinking_sent_apart_is_never_read(env):
    env.save_key()
    env.replies = [Response(200, answer(CLIPS, reasoning_content="{not the answer}"))]
    assert env.backend().generate("pick clips", json_mode=True) == CLIPS


def test_an_answer_cut_off_mid_thought_says_it_ran_out_of_room(env):
    env.save_key()
    cut = answer("<think>Still weighing the block at 9:41 against the")
    cut["choices"][0]["finish_reason"] = "length"
    env.replies = [Response(200, cut)]
    with pytest.raises(LLMError) as e:
        env.backend().generate("pick clips")
    assert e.value.kind == "bad_response" and "ran out of room" in e.value.message


def test_a_rate_limit_waits_as_long_as_nebius_says_then_goes_on(env):
    env.save_key()
    limited = {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded",
                         "message": "Rate limit exceeded. Retry the request after 30 sec"}}
    env.replies = [Response(429, limited, {"Retry-After": "30"}), Response(200, answer(CLIPS))]
    assert env.backend().generate("pick clips", json_mode=True) == CLIPS
    assert env.waits == [30.0]


@pytest.mark.parametrize("status, body, kind", [
    (402, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted",
                     "message": "Payment Required: You have exhausted your budget"}}, "no_credits"),
    (401, {"error": {"type": "invalid_request_error", "code": "unauthenticated",
                     "message": "Couldn't authenticate. Reason: Unable authenticate"}}, "invalid_key"),
    (403, {"error": {"type": "invalid_request_error", "code": "unauthorized", "message": "Forbidden"}},
     "rejected"),
    (429, {"error": {"type": "rate_limit_error", "code": "slow_down",
                     "message": "Your request rate increased too quickly"}}, "rate_limited"),
    (503, {"error": {"type": "service_unavailable_error", "message": "No available endpoints"}},
     "provider_down"),
    (529, {"error": {"type": "server_error", "message": "System is overloaded"}}, "provider_down"),
])
def test_nebius_errors_come_back_in_words_a_creator_can_act_on(env, status, body, kind):
    env.save_key()
    env.replies = [Response(status, body)] * 3
    with pytest.raises(LLMError) as e:
        env.backend().generate("pick clips")
    assert e.value.kind == kind and KEY not in e.value.message
    if kind == "no_credits":
        assert len(env.waits) == 0  # out of credit is never asked again


def test_a_retired_model_says_to_pick_another(env):
    # Nebius retires models on a schedule and doesn't reroute them (November 2026 notice).
    env.save_key()
    gone = {"error": {"type": "invalid_request_error", "code": "model_not_found", "param": "model",
                      "message": "The model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` does not exist."}}
    env.replies = [Response(404, gone)] * 3
    with pytest.raises(LLMError) as e:
        env.backend("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B").generate("pick clips")
    assert e.value.kind == "model_unavailable" and "Pick another one in Settings → AI" in e.value.message
