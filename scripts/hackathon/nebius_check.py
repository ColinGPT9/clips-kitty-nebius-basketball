"""One real NVIDIA Nemotron call through Nebius Token Factory, the app's own way.

Hackathon tooling (docs/hackathon/JUDGE_SETUP.md). It uses the same provider
entry, adapter and parser a Basketball job uses, on your own key:

  1. lists the models your key can call (GET /v1/models?verbose=true), with
     the NVIDIA ones and their prices first;
  2. asks the chosen model to pick the moments of a short piece of real
     basketball commentary, held to the app's clip schema, exactly as
     analysis/highlights.py asks during a job, and prints the clips it found,
     how long it took and the tokens Nebius reported;
  3. with --probe, asks the same question in each response format the adapter
     can fall back to (strict JSON schema, JSON object, plain) and says, for
     each, whether the answer parsed and where the model's thinking came back
     (apart, in the answer, or not at all).

  NEBIUS_API_KEY=... python scripts/hackathon/nebius_check.py
  NEBIUS_API_KEY=... python scripts/hackathon/nebius_check.py --model nvidia/Nemotron-3-Ultra-550b-a55b --probe

The key is read from NEBIUS_API_KEY, or from the app's own key store when you
pass --data-dir (the folder Settings → AI saved it in). It is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"

# The last half minute of the public-domain 1995 Belmont vs Salem game
# (archive.org/details/basketball95), as Whisper wrote it: the shape of every
# chunk a Basketball job sends.
SAMPLE = """[1290.0 - 1296.5] They have three seconds left, they just can't turn it over and get a shot off.
[1296.5 - 1303.0] Plenty of time to get a shot off, as Matt sets up at the free throw line.
[1303.0 - 1309.0] Costello on the weak side of the floor, where he is deadly.
[1309.0 - 1314.5] Here it is, plain and simple. The quarterback gets it in bounds.
[1314.5 - 1320.0] And it's not going to happen! They didn't get a shot off!
[1320.0 - 1327.0] And Salem charges the court! Salem wins it, sixty-one to fifty-nine!
[1327.0 - 1334.0] Belmont played a great game, but couldn't get it done here in the second half."""


def _key(args) -> tuple[Path, str]:
    from llm.providers import keys

    data_dir = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="ck-nebius-"))
    env_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if env_key:
        keys.save_key(data_dir, "nebius", env_key)
    key = keys.load_key(data_dir, "nebius")
    if not key:
        sys.exit("No Nebius key: set NEBIUS_API_KEY, or pass --data-dir with a key saved in Settings → AI.")
    return data_dir, key


def _prompt() -> str:
    """The prompt find_highlights builds for one chunk of a Basketball job."""
    import sports
    from analysis.highlights import _events_block, _guidance_block

    config = {"clips": {"sport": {"name": "basketball", "highlights": "best", "period": "full"}}}
    guidance = sports.profile_for(config).guidance("clips")
    first, last = float(SAMPLE.split("]")[0].strip("[").split(" - ")[0]), float(SAMPLE.split("[")[-1].split(" - ")[1].split("]")[0])
    prompt = (ROOT / "config" / "prompts" / "score_clips.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("{transcript}", SAMPLE)
    prompt = prompt.replace("{mode_guidance}", _guidance_block(guidance))
    prompt = prompt.replace("{events}", _events_block(EVENTS, first, last, "GAME / CHAT / AUDIO EVENTS"))
    prompt = prompt.replace("{min_duration}", "10").replace("{max_duration}", "60")
    return prompt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", help="the app's data folder, to use the key saved in Settings → AI")
    parser.add_argument("--probe", action="store_true", help="try every response format the adapter can use")
    args = parser.parse_args(argv)

    from analysis.highlights import CLIPS_SCHEMA, _parse_clips_json
    from llm.base import generate_json
    from llm.providers import usage
    from llm.providers.adapters import chat_completions
    from llm.providers.base import LLMError
    from llm.providers.catalog import PROVIDERS
    from llm.providers.cloud_backend import CloudBackend
    from llm.providers.http import send

    spec = PROVIDERS["nebius"]
    data_dir, key = _key(args)

    print(f"1. Models on your key ({spec.base_url}{spec.models_path})")
    try:
        models = chat_completions.list_models(spec, key)
    except LLMError as e:
        print(f"   {e.kind}: {e.message}")
        return 1
    nvidia = [m for m in models if m.vendor == "nvidia"]
    for m in nvidia + [m for m in models if m.vendor != "nvidia"]:
        price = f"${m.price[0]:g} / ${m.price[1]:g} per 1M tokens" if m.price else "price not listed"
        mark = "*" if m.id == args.model else " "
        print(f"  {mark} {m.id:45} {m.context:>9,} ctx  {price}")
    if args.model not in {m.id for m in models}:
        print(f"   {args.model} is not offered to this key; pass --model with one listed above.")
        return 1

    print(f"\n2. A real basketball request to {args.model}, as a Basketball job makes it")
    usage.reset()
    backend = CloudBackend(spec, args.model, data_dir)
    prompt = _prompt()
    started = time.monotonic()
    try:
        raw = generate_json(backend, prompt, CLIPS_SCHEMA)
    except LLMError as e:
        print(f"   {e.kind}: {e.message}")
        return 1
    seconds = time.monotonic() - started
    clips = _parse_clips_json(raw)
    print(f"   {seconds:.1f}s; usage reported by Nebius: {usage.summary(usage.totals()) or 'none'}")
    if clips is None:
        print(f"   The answer had no readable clip list. It began: {raw[:300]!r}")
    for c in clips or []:
        print(f"   [{c.score:3d}] {c.start:7.1f}s - {c.end:7.1f}s  {c.hook}  ({c.reason})")

    if args.probe:
        print("\n3. Every response format the adapter can fall back to")
        formats = chat_completions._formats(True, CLIPS_SCHEMA)
        for fmt in formats:
            label = fmt["type"] if fmt else "plain"
            body = {"model": args.model, "max_tokens": chat_completions.GENERATE_MAX_TOKENS,
                    "messages": [{"role": "user", "content": prompt}]}
            if fmt:
                body["response_format"] = fmt
            started = time.monotonic()
            try:
                data = send(spec, key, "POST", "/chat/completions", json_body=spec.body_extras(body))
            except LLMError as e:
                print(f"   {label:12} {e.kind}: {e.message}")
                continue
            seconds = time.monotonic() - started
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = chat_completions._content(message.get("content"))
            apart = bool(message.get("reasoning_content") or message.get("reasoning"))
            inline = "<think>" in content or "</think>" in content
            found = _parse_clips_json(chat_completions._without_thinking(content))
            u = data.get("usage") or {}
            reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
            print(f"   {label:12} {seconds:5.1f}s  finish={choice.get('finish_reason')}  "
                  f"parsed={'yes, %d clip(s)' % len(found) if found is not None else 'no'}  "
                  f"thinking={'apart' if apart else 'in the answer' if inline else 'none seen'}  "
                  f"tokens in/out={u.get('prompt_tokens')}/{u.get('completion_tokens')}"
                  + (f" ({reasoning} reasoning)" if reasoning else ""))
        print("\n   " + json.dumps({"service_tier": data.get("service_tier"), "model": data.get("model")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
