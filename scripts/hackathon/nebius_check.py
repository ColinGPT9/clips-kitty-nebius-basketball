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
import atexit
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"

# The final possession of the public-domain 1995 Belmont vs Salem game
# (archive.org/details/basketball95), exactly as Whisper wrote it in a Basketball
# job here, 1222.9 s to 1290.1 s into the game's last 22 minutes.
SEGMENTS = [
    (1222.9, 1223.9, "They got three seconds left."),
    (1224.6, 1225.6, "They just can't turn it over."),
    (1225.7, 1226.4, "Get a shot off."),
    (1227.3, 1229.0, "Plenty of time to get a shot off."),
    (1231.0, 1233.2, "And there's Matt right to the free throw line."),
    (1233.6, 1236.0, "Costello left side of the floor where he is deadly."),
    (1237.3, 1239.2, "Doug Conley, quarterback taking out a bounce"),
    (1241.2, 1242.7, "Okay, Marauders now"),
    (1243.2, 1248.9, "There it is, plain and simple, Dougie's got to toss it up, avoid, avoid the sign"),
    (1250.3, 1252.2, "Adam, you got Lapierre here in the front court"),
    (1253.0, 1255.6, "He's just out of the way, I'll tell you right now"),
    (1256.6, 1259.5, "Costello, he's going to catch it. The ball is going to Christensen"),
    (1260.1, 1264.6, "Okay, here we go, the quarterback gets it in, bounce, and it's not going to do it"),
    (1264.6, 1272.8, "Salem wins it. So the Salem wins the ball game. Marauders didn't get a shot off and "
                     "they're upset"),
    (1275.1, 1279.8, "So and they go off and they charge the court Marauders played a great game"),
    (1280.5, 1284.3, "But couldn't get it done here in the second half Salem an excellent ballclub"),
    (1284.3, 1290.1, "Scoot Salem is an excellent ballclub. You got to give it to him Belmont had a "
                     "fantastic season"),
]
# The same job's game sounds (PANNs) had nothing in this stretch: its last
# "a crowd cheering" was at 1209 s. So this chunk's prompt has no events block.
EVENTS: list[tuple[float, str]] = []


def _key(args) -> tuple[Path, str]:
    from llm.providers import keys

    env_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if env_key:
        # The app reads keys from its own store, so the key from the
        # environment goes into a private folder of its own, removed on exit.
        data_dir = Path(tempfile.mkdtemp(prefix="ck-nebius-"))
        atexit.register(shutil.rmtree, data_dir, True)
        keys.save_key(data_dir, "nebius", env_key)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        sys.exit("No Nebius key: set NEBIUS_API_KEY, or pass --data-dir with a key saved in Settings → AI.")
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
    transcript = "\n".join(f"[{start:.1f} - {end:.1f}] {text}" for start, end, text in SEGMENTS)
    prompt = (ROOT / "config" / "prompts" / "score_clips.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("{transcript}", transcript)
    prompt = prompt.replace("{mode_guidance}", _guidance_block(guidance))
    prompt = prompt.replace("{events}", _events_block(EVENTS, SEGMENTS[0][0], SEGMENTS[-1][1],
                                                      "GAME / CHAT / AUDIO EVENTS"))
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
                  f"parsed={f'yes, {len(found)} clip(s)' if found is not None else 'no'}  "
                  f"thinking={'apart' if apart else 'in the answer' if inline else 'none seen'}  "
                  f"tokens in/out={u.get('prompt_tokens')}/{u.get('completion_tokens')}"
                  + (f" ({reasoning} reasoning)" if reasoning else ""))
        print("\n   " + json.dumps({"service_tier": data.get("service_tier"), "model": data.get("model")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
