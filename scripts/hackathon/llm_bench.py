"""Gemma against Nemotron on the same basketball game, with only the LLM changed.

Hackathon tooling (docs/hackathon/BENCHMARKS.md). Nothing in the app imports
this file, and it changes no setting.

The basketball pipeline reads the game once (Whisper, PANNs, the score bug's
OCR, YOLO and the cutaway finder) and only then asks the language model about
it, inside analysis.fusion.find_clips. So the benchmark freezes everything
that exists at that moment and replays the LLM-dependent stages, unchanged,
with each backend:

  serve    Run the real app (`main.py serve`), saving the inputs of every
           find_clips call to --freeze DIR before the call runs. The job itself
           goes on exactly as usual: clips are scored, titled and rendered.
  replay   Load one frozen file and run, with the backend you name:
           find_clips (chunk scoring, window scoring, rerank) -> the sport's
           own window refinement (YOLO + OCR, deterministic) -> titles and
           descriptions -> the sport's title check. Every LLM call is timed,
           its JSON checked, and its token usage read where the provider
           reports it.
  compare  Tables of two or more replays, for BENCHMARKS.md.

The frozen inputs are the transcript, the audio and visual curves, the game's
sound curves and the BasketballProfile with its score-bug readings, shots and
cutaways. Replays never run Whisper or the read of the game again, so every
backend sees the same evidence.

  python scripts/hackathon/llm_bench.py serve --freeze bench/frozen -- --port 8765
  python scripts/hackathon/llm_bench.py replay bench/frozen/<video>.pkl \
      --backend ollama/gemma:7b --label gemma --out bench/runs
  NEBIUS_API_KEY=... python scripts/hackathon/llm_bench.py replay bench/frozen/<video>.pkl \
      --backend nebius/nvidia/Nemotron-3_5-Lightning --label nemotron --out bench/runs
  python scripts/hackathon/llm_bench.py compare bench/runs/gemma.json bench/runs/nemotron.json
"""

from __future__ import annotations

import argparse
import atexit
import copy
import inspect
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN_VERSION = 1


# ---- serve: the real app, with find_clips' inputs saved -------------------------


def _caller() -> tuple[dict, dict | None, bool]:
    """From process_video's frame, the one that called find_clips: its `video`,
    the job's own `config` (find_clips may get a copy with a larger shortlist
    for Rate plugins) and whether the job chose plugin steps."""
    frame = inspect.currentframe()
    while frame is not None:
        local = frame.f_locals
        video = local.get("video")
        if video is not None and hasattr(video, "video_id") and hasattr(video, "path"):
            info = {k: getattr(video, k, None) for k in ("video_id", "title", "channel", "duration", "path")}
            config = local.get("config") if isinstance(local.get("config"), dict) else None
            return info, config, bool(local.get("steps_chosen"))
        frame = frame.f_back
    return {}, None, False


def install_freeze(out_dir: Path) -> None:
    import core.pipeline as pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    original = pipeline.find_clips

    def find_clips(video_path, segments, llm, config, *args, **kwargs):
        video, job_config, steps_chosen = _caller()
        frozen = {
            "version": FROZEN_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "llm_at_capture": getattr(llm, "name", str(llm)),
            "video": {k: str(v) if isinstance(v, Path) else v for k, v in video.items()},
            "video_path": str(video_path),
            "segments": segments,
            "config": config,
            "job_config": job_config,
            "steps_chosen": steps_chosen,
            "args": args,
            "kwargs": kwargs,
        }
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video.get("video_id") or Path(video_path).stem))
        try:
            # Pickled BEFORE the call: find_clips fills the sport profile as it
            # goes, and the replay has to start from what it started from.
            blob = pickle.dumps(frozen, protocol=pickle.HIGHEST_PROTOCOL)
            path = out_dir / f"{name}.pkl"
            path.write_bytes(blob)
            print(f"      (llm_bench: froze find_clips inputs to {path}, {len(blob) / 1e6:.1f} MB)")
        except Exception as e:  # never stops the job
            print(f"      (llm_bench: could not freeze the inputs: {type(e).__name__}: {e})")
        return original(video_path, segments, llm, config, *args, **kwargs)

    pipeline.find_clips = find_clips


def cmd_serve(args) -> int:
    install_freeze(Path(args.freeze))
    import main as app_main

    sys.argv = ["main.py", "serve", *args.rest]
    return app_main.main()


# ---- replay: the LLM-dependent stages, with one backend --------------------------

_STAGES = {
    "find_highlights": "chunk scoring",
    "_score_window_batch": "window scoring",
    "_rerank": "rerank",
    "generate_metadata_batch": "titles",
    "_ask": "direction",
    "look": "vision",
}
_EXPECTED_KEY = {"chunk scoring": "clips", "window scoring": "clips", "rerank": "order", "titles": "items",
                 "title check": "items"}


def _stage() -> str:
    """Which pipeline step made this call, from the stack (titles called from
    the sport's title check are "title check")."""
    names = [f.function for f in inspect.stack()[2:30]]
    if "generate_metadata_batch" in names and any(n in ("check_titles", "check") for n in names):
        return "title check"
    for name in names:
        if name in _STAGES:
            return _STAGES[name]
    return "other"


_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def _json_ok(text: str, key: str | None) -> bool:
    """Whether the answer holds the JSON object the step reads, the way the
    app's own parsers find it (first "{" to last "}", fences allowed)."""
    text = _THINK.sub("", text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return False
    return isinstance(data, dict) and (key is None or key in data)


class Meter:
    """An LLMBackend that times and checks every call it passes on."""

    def __init__(self, inner):
        from llm.base import LLMBackend

        self.inner = inner
        self.calls: list[dict] = []
        self.supports_schema = getattr(inner, "supports_schema", False)
        # Registered as a virtual subclass so isinstance checks still pass.
        LLMBackend.register(type(self))

    @property
    def name(self) -> str:
        return self.inner.name

    def sees_images(self) -> bool:
        return self.inner.sees_images()

    def look(self, prompt, images):
        return self._timed("look", lambda: self.inner.look(prompt, images), prompt)

    def chat(self, messages, tools):
        return self.inner.chat(messages, tools)

    def generate(self, prompt: str, *, json_mode: bool = False, schema: dict | None = None) -> str:
        if self.supports_schema and schema is not None:
            return self._timed("generate", lambda: self.inner.generate(prompt, json_mode=json_mode, schema=schema),
                               prompt)
        return self._timed("generate", lambda: self.inner.generate(prompt, json_mode=json_mode), prompt)

    def _timed(self, kind: str, call, prompt: str):
        from llm.providers import usage

        stage = _stage()
        before = usage.totals()
        start = time.monotonic()
        entry: dict = {"stage": stage, "kind": kind, "prompt_chars": len(prompt)}
        try:
            text = call()
        except Exception as e:
            entry.update(seconds=round(time.monotonic() - start, 2), error=f"{type(e).__name__}: {e}"[:300],
                         json_ok=False)
            entry["usage"] = usage.diff(before, usage.totals())
            self.calls.append(entry)
            raise
        entry.update(seconds=round(time.monotonic() - start, 2), answer_chars=len(text or ""),
                     json_ok=_json_ok(text, _EXPECTED_KEY.get(stage)),
                     answer_head=(text or "")[:400])
        entry["usage"] = usage.diff(before, usage.totals())
        self.calls.append(entry)
        return text


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class _Video:
    def __init__(self, info: dict, path: str):
        self.video_id = info.get("video_id") or Path(path).stem
        self.title = info.get("title") or Path(path).stem
        self.channel = info.get("channel") or ""
        self.duration = info.get("duration") or 0.0
        self.path = Path(path)


def _backend(spec: str, config: dict, data_dir: Path):
    from llm.registry import create_backend

    llm_cfg = copy.deepcopy(config.get("llm") or {})
    llm_cfg["backend"] = spec
    llm_cfg["data_dir"] = str(data_dir)
    provider = spec.partition("/")[0]
    env_key = os.environ.get(f"{provider.upper()}_API_KEY", "").strip()
    if env_key and provider != "ollama":
        # The app reads keys from its own store: the key from the environment
        # goes into a private folder of its own, removed on exit.
        from llm.providers import keys

        key_dir = Path(tempfile.mkdtemp(prefix="ck-bench-key-"))
        atexit.register(shutil.rmtree, key_dir, True)
        keys.save_key(key_dir, provider, env_key)
        llm_cfg["data_dir"] = str(key_dir)
    return create_backend(llm_cfg)


def _title_origin(meta, first, fallback) -> str:
    """Whose title a clip ended up with: the model's first answer, the model's
    rewrite after the sport's check, the check's own from the scoreboard, or
    the app's hook-based fallback for an answer it couldn't read."""
    title = getattr(meta, "title", "")
    if not title:
        return "none"
    if title == getattr(fallback, "title", None):
        return "fallback"
    if first is None or title == getattr(first, "title", None):
        return "model"
    return "rewritten or scoreboard"


def cmd_replay(args) -> int:
    from analysis.fusion import find_clips
    from analysis.metadata import generate_metadata_batch
    from core.pipeline import _refine_windows
    from llm.providers import usage

    frozen = pickle.loads(Path(args.frozen).read_bytes())
    config = frozen["config"]                       # what find_clips got
    job_config = frozen.get("job_config") or config  # what the rest of process_video uses
    if frozen.get("steps_chosen"):
        print("Note: this job chose plugin steps (Understand, Rate). Their LLM calls are not replayed, and the "
              "titles are written for the whole shortlist find_clips returns.")
    kwargs = dict(frozen["kwargs"])
    sport = kwargs.get("sport")
    video_path = Path(args.video or frozen["video_path"])
    if not video_path.exists():
        print(f"Source video not found: {video_path} (pass --video)")
        return 2
    video = _Video(frozen.get("video") or {}, str(video_path))
    data_dir = Path(args.data_dir or config["llm"].get("data_dir") or tempfile.mkdtemp(prefix="ck-bench-"))
    llm = Meter(_backend(args.backend, config, data_dir))

    print(f"Replaying {video.title} ({len(frozen['segments'])} segments) with {llm.name}")
    usage.reset()
    started = time.monotonic()
    # A clip direction is parsed by the language model too, so it is parsed
    # again with this backend rather than reused from the capture.
    kwargs.pop("intent", None)
    from core.pipeline import clip_direction

    intent = clip_direction(job_config, llm, video.duration)
    if intent is not None:
        kwargs["intent"] = intent
    candidates, rejections = find_clips(video_path, frozen["segments"], llm, config, *frozen["args"], **kwargs)
    found_s = time.monotonic() - started
    candidates = _refine_windows(sport, candidates, video, frozen["segments"], job_config)
    refined_s = time.monotonic() - started - found_s
    metas, first = [], []
    if candidates:
        ctx = kwargs.get("creator_context")
        creator = ctx.summary if ctx else ""
        title_rules = getattr(sport, "title_rules", None)
        metas = generate_metadata_batch(
            candidates, frozen["segments"], video.title, llm, creator_context=creator,
            **({"rules": title_rules()} if title_rules is not None else {}))
        first = list(metas)
        check_titles = getattr(sport, "check_titles", None)
        if check_titles is not None:
            metas = check_titles(candidates, metas, lambda clips, rules: generate_metadata_batch(
                clips, frozen["segments"], video.title, llm, creator_context=creator, rules=rules))
    total_s = time.monotonic() - started

    from analysis.metadata import _fallback

    clips = []
    for i, (c, meta) in enumerate(zip(candidates, metas or [None] * len(candidates))):
        s = c.subscores or {}
        clips.append({
            "start": round(c.start, 2), "end": round(c.end, 2), "score": c.score,
            "hook": c.hook, "reason": c.reason, "source": c.source,
            "moment": _plain(s.get("sport_label") or s.get("sport_event") or s.get("game_why") or ""),
            "subscores": _plain(s),
            "title": getattr(meta, "title", ""), "description": getattr(meta, "description", ""),
            "hashtags": list(getattr(meta, "hashtags", []) or []),
            "title_origin": _title_origin(meta, first[i] if i < len(first) else None, _fallback(c, video.title)),
        })
    calls = llm.calls
    result = {
        "label": args.label,
        "backend": llm.name,
        "video": {"title": video.title, "video_id": video.video_id, "path": str(video_path)},
        "frozen": str(args.frozen),
        "frozen_saved_at": frozen.get("saved_at"),
        "seconds": {"find_clips": round(found_s, 1), "refine_windows": round(refined_s, 1),
                    "total": round(total_s, 1)},
        "clips": clips,
        "rejections": [{"start": round(r.candidate.start, 2), "end": round(r.candidate.end, 2),
                        "score": r.candidate.score, "reason": r.reason} for r in rejections],
        "calls": calls,
        "usage": usage.totals(),
        "sport_report": _plain(getattr(sport, "report_data", None)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.label}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for c in calls if c.get("json_ok"))
    print(f"{len(clips)} clip(s), {len(calls)} LLM call(s), {ok} with usable JSON, {total_s:.0f}s -> {path}")
    return 0


# ---- compare --------------------------------------------------------------------

PRICES_URL = "https://tokenfactory.nebius.com/api/public/models_info"


def _prices() -> dict:
    """USD per 1M tokens (input, cached input, output) from Nebius' public
    catalogue, which needs no key. {} when it can't be read."""
    try:
        import requests

        rows = requests.get(PRICES_URL, timeout=20).json()
    except Exception:
        return {}
    prices = {}
    for row in rows if isinstance(rows, list) else []:
        for flavor in row.get("flavors") or []:
            prices[flavor.get("model_id")] = (
                float(flavor.get("input_price_per_million_tokens") or 0),
                float(flavor.get("input_cached_price_per_million_tokens") or 0),
                float(flavor.get("output_price_per_million_tokens") or 0),
            )
    return prices


def _cost(run: dict, prices: dict) -> float | None:
    backend = run["backend"]
    provider, _, model = backend.partition("/")
    if provider != "nebius" or model not in prices:
        return None
    price_in, price_cached, price_out = prices[model]
    total = 0.0
    for row in (run.get("usage") or {}).values():
        cached = row.get("cached_tokens", 0)
        total += (row.get("prompt_tokens", 0) - cached) * price_in + cached * price_cached \
            + row.get("completion_tokens", 0) * price_out
    return total / 1e6


def _overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def _ts(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def cmd_compare(args) -> int:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs]
    prices = _prices()
    lines = ["| Run | Backend | Clips | LLM calls | Usable JSON | Errors | LLM time (s) | Total (s) "
             "| Tokens in / out | Cost (USD) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        calls = run["calls"]
        ok = sum(1 for c in calls if c.get("json_ok"))
        errors = sum(1 for c in calls if c.get("error"))
        llm_s = sum(c.get("seconds", 0) for c in calls)
        tok_in = sum(r.get("prompt_tokens", 0) for r in (run.get("usage") or {}).values())
        tok_out = sum(r.get("completion_tokens", 0) for r in (run.get("usage") or {}).values())
        cost = _cost(run, prices)
        tokens = f"{tok_in:,} / {tok_out:,}" if tok_in or tok_out else "not reported (local)"
        lines.append(f"| {run['label']} | `{run['backend']}` | {len(run['clips'])} | {len(calls)} | "
                     f"{ok}/{len(calls)} | {errors} | {llm_s:.0f} | {run['seconds']['total']:.0f} | {tokens} | "
                     f"{f'${cost:.4f}' if cost is not None else 'n/a'} |")
    lines += ["", "Per step (calls, usable JSON, mean seconds):", ""]
    stages = sorted({c["stage"] for run in runs for c in run["calls"]})
    lines.append("| Step | " + " | ".join(run["label"] for run in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for stage in stages:
        cells = []
        for run in runs:
            rows = [c for c in run["calls"] if c["stage"] == stage]
            if not rows:
                cells.append("-")
                continue
            ok = sum(1 for c in rows if c.get("json_ok"))
            mean = sum(c.get("seconds", 0) for c in rows) / len(rows)
            cells.append(f"{len(rows)} calls, {ok} ok, {mean:.1f}s")
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    if len(runs) >= 2:
        a, b = runs[0], runs[1]
        # One to one: the closest pairs first, and no clip counted twice.
        pairs = sorted(((_overlap(x, y), i, j) for i, x in enumerate(a["clips"]) for j, y in enumerate(b["clips"])),
                       reverse=True)
        used_a, used_b, same = set(), set(), []
        for iou, i, j in pairs:
            if iou >= 0.5 and i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                same.append((i, j))
        lines += ["", f"Clips both picked ({a['label']} vs {b['label']}, time overlap of at least half): "
                      f"{len(same)} of {len(a['clips'])} / {len(b['clips'])}.", ""]
    lines += ["", "Where each run's final titles came from:", "",
              "| Run | Model | Rewritten or from the scoreboard | App fallback |", "|---|---|---|---|"]
    for run in runs:
        origins = [c.get("title_origin") for c in run["clips"]]
        lines.append(f"| {run['label']} | {origins.count('model')} | {origins.count('rewritten or scoreboard')} | "
                     f"{origins.count('fallback')} |")
    for run in runs:
        lines += ["", f"#### {run['label']} (`{run['backend']}`)", "",
                  "| # | Window | Score | Source | Title |", "|---|---|---|---|---|"]
        for i, c in enumerate(sorted(run["clips"], key=lambda c: c["start"]), 1):
            title = (c.get("title") or c.get("hook") or "").replace("|", "/")
            lines.append(f"| {i} | {_ts(c['start'])}-{_ts(c['end'])} | {c['score']} | {c['source']} | {title} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="run the app, freezing find_clips' inputs")
    p.add_argument("--freeze", required=True, help="folder for the frozen inputs")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for `main.py serve`, after --")
    p = sub.add_parser("replay", help="replay the LLM stages of one frozen video with one backend")
    p.add_argument("frozen")
    p.add_argument("--backend", required=True, help="e.g. ollama/gemma:7b, nebius/nvidia/Nemotron-3_5-Lightning")
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="bench/runs")
    p.add_argument("--video", help="the source video, if it moved since it was frozen")
    p.add_argument("--data-dir", help="data folder holding the provider key (default: the frozen config's)")
    p = sub.add_parser("compare", help="tables of two or more replays")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "serve":
        args.rest = [a for a in args.rest if a != "--"]
        return cmd_serve(args)
    if args.command == "replay":
        return cmd_replay(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
