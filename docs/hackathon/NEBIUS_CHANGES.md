# What changed for the Nebius × NVIDIA hackathon

This repository is Clips Kitty with one new language-model backend. It is a
copy of [ColinGPT9/clips-studio](https://github.com/ColinGPT9/clips-studio)
with its full history, starting at commit
`34f0c23f2423ab609c0341aa904698a5c73ba730` ("Thai: the 68 missing strings
(#185)", 2026-10-28). The original repository and its `main` branch were not
changed, and neither were its Microsoft Store releases or its website.

Every change since that commit is listed below. `git diff 34f0c23` shows all
of them.

## The one AI change: Nemotron through Nebius Token Factory

| File | What |
|---|---|
| `llm/providers/nebius.py` (new) | The Nebius Token Factory provider: one `ProviderSpec` and a model filter. It uses the global endpoint `https://api.tokenfactory.nebius.com/v1` with a Bearer key. Models come from the live `GET /v1/models?verbose=true`, with context length and price. Embedding, rerank, guard and image-generation models are dropped. `nvidia/Nemotron-3_5-Lightning` is preferred, but only when the live list has it, and the other NVIDIA models follow it under "NVIDIA Nemotron". |
| `llm/providers/catalog.py` | Nebius added to the provider list, after the existing direct providers. OpenRouter stays first, as the recommended cloud path. |
| `llm/providers/adapters/chat_completions.py` | Two additions to the shared OpenAI-compatible adapter. **(1)** A reasoning model's thinking written into the answer is removed before the JSON is read: a leading `<think>…</think>`, or thinking that ends in a lone `</think>` followed by the JSON. An answer that starts as JSON, or a reply in words, keeps any `</think>` it quotes. An answer cut off mid-thought now reports "ran out of room". **(2)** The `usage` of every answer is recorded. Both changes apply to every provider that speaks this format. Requests and parsing are otherwise unchanged; the only visible difference for the other providers is the usage line in the job log. |
| `llm/providers/usage.py` (new) | Adds up the tokens each `<provider>/<model>` was billed for: calls, prompt, completion, cached and reasoning tokens. It stores counts only, never prompts, answers or keys. |
| `core/pipeline.py` | One log line after the titles are written, or when a video gives no clips: `AI usage while this job ran, as the provider reported it: nebius/nvidia/Nemotron-3_5-Lightning: N call(s), … tokens in, … out`. It counts the whole engine, so an agent chat at the same time is included. Nothing else in the pipeline changed. |
| `ui/src/main/index.ts` | The Nebius key page and pricing page added to the allow-list of external links, so the existing "Get a key" and "Pricing" links open. |
| `ui/src/renderer/src/components/AICard.tsx`, `ui/src/renderer/src/lib/types.ts` | A provider can word its own preferred model and family (`text_why`, `text_family_label`). Without this, Nemotron would be listed as "the cheapest for text" under a "Gemma" heading, which is OpenRouter's wording. OpenRouter's wording is unchanged. |
| `tests/test_nebius.py` (new) | 25 offline tests: the provider entry, the model filter, the key check and the bad-key case, Test connection, activation, a clip request held to the app's schema, thinking inline or apart, answers that only quote `</think>`, a cut-off answer, usage (including odd values), `Retry-After`, and every Nebius error body in the API reference (402, 401, 403, 429, 503, 529, 404 for a retired model). |
| `tests/test_llm_adapters.py`, `tests/test_ai_api.py` | The expected provider order now ends with `nebius`. |

**No settings card was written.** The existing **Settings → AI → Direct
provider API** card reads the provider list, so Nebius appears there with the
key field, model picker, **Test connection** and status the other providers
already have. Choosing a Nebius model writes `model: nebius/<model id>`, the
same way every other provider is chosen.

## Hackathon tooling (not used by the app)

| File | What |
|---|---|
| `scripts/hackathon/nebius_check.py` | One real Nemotron request, made the app's way: the same provider entry, adapter, prompt, schema and parser a Basketball job uses, on your key. `--probe` tries each response format the adapter can fall back to and reports where the thinking came back. |
| `scripts/hackathon/llm_bench.py` | The Gemma vs Nemotron benchmark. `serve` runs the real app and saves what `find_clips` received: transcript, sound and picture curves, score-bug readings, shots and cutaways. `replay` runs only the language-model stages again on those saved inputs with a chosen backend. `compare` writes the tables in [BENCHMARKS.md](BENCHMARKS.md). |
| `docs/hackathon/*.md` | This file, [BENCHMARKS.md](BENCHMARKS.md), [DEVPOST_DESCRIPTION.md](DEVPOST_DESCRIPTION.md), [DEMO_SCRIPT.md](DEMO_SCRIPT.md), [JUDGE_SETUP.md](JUDGE_SETUP.md) and [FEEDBACK.md](FEEDBACK.md). The earlier Codex × TwitchCon entry's files moved to `docs/hackathon/codex-twitchcon-2026/`, with one path inside updated and a short index README added. |
| `.github/workflows/{docker-image,pages,mirror,stats}.yml` | A guard of `if: github.repository == 'ColinGPT9/clips-studio'` on each job. Without it, this copy could publish over the app's Docker image (`ghcr.io/colingpt9/clips-studio-engine`), its website, its IndexNow ping, its Hugging Face mirror or its stats. In the original repository the guard does nothing. |
| `README.md` | A hackathon section at the top. The original README follows it, unchanged. |

## What was not changed

Nothing below was retrained, replaced or reconfigured. `git diff 34f0c23 --stat`
shows no change under these paths.

- **YOLO and OpenCV** (`video/tracker.py`, `sports/soccer/ball.py`,
  `sports/basketball/reactions.py`, `leadin.py`, `sports/basketball/crop/observe.py`).
  COCO YOLOv8n still finds the shots, cutaways, reactions and the people in a
  close-up. OpenCV and FFmpeg still decode frames and give the cut and pan signals.
- **E-BARD** (`sports/basketball/crop/sources.py`, `config/sports.yaml`
  `framing.sources`). Its YOLOv8n weights, `BODD_yolov8n_0001.pt` (CC BY 4.0,
  unmodified), still detect the basketball, hoop, players and referees for the 9:16
  framing. Clips Kitty's own tracker and planner (`sports/basketball/crop/`) turn
  those detections into the crop.
- **Whisper / faster-whisper** (`transcription/`): the transcript, word
  timings, captions and subtitles.
- **PANNs** (crowd, whistle, buzzer), **RapidOCR** (the score bug, the game clock)
  and the **Sports → Basketball** profile: moments, highlight windows, scoring,
  selection, title checks.
- **FFmpeg rendering**, the **Clip Editor**, and **WoopSocial** (`publish/woopsocial.py`).
- **Every other provider**, OpenRouter included. Its requests are unchanged.

## Why this stays modular for the next model

The Basketball code never names a provider or a model (`grep -rn "nebius\|nemotron" sports/`
finds nothing). A job gets whatever backend `llm.backend` names, through
`llm/registry.create_backend`, behind the one `LLMBackend` interface
(`llm/base.py`). Adding Meta's models later is another catalogue entry, such as the
existing `meta` provider (Muse Spark), which is already in the catalogue. Nebius
also lists `meta-models/muse-spark-open`, so it can be selected on the same
Nebius key with no code at all.
