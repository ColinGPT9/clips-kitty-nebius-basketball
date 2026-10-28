<div align="center">

# Clips Kitty — Basketball Highlights with NVIDIA Nemotron

**Open-source AI basketball highlights using NVIDIA Nemotron through Nebius Token Factory, with existing YOLO, OpenCV, E-BARD tracking, and Whisper.**

Nebius × NVIDIA Global AI Hackathon 2026 · Track: **Best Apps and Agents** · Licence: **AGPL-3.0**

[What changed](docs/hackathon/NEBIUS_CHANGES.md) · [Benchmarks](docs/hackathon/BENCHMARKS.md) · [Judge setup](docs/hackathon/JUDGE_SETUP.md) · [Demo script](docs/hackathon/DEMO_SCRIPT.md) · [Feedback](docs/hackathon/FEEDBACK.md)

</div>

> This is the hackathon edition of **[Clips Kitty](https://github.com/ColinGPT9/clips-studio)**
> (`ColinGPT9/clips-studio`). It is a copy with full history, branched at commit
> `34f0c23`. The original app, its Microsoft Store release and its website were not
> changed. [The original README](#clips-kitty-the-free-open-source-opus-clip-alternative-for-chatting-gaming-and-sports-on-your-own-pc)
> follows this section, unchanged.

## Original Clips Kitty

Clips Kitty is a free, open-source AI video clipping app for small YouTubers,
streamers and sports creators. They often can't pay for Opus Clip, Klap or an
editor. You give it a long video: a stream, a podcast, or a game. It finds the
moments worth posting, cuts them to 9:16 with captions and a crop that follows
the action, lets you fix them in its Clip Editor, and can post them for you.
By default it runs on your own PC: Google Gemma through Ollama, and Whisper.
There is no per-clip fee and no server in between. It ships on the Microsoft Store.

Its **Sports → Basketball** mode reads a game the way a highlights editor
would. It uses the score bug, the commentary, the crowd and the buzzer, and the
camera's cutaways to the crowd. It clips one play at a time, with the build-up
and the reaction, and follows the ball in 9:16.

## Hackathon adaptation

Clips Kitty already had a language-model interface with interchangeable
backends: local Ollama, OpenRouter, and direct provider APIs on the user's own
key. Nemotron was reachable before, through OpenRouter or as a small local
model. This edition adds **Nebius Token Factory** as one more provider in that
architecture, so the Basketball pipeline can run its language-model reasoning on
**NVIDIA Nemotron served by Nebius**, on the creator's own Nebius key.

It is one new provider file, `llm/providers/nebius.py`, plus two small
additions to the shared OpenAI-compatible adapter. The first strips a reasoning
model's thinking from the answer. The second counts the tokens the provider
reports. Settings → AI picks Nebius up from the provider list with no new
screen. Basketball doesn't know which model it's talking to. Every file
changed is listed in [NEBIUS_CHANGES.md](docs/hackathon/NEBIUS_CHANGES.md).

## Motivation

I'm Colin, a Toronto Raptors fan. I got into AI through basketball, computer
vision and the creators who make highlights. Pages like House of Highlights
are the kind of basketball content I wanted Clips Kitty to make possible for
someone posting from a laptop. That means the play, the shot, the bench and the
crowd, cut fast and vertical. House of Highlights is only an inspiration for
that style. This project has no affiliation with it and uses none of its
branding, footage, watermarks or overlays.

Highlight editing is skilled, repetitive work. The tools that automate it are
priced for media companies. Clips Kitty is AGPL and free. Putting Nemotron on
Nebius behind it gives creators without a GPU a fast, cheap way to run the
reasoning step on an open model. It is billed per token to the creator's own
account, and [BENCHMARKS.md](docs/hackathon/BENCHMARKS.md) has the measured cost
of one game.

## Technical architecture

What runs, in the order the pipeline actually runs it:

```
Basketball video (a file or a URL)
  │
  ├─ while Whisper runs, in parallel ───────────────────────────────────────────────
  │    OpenCV / FFmpeg  frames, audio and motion curves, scene cuts
  │    PANNs            game sounds: crowd, whistle, buzzer
  │    RapidOCR         the score bug and game clock, read from OpenCV frame grabs
  │    YOLO (COCO v8n)  shot types and cutaways to the crowd and bench, on keyframes
  │
  ├─ Whisper (faster-whisper)  commentary transcript with word timings
  │
  ├─ find_clips ──────────────────────────────────────────────────────────────────
  │    NVIDIA Nemotron (Nebius)  scores each commentary chunk, with the sound
  │                              and picture events beside it
  │    Basketball profile        moments from the score bug, the commentary's
  │                              callouts (dunk, three, block, steal…), the crowd,
  │                              the clock (clutch, buzzer-beater, game-winner)
  │    NVIDIA Nemotron (Nebius)  scores the moment and signal windows
  │    Fusion + selection        one play per clip, scores and bonuses
  │    NVIDIA Nemotron (Nebius)  reranks the finalists
  │
  ├─ Window refinement      YOLO + OCR of the game clock: start on the court, not a
  │                         cutaway (broadcast footage with a score bug only)
  ├─ NVIDIA Nemotron (Nebius)  titles, descriptions, hashtags; with a score bug,
  │                            each clip's score-bug note, and the basketball title
  │                            check rewrites any that name the wrong scorer
  │
  ├─ Render (FFmpeg)        cut, captions from Whisper's word timings, 9:16 crop:
  │    E-BARD (YOLOv8n)     detects ball, hoop, players and referees per frame
  │    Clips Kitty tracker  player tracks, the ball's path, the basket, the play's
  │                         phases → the crop for every frame (OpenCV: cuts, pan)
  │
  ├─ Clip Editor            review, trim, retitle
  └─ WoopSocial             post to the creator's connected accounts, when they press Publish
```

| Component | Role in this project | Changed? |
|---|---|---|
| **YOLO** (Ultralytics YOLOv8n, COCO) | Finds people for shot types, cutaways to the crowd and bench, close-ups, and where clips start | No |
| **OpenCV** | Frame decoding and grabs, motion and cut signals, phase-correlation camera pan, the orange-ball colour gate | No |
| **E-BARD** | [Gabriele Giudici's](https://github.com/GabrieleGiudic/E-BARD) YOLOv8n fine-tuned on NBA broadcasts (CC BY 4.0, unmodified). It detects the **basketball, hoop, players and referees** in each frame for the 9:16 framing. Clips Kitty's own code in `sports/basketball/crop/` tracks those detections and plans the crop around the play | No |
| **Whisper** (faster-whisper) | Transcript, word timings, captions and subtitles | No |
| **PANNs**, **RapidOCR** | Crowd, whistle and buzzer; the score bug and the game clock | No |
| **NVIDIA Nemotron** | The language-model reasoning: commentary chunks, moment windows, the rerank, titles and descriptions, and the title check | **New backend** (same prompts, schemas and parsers as before) |
| **Nebius Token Factory** | Hosts Nemotron. An OpenAI-compatible API, called directly on the user's key | **New provider** |
| **FFmpeg** | Cutting, captions, crop and encoding | No |
| **WoopSocial** | Posting to the creator's connected social accounts | No |

**What Nemotron sees.** Nemotron gets the same evidence the existing model
always got, through the same prompts:

- the Whisper transcript, chunk by chunk;
- the events found beside it: crowd and buzzer sounds, audio spikes, motion,
  scene cuts and voice jumps;
- the windows the basketball moments and signal peaks proposed;
- for titles, on footage with a score bug, each clip's score-bug note: who
  scored, the score, the quarter and clock, and whether it was crunch time.

No cloud model is ever sent the picture. The relabelling of reaction shots
(courtside, bench, coach) only runs with a local model that can see images, so
with Nemotron the reactions keep the labels the detectors and the commentary
gave them. (`gemma:7b`, the default, can't see images either.)

## Installation and demonstration

The full, tested steps are in **[docs/hackathon/JUDGE_SETUP.md](docs/hackathon/JUDGE_SETUP.md)**.
The short version:

```bash
git clone https://github.com/ColinGPT9/clips-kitty-nebius-basketball
cd clips-kitty-nebius-basketball
pip install -r requirements.txt
python scripts/fetch_basketball_model.py && python scripts/fetch_panns.py
export NEBIUS_API_KEY=...                   # from tokenfactory.nebius.com/project/api-keys
python scripts/hackathon/nebius_check.py    # one real Nemotron call, made the app's way
cd ui && npm install && npm run dev         # the desktop app
```

In the app:

1. Open **Settings → AI → Direct provider API → Nebius Token Factory**.
2. Paste the key and choose **Nemotron 3.5 Lightning**.
3. Press **Test connection**.
4. Go to **Add videos → Sports → Basketball**, add a game, and press **▶ Start queue**.

Use footage you have the rights to. The demo uses a public-domain 1995
high-school game from the Internet Archive.

## Benchmarks

Gemma and Nemotron were run on the same 22-minute game, with the same Whisper
transcript and the same sound and picture readings, frozen once and replayed
per model. The language model was the only variable. The results, including
what Nemotron did worse, are in **[docs/hackathon/BENCHMARKS.md](docs/hackathon/BENCHMARKS.md)**.

<!-- BENCHMARK SUMMARY -->

## New versus existing work

Clips Kitty started in June 2026. During the hackathon's submission period
(from 2026-08-26), Colin built most of what this entry demonstrates:

- **2026-09-25:** cloud AI on the user's own key, with OpenRouter, OpenAI, Gemini,
  Claude, xAI and Meta.
- **2026-09-30:** the Sports framework.
- **2026-10-05 to 2026-10-21:** Basketball, including the E-BARD 9:16 crop
  (#151, #156) and release 2.1.0 (#168).

331 of the 752 commits up to `34f0c23`, where this edition starts, were made
inside the submission period. The
WoopSocial publisher, the Clip Editor, Whisper and the YOLO and OpenCV work
predate it or were built alongside.

**New in this hackathon edition**, 2026-10-28 onward:

- the Nebius Token Factory provider;
- reasoning-model output handling and token-usage reporting in the shared
  adapter;
- the frozen-evidence benchmark harness and the Nemotron check script;
- the guards that stop this copy publishing over the original's image or
  website;
- this documentation.

Nothing was trained, fine-tuned or replaced.
[NEBIUS_CHANGES.md](docs/hackathon/NEBIUS_CHANGES.md) lists every changed file.

## Future development

The language model sits behind one interface (`llm/base.py`). A provider is an
entry in a catalogue (`llm/providers/catalog.py`), and the Basketball pipeline
never names one. The next model is a configuration change, not a computer-vision
rebuild. That includes Meta's models for the Meta Global AI Developer
Hackathon. The catalogue already has Meta's own API (Muse Spark), and Nebius
lists `meta-models/muse-spark-open` on the same key. No Meta-specific work was
done here.

## Licensing

AGPL-3.0, as the original, in [LICENSE](LICENSE). Copyright (C) 2026 ColinGPT9,
as in [NOTICE](NOTICE). Third-party components and model weights keep their own
licences, listed in NOTICE:

- the E-BARD weights are CC BY 4.0, by Gabriele Giudici;
- Ultralytics YOLOv8 is AGPL-3.0;
- faster-whisper and the Whisper weights are MIT.

Nemotron is called as a hosted API on Nebius, and no model weights are
distributed here. Nebius lists Nemotron 3.5 Lightning under OpenMDW-1.1, and
Nemotron 3 Nano and Super under the NVIDIA Open Model License.

---
