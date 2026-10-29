# Devpost submission text

Paste each section into the field it is named for. Benchmark figures come from
[BENCHMARKS.md](BENCHMARKS.md). Update them there first, then copy them here.

---

**Project name:** Clips Kitty — Basketball Highlights with NVIDIA Nemotron

**Elevator pitch (200 characters):** Free, open-source AI basketball highlights for small creators: NVIDIA Nemotron on Nebius Token Factory does the reasoning, alongside YOLO, OpenCV, E-BARD and Whisper.

**Track:** Best Apps and Agents

**Try it out:**

- Repository: https://github.com/ColinGPT9/clips-kitty-nebius-basketball (AGPL-3.0)
- Test build and setup: https://github.com/ColinGPT9/clips-kitty-nebius-basketball/blob/main/docs/hackathon/JUDGE_SETUP.md
- Original app: https://github.com/ColinGPT9/clips-studio

**Video:** _the public YouTube link, under three minutes_

**Built with:** nvidia-nemotron, nebius-token-factory, python, fastapi, electron,
react, typescript, ffmpeg, opencv, ultralytics-yolo, e-bard, faster-whisper, panns,
rapidocr, woopsocial

---

## Inspiration

I'm a Toronto Raptors fan, and I love the way basketball highlight pages
like House of Highlights cut a game: the play, the shot, the bench losing it,
all in a few vertical seconds. That editing is skilled and repetitive, and the
tools that automate it are priced for media companies, not for a kid posting
their school's games from a laptop. I built Clips Kitty as a free, open-source
clipping app for exactly those smaller creators. For this hackathon I wanted
its basketball mode to reason with an open NVIDIA model, without anyone
needing a GPU. (House of Highlights is only an inspiration for the style. There
is no affiliation, and none of its branding or footage is used.)

## What it does

You give Clips Kitty a basketball game: a file, a stream, or a URL. Choose
**Sports → Basketball**. It returns vertical highlight clips, one play each,
with the build-up, the shot and the reaction. The 9:16 crop follows the ball
and the players, the captions are word-synced, and the titles name who scored.
You review and edit them in its Clip Editor, then publish through WoopSocial to
your own connected accounts.

In this edition, the reasoning is done by **NVIDIA Nemotron served by Nebius
Token Factory**. That means scoring the commentary, judging the candidate
moments, reranking the finalists, and writing and checking the titles. It
runs on the creator's own Nebius key, chosen in **Settings → AI** like any other
provider. The computer vision and transcription are Clips Kitty's existing ones.

## How we built it

Clips Kitty already had one language-model interface with interchangeable
backends: local Ollama, OpenRouter, and direct provider APIs. Nebius Token
Factory speaks the OpenAI chat-completions format. So the new provider is one
catalogue entry, `llm/providers/nebius.py`. It lists the live models from
`/v1/models?verbose=true` with their context and price, prefers
`nvidia/Nemotron-3_5-Lightning` when the list has it, and puts the other
NVIDIA models after it. Nothing in the Basketball code names a provider.

The basketball pipeline reads the game with its existing tools:

- **faster-whisper** writes the commentary with word timings.
- **PANNs** hears the crowd, whistle and buzzer.
- **RapidOCR** reads the score bug and the clock.
- **YOLO** and **OpenCV** find the cutaways and the shots.

Then Nemotron reasons over that evidence through the same prompts and JSON
schemas the existing model used. At render time, **E-BARD**'s
YOLOv8n basketball detector finds the ball, hoop, players and referees.
Clips Kitty's own tracker turns those detections into a crop that follows
the play, and **FFmpeg** renders it.

Two additions to the shared adapter came out of making a reasoning model
reliable in a JSON pipeline:

- Thinking written into the answer (`<think>…</think>`) is removed before
  the JSON is read.
- The token usage Nebius reports on every answer is added up and logged for each job.

To compare models honestly, we built a benchmark harness. It freezes
everything the pipeline knows about a game at the moment the language model
is first asked, then replays only the language-model stages with each
backend. Gemma and Nemotron see identical evidence.

## Challenges we ran into

- **Reasoning models in a JSON pipeline.** Clips Kitty's parsers read from
  the first `{` to the last `}`. A reasoning model that thinks inline puts
  braces in its thinking, so a whole chunk's clips could be lost. Nebius
  documents `reasoning_content`, but not how each Nemotron model returns its
  thinking. The adapter now handles thinking that comes back apart, inline, or not at all.
- **Thinking that never ended.** On the first live run, Nemotron spent all
  8,192 output tokens the app allows thinking about the first chunk of a game,
  answered nothing, and the job failed. Asking for `reasoning_effort: "low"`
  didn't shorten it. Turning thinking off through the chat template
  (`enable_thinking: false`) did: the same game then ran end to end, with every
  answer usable. The Nebius provider now sends that switch to NVIDIA models.
- **Models change fast.** Two of the five NVIDIA models in today's
  catalogue are retired on 2026-11-09, and requests aren't rerouted. So the app
  never hard-codes a Nebius model. It lists the live ones and only prefers one.
- **A fair comparison.** Re-running a whole game per model would change the
  transcript and the detections between runs. Freezing the evidence once made
  the language model the only variable.
- **Footage we can share.** NBA footage, and even many Creative Commons games,
  can't be posted as cut highlights. The demo uses a 1995 high-school game that
  its broadcaster released into the public domain.

## Accomplishments that we're proud of

- The whole Basketball pipeline can run on Nemotron through Nebius with no change
  to its computer vision, transcription, rendering or publishing.
- Adding Nebius took one provider file. The existing settings card, model picker,
  connection test and job runner pick it up from the provider list. The only UI
  change lets a provider word its own preferred model.
- A benchmark that holds everything but the model constant, and reports what
  each model did worse as well as better.
- One real game, a 43-minute NBA broadcast, ran end to end on Nemotron through
  Nebius: 43 calls, every answer usable, 62 rendered clips, for under a cent
  ($0.009).
- 26 new offline tests cover every documented Nebius error body, rate-limit
  waits, thinking inline or apart, Nemotron asked not to think, answers that
  only quote the closing tag, and usage.

## What we learned

Measured on one 43-minute NBA broadcast, on an RTX 3060
([BENCHMARKS.md](BENCHMARKS.md#live-run-a-2026-nba-game-on-nemotron)):

- **A reasoning model needs its thinking turned off in a JSON pipeline.** With
  it on, one call used the whole output budget and the job failed. With it off,
  43 of 43 answers were usable JSON on the first try.
- **It's fast and cheap.** The language model took 90 s for the whole game and
  cost $0.0087. The whole job, from the file to 62 rendered clips, took
  30.1 minutes, most of it transcription, reading the score bug and rendering.
- **Titles still need the scoreboard.** The basketball title check found 37 of
  57 titles named the wrong play or scorer, and fixed them by rewriting or
  writing from the score bug. One wrong title still got through.
- **On the same evidence as Gemma, Nemotron wrote better titles.** On the 1995
  benchmark game, two Nemotron replays got 4 of 9 titles right and 4 or 5
  partly right, against Gemma's 1 or 2 right and 4 wrong. Nemotron also found
  the court storm Gemma missed. Both missed the same four early plays. Each
  replay took about 30 s and cost about a third of a cent.

## What's next for Clips Kitty

Clips Kitty will stay free and open source. The provider architecture is meant
to take the next model without a computer-vision rebuild, and that is next:
Meta's models for the Meta Global AI Developer Hackathon. After that come better
reactions from the picture, and NBA-style score bugs on more broadcasts.

---

## How the project was significantly updated during the submission period

Clips Kitty started in June 2026. The submission period opened on 2026-08-26.
Since then:

- **2026-09-25:** cloud AI on the creator's own key. This added OpenRouter, then
  OpenAI, Gemini, Claude, xAI and Meta.
- **2026-09-30:** the Sports framework.
- **2026-10-05 to 2026-10-21:** the whole Basketball module. This covers moments
  from the score bug, commentary and crowd; the E-BARD 9:16 crop around the play
  (#151, #156); and release 2.1.0 (#168).
- **2026-10-28 onward (this edition):** the Nebius Token Factory provider for
  NVIDIA Nemotron, reasoning-output handling and token-usage reporting, the
  frozen-evidence benchmark, and the documentation.

331 of the 752 commits up to `34f0c23`, where this edition starts, were made
inside the submission period. The
git history is public and dated. Every change in this edition is listed in
[NEBIUS_CHANGES.md](NEBIUS_CHANGES.md). Nothing was trained or fine-tuned, and
existing components (YOLO, OpenCV, E-BARD, Whisper, FFmpeg, WoopSocial) are
credited as existing.

## Feedback on Nebius Token Factory and NVIDIA

See [FEEDBACK.md](FEEDBACK.md). Paste it into the feedback field.
