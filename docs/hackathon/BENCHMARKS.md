# Gemma vs Nemotron on the same basketball game

**Question.** What changes when the Basketball pipeline's language model is
NVIDIA Nemotron on Nebius Token Factory instead of Clips Kitty's default Gemma,
with everything else held constant?

**Status.** See the run table under [Results](#results). A run is reported
only once it has actually been made. Nothing below is estimated or projected.

## Method

### The game

[*BHS Sports TV Presents: Varsity Boys Basketball – Belmont Marauders vs Salem
Witches, 03/11/1995*](https://archive.org/details/basketball95), from Belmont Media
Center, marked **Public Domain Mark 1.0**. It is 81 minutes of 640×480 VHS, shot
handheld from the stands, with two commentators and no score bug. The benchmark
uses the final 22:27 (`ffmpeg -ss 3510 … -c copy`). In that stretch the lead
changes hands in the last minute and Salem wins 61–59 on a drive with about
five seconds left. Belmont then fails to get a shot off from a three-second
inbound, and the crowd storms the court.

This is a hard case for Basketball mode. There is no score bug, so there are no
confirmed baskets from OCR, and the score-bug title check and window refinement
don't apply. Only one camera is used, so there are no real broadcast cutaways.
The cutaway finder still flagged 7 shots, most likely when the handheld camera
swung to the crowd or the bench, and both models are given them. The plays come
from the commentary, the crowd and the picture.

### Holding everything but the language model constant

The pipeline asks its language model nothing until `find_clips`
(`analysis/fusion.py`). By then every other reading of the game is finished.
`scripts/hackathon/llm_bench.py serve` runs the real app and saves exactly what
`find_clips` received, before it runs:

- the Whisper transcript (`small` on CPU, 314 segments with word timings);
- the audio and visual curves (FFmpeg and OpenCV);
- PANNs' game sounds (9 crowd cheers);
- the `BasketballProfile`, with its score-bug pass (none on screen), the
  COCO-YOLO shots and the 7 cutaways it found.

The saved file is `local_1144e6ee30c4.pkl`, 0.3 MB.

`llm_bench.py replay` loads that file and runs, with one named backend:

1. `find_clips`: the language model scores each commentary chunk and the
   moment and signal windows, the basketball moments are built, scores are
   fused and the plays selected, and the language model reranks the finalists.
2. The basketball window refinement. On broadcast footage this runs YOLO and
   OCR of the clock again on the clip windows, which is deterministic for a given
   window. On this game it does nothing, because there is no score bug
   (`BasketballProfile.refine_windows` skips club and phone footage).
3. Titles and descriptions, then the basketball title check.

All three use the app's own code, unchanged. Every language-model call goes
through a meter that records:

- the pipeline step that made it;
- its duration;
- whether the answer held the JSON object that step reads (judged the way the
  app's own parsers find it);
- the tokens the provider reported.

Whisper, PANNs and E-BARD are **not** run again, nor is the read of the game
(score bug, shots, cutaways). The same source, transcript and detections go to
every backend. Rendering (FFmpeg,
E-BARD framing) comes after the language model's last decision and doesn't
change what was picked, so it is left out of the timings.

**Only the model is changed.** The prompts, schemas, chunk sizes
(`analysis.chunk_seconds: 300`), thresholds (`clips.min_score: 55`) and the
Highlights choice (**Best moments**) are the app's defaults for every run.

**One difference comes from the backends themselves, not the benchmark.** A cloud
backend is asked for a strict JSON schema, with the app's existing fallbacks to
a JSON object and then plain text. Ollama is asked for `format: json` with a
1,024-token answer limit. That has always been true of local and cloud models in
Clips Kitty (`llm/base.generate_json`). The third run below, Gemma on Nebius,
separates the model from the way it is served.

Neither side looks at the picture. Basketball relabels reaction shots only with
a local model that can see images, and `gemma:7b` can't, so the vision step is
skipped for every run here.

### The runs

| Label | Backend | Where it runs |
|---|---|---|
| `gemma` | `ollama/gemma:7b`, the app's default in `config/settings.yaml` | Ollama 0.40.12 (Q4 GGUF) on the 4-core CPU of the test machine, no GPU |
| `nemotron` | `nebius/nvidia/Nemotron-3_5-Lightning`, the preferred Nebius model | Nebius Token Factory |
| `gemma-nebius` | `nebius/google/gemma-3-27b-it`: Gemma on the same API as Nemotron | Nebius Token Factory |

Test machine: a 4 vCPU, 16 GB Linux cloud container, Python 3.11, CPU-only.
Local Gemma's latency there is far slower than on a creator's GPU. Treat its
timings as a lower bound on speed, not as typical.

### Reference: the plays a highlights editor would want

There is no official play-by-play for a 1995 high-school game. This list was
written by Claude from the Whisper transcript, before any model's clips were
seen. Frames were checked at the two that matter most (1115 s: Kamah's drive;
1268 s: the court storm). Each item is a play the commentary describes as it
happens. Times are seconds into the 22-minute file.

| # | When (s) | Play | Kind |
|---|---|---|---|
| R1 | 57–73 | Kohane's jumper misses, and Christensen's putback gives Belmont its largest lead, 5. Salem calls a timeout and the crowd cheers | putback, crowd |
| R2 | 170–182 | Mateo's "big three-pointer" for Salem | three |
| R3 | 235–264 | Downes' wild shot. A no-call, then a travel call; "Belmont fans very upset" | controversy, crowd |
| R4 | 268–295 | No towel in the gym, so the ref wipes the floor with a Kleenex | light moment |
| R5 | 337–361 | Penn's defensive play ties it at 54, and Christensen scores inside | defense, post score |
| R6 | 393–418 | Christensen's big rebound, then Penn steals it from Conley | rebound, steal |
| R7 | 497–510 | Penn is blocked by LaPierre, Mateo steals it, and Belmont is fouled | block, steal |
| R8 | 551–567 | Christensen's "big free throw" | free throw |
| R9 | 587–604 | Costello strips Penn, Penn misses a three, Conley rebounds | steal |
| R10 | 824–837 | 58–57 with a minute left, and Penn turns it over | turnover, clutch |
| R11 | 849–895 | Christensen is fouled: "the biggest free throws of this kid's career" in a standing, loud gym | clutch free throws |
| R12 | 976–995 | Christensen misses the first and makes the second: tied with 33 s left | clutch free throw |
| R13 | 1107–1158 | Kamah "shakes and bakes" and scores with about five seconds left. Time is put back on the clock | **game-winner** |
| R14 | 1158–1280 | The last three seconds: the inbound, no shot, "Salem wins it", and the crowd storms the court | final possession, **court storm** |

The must-haves for a highlights page are R1, R2, R6, R7, R9, R13 and R14.

### What is measured

| Measure | How |
|---|---|
| Highlight selection | The clips each run kept, matched to the reference by time overlap |
| Missed moments | Must-have reference plays with no clip overlapping them |
| False positives | Kept clips that overlap no reference play and show no play of their own on review: talk, timeouts, the broadcast's sign-off |
| Candidate ranking | Each kept clip's final score, and where the must-haves land in the order |
| Event interpretation | Whether each clip's hook and reason describe what happens in it |
| Title and description accuracy | Each title read against its clip's transcript: right team, right player, right result, nothing invented |
| Structured response reliability | The share of calls whose answer held the JSON object the step reads, the calls retried, and the fallbacks used |
| Inference latency | Seconds per call and per step, and the total language-model time |
| API usage | Calls, and the prompt, completion, cached and reasoning tokens the provider reported (Ollama doesn't report them through the app) |
| Cost | Nebius' reported tokens × its listed prices, read from its public catalogue at compare time. Local Gemma has no per-token cost |

The selection, interpretation and title judgements are made by reading each
clip's transcript against the reference, by Claude, and are marked as such.
Colin should spot-check them by watching the clips.

## Results

| Run | Status |
|---|---|
| `gemma` (app run, job 1) | done, 2026-10-28: 6 clips in 88.8 min, below |
| `gemma` (replay) | done, 2026-10-28: 7 clips, 14 calls, 56.8 min, below |
| `nemotron` (replay) | not run: one game on Nebius, by choice ([the live run](#live-run-a-2026-nba-game-on-nemotron) used a different game) |
| `gemma-nebius` (replay) | not run: one game on Nebius, by choice |

### The app run: Gemma, the whole job (job 1)

The real app, from the file to rendered clips, with `ollama/gemma:7b` on the
test machine's CPU. Whisper, the read of the game, the language model, E-BARD
framing and FFmpeg all ran. It took 88.8 minutes. About an hour of that was the
language model: the inputs were frozen at 10:09 and the first clip was rendered
at 11:11.

| # | Window (s) | Score | Reference | Title and description | Title judged |
|---|---|---|---|---|---|
| 1 | 1111.7–1146.8 | 86 | **R13, the game-winner** | "Marauders' Hail Mary": "Kamah makes a crucial play to regain possession for Belmont." | Wrong. Kamah's drive wins it for Salem, and it isn't a Hail Mary |
| 2 | 780.3–808.5 | 82 | none. Mostly talk about fouling tactics, then "money free throw by Kamah" | "Free Throw Gamble": "Kamah attempts a costly free throw in a tight game." | Wrong. He makes it |
| 3 | 407.7–423.4 | 82 | **R6**, Penn's steal from Conley | "Salem's Defensive Stance": "Doug Conley struggles to handle Penn's quick ball movement." | Right players and team, though it doesn't say "steal" |
| 4 | 290.8–307.4 | 74 | the last 4 s of R4 (the Kleenex). Mostly talk, then a pass to Kamal | "Kamal's Seamless Pass" | Wrong. Kamal receives the pass |
| 5 | 820.5–835.5 | 70 | R10, 58–57 and Penn's turnover | "Marauders Close Out": "Belmont narrowly edges out Penn in the final minutes", #gamewinner | Wrong. Penn is a player, and Belmont lost |
| 6 | 242.3–254.7 | 67 | R3, the no-call | "Controversial Call": "A questionable refereeing decision impacts Downes' scoring opportunity." | Right |

- **Selection:** 4 of the 6 clips are reference plays, and 2 of the 7 must-haves
  are there: R6 and R13.
- **Missed moments:** R1 (the putback and timeout), R2 (Mateo's three), R7 (the
  block and steal), R9 (Costello's strip) and R14 (the last three seconds and
  the court storm).
- **False positives:** clip 4 is mostly talk. Clip 2 is mostly talk too, but it
  ends on a real free throw that the reference list doesn't have.
- **Ranking:** the game-winner was ranked first.
- **Titles:** 2 of 6 right. 4 name the wrong team, the wrong result or the wrong
  direction of a pass. There is no score bug on this footage, so the basketball
  title check had nothing to hold them to.
- **Structured responses:** Ollama's model runner stopped once while the
  windows were scored ("model runner has unexpectedly stopped"), and the app
  asked again without JSON mode. Every other answer was read.
- **Rendering:** every clip was rendered 1080×1920. E-BARD framing ran in
  "assisted" mode (the ball led 0–30% of each play). The 4:3 VHS picture sits on
  a blurred fill with word-synced captions. The end card was skipped because
  the container lacks the Windows font it uses. That's a property of the test
  machine, not of this change.

The selection, interpretation and title judgements above were made by Claude,
reading each clip's transcript against the reference list, and checking the
rendered game-winner clip's picture.

### The replay: Gemma on the saved evidence

`llm_bench.py replay … --backend ollama/gemma:7b --label gemma`. It is the same
model on the same evidence as the app run, so it also shows how much one model
varies from run to run. The run's file is [bench/gemma.json](bench/gemma.json).

| Step | Calls | Usable JSON | Mean seconds per call |
|---|---|---|---|
| Chunk scoring | 6 | 6 | 316.3 |
| Window scoring | 6 | 6 | 209.8 |
| Rerank | 1 | 1 | 60.3 |
| Titles | 1 | 1 | 188.3 |
| **Total** | **14** | **14** | **3,405 s (56.8 min)** |

The basketball title check made no calls, because there is no score bug to
check against. Ollama's runner stopped once again during window scoring. The
backend asked again without JSON mode, inside the call, so the meter counts
that call as one usable answer. Ollama doesn't report tokens through the app,
and the run has no per-token cost.

| # | Window (s) | Score | Reference | Title and description | Title judged |
|---|---|---|---|---|---|
| 1 | 785.9–808.5 | 83 | none: talk, then Kamah's free throw | "Free Throw": "Kamah sinks a free throw after Penn attempts a questionable play." | Partly. He makes it, but Penn tries nothing |
| 2 | 242.3–254.7 | 82 | R3, the no-call | "Downes' Wild Shot": "Downes attempts a contested shot after receiving the ball from LaPierre." | Partly. LaPierre defends, and the no-call is missed |
| 3 | 407.7–423.4 | 81 | **R6**, Penn's steal | "Conley's Struggles": "Doug Conley struggles to handle the ball against Penn's pressure." | Right |
| 4 | 290.8–307.4 | 76 | the last 4 s of R4 | "Kamal's Move": "Kamal makes a play to receive the ball from the referee." | Wrong |
| 5 | 820.5–835.5 | 71 | R10 | "Marauders Close": "The Marauders take the lead with just over a minute remaining." | Wrong. They already led, and Penn's turnover is missed |
| 6 | 1118.0–1145.5 | 69 | **R13, the game-winner** | "Kamah's Highlight": "Kamah makes a strong move to regain possession for the Marauders." | Wrong team |
| 7 | 885.0–908.9 | 57 | the end of R11 | "Witches' Timeout": "Salem calls a timeout with the game tied." | Wrong. The Witches lead by one, and the Marauders need the timeout |

- **Against the app run:** the same six plays were picked again, each with a
  time overlap of at least 0.78, plus one more (clip 7). The game-winner fell
  from first to sixth of seven, and every title was written differently. So a
  single run of one model can't separate two models on ranking or titles. A
  Nemotron replay on this game should be run twice for that reason.
- **Selection:** 2 of the 7 must-haves again (R6 and R13), and the same five are
  missed: R1, R2, R7, R9 and R14.
- **Titles:** 1 right, 2 partly right, 4 wrong.

## Live run: a 2026 NBA game on Nemotron

**This is a different game from the 1995 benchmark above**, so it is not a
like-for-like comparison with Gemma. It is the first time the pipeline ran on
Nemotron through Nebius with a real key, from the file to rendered clips.

### The game and the machine

- **The game:** Toronto Raptors at Washington Wizards, 23 October 2026. A
  42:49 extended-highlights upload of the broadcast, 720p60, with the score bug
  on screen. It is NBA footage, so neither the video nor any clip is in this
  repository or published anywhere. Only numbers and titles are reported here.
- **The machine:** Colin's Windows PC, an RTX 3060 (12 GB), NVENC encoding.
  Whisper `large-v3-turbo` ran on CUDA (512 segments). The score bug was read
  564 times in 433 s and gave 77 baskets.
- **The model:** `nebius/nvidia/Nemotron-3_5-Lightning`, $0.06 in and $0.24 out
  per million tokens.

### First attempt: thinking on, and the job failed

The provider sent Nemotron's default, which is to think before answering. The
first chunk-scoring call came back with `finish_reason: length`: Nemotron
spent all 8,192 output tokens the app allows (`GENERATE_MAX_TOKENS`) thinking,
and wrote no answer. The job stopped there.

Three short checks on `nebius_check.py`'s 67-second prompt from the 1995 game:

| Request | Seconds | Tokens out | Of which reasoning | Finish | Answer |
|---|---|---|---|---|---|
| Thinking on (the default) | 22.2 | 4,273 | 4,265 | stop | an empty clip list |
| `reasoning_effort: "low"` | 34.1 | 8,192 | 8,192 | length | none |
| `chat_template_kwargs: {"enable_thinking": false}` | 0.6 | 7 | 0 | stop | an empty clip list, valid JSON |

With `reasoning_effort: "low"` the reasoning came back apart, in
`reasoning_content`, not inline. It did not shorten the thinking.

**The fix** (commit `5159a2a`): the Nebius provider now asks every `nvidia/`
model to answer without thinking, through `chat_template_kwargs`
(`_no_thinking` in `llm/providers/nebius.py`). Other Nebius models are sent
exactly what they were before. `tests/test_nebius.py` checks both.

### The replay: Nemotron on the saved evidence, thinking off

`llm_bench.py replay` on the frozen inputs of the first attempt. Its JSON stays
on Colin's PC, because its answers quote the broadcast's commentary.

| Step | Calls | Usable JSON | Mean seconds per call |
|---|---|---|---|
| Chunk scoring | 11 | 11 | 1.7 |
| Window scoring | 11 | 11 | 3.2 |
| Rerank | 8 | 8 | 0.6 |
| Titles | 8 | 8 | 2.4 |
| Title check | 5 | 5 | 2.5 |
| **Total** | **43** | **43** | **90 s of language model in 346 s** |

- **Errors:** none. Every answer was read on the first try, in the strict
  `json_schema` format the app asks for first.
- **Time:** the other 256 s is the basketball window refinement, YOLO and the
  OCR of the game clock, which doesn't use the language model.
- **Tokens:** 64,846 in and 20,154 out. **Cost: $0.0087.**
- **Clips:** 59.

### The app run: the whole job, thinking off

The real app, from the file to rendered clips, on the same PC.

| Measure | Result |
|---|---|
| Time, end to end | 30.1 min |
| Nemotron | 43 calls, 65,834 tokens in, 19,537 out, about $0.009 |
| Clips | 62, rendered 1080×1920 with end cards |
| E-BARD framing | 49 tracked, 12 assisted, 1 uncertain |
| Basketball title check | 37 of 57 titles named the wrong play or scorer: 23 rewritten by Nemotron, 14 written from the scoreboard |

- **The top clip** scored 90: "And-one Raptors lead by 15".
- **Titles are the weak point.** The score bug caught most wrong titles, but not
  all. "Bulls's Three Puts the Raptors Up 7" survived, in a Wizards game. Many of
  the titles written from the scoreboard are plain, such as "Wizards trail by 4".
- **Not judged yet:** no reference list of plays was written for this game, so
  selection isn't scored against one. Colin hasn't watched the clips yet.

<!-- RESULTS -->

## Reproduce

```bash
# 1. The real app, with find_clips' inputs saved on the way (the job runs as usual).
CLIPS_STUDIO_DATA_DIR=$PWD/bench-data python scripts/hackathon/llm_bench.py serve --freeze bench/frozen -- --port 8765
curl -X POST localhost:8765/queue/resume
curl -X POST localhost:8765/videos/local -H 'content-type: application/json' -d '{"path": "'$PWD'/belmont-salem-1995-final.mp4",
  "title": "Belmont Marauders vs Salem Witches, boys varsity basketball, 1995 (final 22 minutes)",
  "channel": "BHS Sports TV", "sport": {"name": "basketball", "highlights": "best", "period": "full"}}'

# 2. Replays: only the language-model stages, on the saved inputs.
F=bench/frozen/local_1144e6ee30c4.pkl
python scripts/hackathon/llm_bench.py replay $F --backend ollama/gemma:7b --label gemma --out bench/runs
NEBIUS_API_KEY=... python scripts/hackathon/llm_bench.py replay $F --backend nebius/nvidia/Nemotron-3_5-Lightning --label nemotron --out bench/runs
NEBIUS_API_KEY=... python scripts/hackathon/llm_bench.py replay $F --backend nebius/nvidia/Nemotron-3_5-Lightning --label nemotron-2 --out bench/runs
NEBIUS_API_KEY=... python scripts/hackathon/llm_bench.py replay $F --backend nebius/google/gemma-3-27b-it --label gemma-nebius --out bench/runs

# 3. Tables.
python scripts/hackathon/llm_bench.py compare bench/runs/gemma.json bench/runs/nemotron.json bench/runs/gemma-nebius.json
```

Nemotron is asked not to think automatically. The Nebius provider adds the
switch for every `nvidia/` model, so these commands need nothing extra.

The frozen inputs and every run's JSON (each call's step, time, JSON check, usage,
the first 400 characters of its answer, and where each final title came from)
go in `docs/hackathon/bench/` as each run finishes, so the tables can be checked.
The NBA live run is the exception. Its frozen inputs and JSON hold the
broadcast's commentary, so they stay off this repository.
