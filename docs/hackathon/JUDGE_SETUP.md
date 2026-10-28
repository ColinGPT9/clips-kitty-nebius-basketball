# Judge setup: run the Basketball pipeline on Nemotron through Nebius

You need a **Nebius Token Factory API key**. Make one at
<https://tokenfactory.nebius.com/project/api-keys>. New accounts get $1 of trial
credit, which covers many runs: Nemotron 3.5 Lightning is $0.06 / $0.24 per
million tokens. The app talks to Nebius directly on your key. There is no
Clips Kitty server, account or proxy in between.

There are three ways in. Pick one.

| | Time | What you see |
|---|---|---|
| **A. One Nemotron call** | 5 minutes | A real request from the Basketball pipeline, made through the app's own code, and Nemotron's clips, latency and token usage |
| **B. The desktop app (Windows)** | 30 minutes, plus one game | The whole product: Settings → AI → Nebius, Sports → Basketball, the Clip Editor and WoopSocial |
| **C. The engine, headless (Linux or macOS)** | 30 minutes, plus one game | The same pipeline driven through its local HTTP API, which is how it was tested for this submission |

**Footage.** Use the public-domain 1995 Belmont vs Salem game:
<https://archive.org/details/basketball95>, Public Domain Mark 1.0, 81 minutes.
CPU transcription is slow, so cut the final 22 minutes:

```bash
curl -L -o basketball95.mp4 https://archive.org/download/basketball95/basketball95.mp4
ffmpeg -ss 3510 -i basketball95.mp4 -c copy -map 0 -avoid_negative_ts make_zero belmont-salem-1995-final.mp4
```

## Common setup (all three)

Python 3.11 and FFmpeg on the PATH. Git, and Node 20+ for option B.

```bash
git clone https://github.com/ColinGPT9/clips-kitty-nebius-basketball
cd clips-kitty-nebius-basketball
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
# CPU-only PCs: install the CPU build of PyTorch first, to skip ~4 GB of CUDA wheels
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
python scripts/fetch_basketball_model.py   # E-BARD basketball detector, 6 MB (CC BY 4.0)
python scripts/fetch_panns.py              # PANNs game sounds: crowd, whistle, buzzer, 24 MB
```

Whisper's weights download by themselves on the first run. On a CPU it uses
`small`, and on an NVIDIA GPU `large-v3-turbo`.

## A. One Nemotron call (5 minutes)

```bash
export NEBIUS_API_KEY=...        # Windows PowerShell: $env:NEBIUS_API_KEY="..."
python scripts/hackathon/nebius_check.py
python scripts/hackathon/nebius_check.py --probe   # also tries each response format
```

The script does three things:

1. It lists the models your key can call, with NVIDIA's first and their prices.
2. It sends the Basketball pipeline's real clip-finding prompt to
   `nvidia/Nemotron-3_5-Lightning`. The prompt is the final possession of the
   1995 game, as the app's Whisper transcribed it. The request goes through
   the app's own provider, adapter, schema and parser. The script prints the
   clips Nemotron picked, the seconds it took and the tokens Nebius reported.
3. With `--probe`, it asks again in each response format the adapter can
   fall back to. It reports whether each answer parsed and where the model's
   thinking came back.

Use `--model` to try another Nemotron, for example
`--model nvidia/Nemotron-3-Ultra-550b-a55b`.

## B. The desktop app (Windows)

```powershell
cd ui
npm install
npm run dev
```

The app starts its own Python engine.

1. **Settings → AI → Direct provider API → Nebius Token Factory.** Paste the
   key. It is checked against Nebius before it is saved. Choose
   **Nemotron 3.5 Lightning**, the starred model, and press **Test connection**.
   No tokens are spent.
2. **Add videos**: drop in `belmont-salem-1995-final.mp4`, turn on **Sports**,
   choose **Basketball** and **Best moments**, and add it. The queue starts
   paused on a fresh install, so press **▶ Start queue**.
3. Watch the job. The log names `nebius/nvidia/Nemotron-3_5-Lightning`. Once the
   titles are written, as rendering starts, it prints
   `AI usage while this job ran, as the provider reported it: …`.
4. Open a clip in the **Clip Editor**.
5. Optional: publish through WoopSocial. In **Settings → Publish through
   WoopSocial**, paste your own WoopSocial key and connect a social account.
   Then choose **Publish** on a clip. Nothing is posted until you press it.

The desktop app needs no local model for this. You can choose **Skip setup** in the
first-run model wizard. Gemma through Ollama is still the default, so the
benchmark's Gemma side needs [Ollama](https://ollama.com) and `ollama pull gemma:7b`.

## C. The engine, headless (Linux or macOS)

This is how the submission is tested, except that the tests start the engine
through the benchmark's wrapper (`llm_bench.py serve`), which only saves the
language model's inputs on the way. The engine is the same `main.py serve` the
desktop app starts.

```bash
export CLIPS_STUDIO_DATA_DIR=$PWD/judge-data
python main.py serve --port 8765 &
sleep 10
curl -s -X POST localhost:8765/queue/resume                      # a fresh queue starts paused

# Save the key (Nebius checks it first) and choose Nemotron.
printf '{"api_key": "%s"}' "$NEBIUS_API_KEY" |
  curl -s -X PUT localhost:8765/ai/providers/nebius/key -H 'content-type: application/json' -d @- > /dev/null
curl -s -X POST localhost:8765/ai/providers/nebius/test -H 'content-type: application/json' \
     -d '{"model": "nvidia/Nemotron-3_5-Lightning"}'
curl -s -X POST localhost:8765/ai/activate -H 'content-type: application/json' \
     -d '{"provider": "nebius", "model": "nvidia/Nemotron-3_5-Lightning"}' > /dev/null
# In a source checkout this writes the choice into config/settings.yaml.
# `git checkout config/settings.yaml` puts Gemma back as the default.

# Sports → Basketball → Best moments, on the 1995 game.
curl -s -X POST localhost:8765/videos/local -H 'content-type: application/json' -d "{
  \"path\": \"$PWD/belmont-salem-1995-final.mp4\",
  \"title\": \"Belmont Marauders vs Salem Witches, boys varsity basketball, 1995 (final 22 minutes)\",
  \"channel\": \"BHS Sports TV\",
  \"sport\": {\"name\": \"basketball\", \"highlights\": \"best\", \"period\": \"full\"}}"

curl -s localhost:8765/jobs/1                                    # status: queued, running, done or failed
tail -f judge-data/logs/job_1.log                                # the whole run, step by step
curl -s localhost:8765/videos | python -m json.tool | head -40   # clip_count, and why
```

The clips are written to `judge-data/clips/BHS Sports TV/<title> [local_…]/clip_*.mp4`.

## The benchmark (Gemma vs Nemotron, same evidence)

See [BENCHMARKS.md](BENCHMARKS.md#reproduce). In short:

1. Run the engine through `python scripts/hackathon/llm_bench.py serve --freeze bench/frozen -- --port 8765`
   instead of `main.py serve`.
2. Submit the game as above. Its transcript and the readings of its sound and
   picture are saved the moment the language model is first needed.
3. Replay only the language-model stages with each backend:
   `llm_bench.py replay … --backend ollama/gemma:7b` and
   `llm_bench.py replay … --backend nebius/nvidia/Nemotron-3_5-Lightning`.
4. Compare the two with `llm_bench.py compare`.

## What was tested, and where

On 2026-10-28, on a 4-core Linux container with no GPU (Python 3.11, CPU PyTorch):

- **Offline:** `pytest tests/test_nebius.py tests/test_ai_api.py tests/test_llm_adapters.py`
  passed, and so did the UI typecheck (`cd ui && npm ci && npm run typecheck`).
- **Option C with the local Gemma** (`ollama/gemma:7b`), on the 22-minute game
  above, started through `llm_bench.py serve`: ran end to end, from the file to six
  rendered 1080×1920 clips with E-BARD framing and captions, in 88.8 minutes.
  The clips are judged in [BENCHMARKS.md](BENCHMARKS.md#results).
- **Not yet run with a live Nebius key:** options A, B and C on Nemotron. See the
  status table in [BENCHMARKS.md](BENCHMARKS.md). This line is updated once
  they have run.
- **Option B on Windows** follows the app's existing from-source steps
  (README, "From source"). The steps are unchanged by this submission.
