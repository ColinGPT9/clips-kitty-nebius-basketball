# Three-minute demo script

Target length is 2:50, which leaves ten seconds of margin under the three-minute
limit. Record it on the Windows PC in the desktop app, from this repository. Upload
it to YouTube as **Public** and paste the link into the Devpost form.

**Footage.** Use the public-domain 1995 Belmont vs Salem game:
[archive.org/details/basketball95](https://archive.org/details/basketball95),
marked Public Domain Mark 1.0 by Belmont Media Center. A 22-minute cut of the
final stretch (`-ss 3510`, see [JUDGE_SETUP.md](JUDGE_SETUP.md)) is enough. Do
not use NBA, House of Highlights or other broadcast footage, or the 2022
Internet Archive games: those are CC BY-NC-ND, and NoDerivatives rules out
publishing cut highlights. The rules also forbid copyrighted music, so use no
background music, or only a track you have the rights to.

**Before you record:**

- Process the same 22-minute file once with Nebius selected, so the clips
  already exist. The full run takes several minutes, so the 0:55 segment shows
  it starting and then cuts to the finished job.
- Connect one social account in WoopSocial that you are happy to post a public
  domain clip to.
- Hide your API keys. The key field only ever shows `••••` and the last four
  characters, but keep the Nebius console's key page off screen.

| Time | Screen | Say (about 2.5 words a second) |
|---|---|---|
| **0:00–0:25** Introduction | The Clips Kitty home screen, then the GitHub page of the original repository | "I'm Colin, a Raptors fan. I built Clips Kitty, a free, open-source AI clipping app for small creators who can't pay for Opus Clip or an editor. It already has a Basketball mode. For this hackathon I connected its language model to NVIDIA Nemotron, running on Nebius Token Factory." |
| **0:25–0:55** Nemotron on Nebius | **Settings → AI → Direct provider API → Nebius Token Factory**: paste the key, choose **Nemotron 3.5 Lightning**, press **Test connection** ("Connected to Nebius Token Factory. nvidia/Nemotron-3_5-Lightning is available."). Then a terminal running `python scripts/hackathon/nebius_check.py` | "Nebius is just another provider in the existing AI settings, on my own key. The model list comes live from Nebius. This is one real Nemotron call, made the way the app makes it during a job: real commentary in, structured clips back, with the time and the tokens Nebius reported." |
| **0:55–1:40** Basketball | **Add videos → Sports → Basketball → Best moments**, add the 1995 game, press Start. Show the job log scrolling: "Transcribing", "Game sounds (PANNs)", "Chunk 1/5: … candidate(s)", then the **AI usage** line naming `nebius/nvidia/Nemotron-3_5-Lightning` | "Nothing in the basketball pipeline changed. Whisper writes the commentary with word timings. PANNs listens for the crowd and the buzzer. YOLO and OpenCV read the picture. Nemotron does the reasoning the language model always did: it scores the commentary and the moments, reranks the finalists and writes the titles. The log shows Nebius' own token count for this game." |
| **1:40–2:15** Highlights | The finished clips in the gallery. Play two of them, including the final possession and the court storm. Then open one in the **Clip Editor** and nudge the title or a caption | "These are the clips it picked from a 1995 high-school game. The 9:16 crop follows the ball and the players. It uses the E-BARD basketball detector, which feeds Clips Kitty's own tracker. The captions come from Whisper. Every clip can still be edited before anything is posted." |
| **2:15–2:35** Publishing | **Publish → WoopSocial**: choose the connected account, check the title, and press **Publish**. Show the clip's post status | "Publishing goes through WoopSocial, which Clips Kitty already supports, on my own connected account. Nothing is posted until I press Publish." |
| **2:35–2:50** Open source | The `clips-kitty-nebius-basketball` repository: README, `llm/providers/nebius.py`, `docs/hackathon/BENCHMARKS.md` | "The hackathon edition is public and AGPL-licensed. The Nebius provider is one small file in the existing provider architecture, so the next model, Meta's included, plugs in the same way. The benchmark compares Gemma and Nemotron on the same game." |

## Things not to claim on camera

- **Do not say Nemotron watches the video.** No cloud model is ever sent the picture: only
  transcripts, timings and text notes (`LLMBackend.sees_images` is false for
  every cloud backend). The vision relabelling of reactions (bench, courtside,
  coach) runs only with a local vision model. With Nemotron the reactions keep
  the labels the detectors and commentary gave them.
- **Do not say E-BARD tracks the players.** E-BARD's YOLOv8n weights detect the
  ball, hoop, players and referees in each frame. Clips Kitty's own code in
  `sports/basketball/crop/` turns those detections into tracks and a crop.
- **Do not say anything was trained or fine-tuned for the hackathon.** Nothing was.
- **Only quote benchmark numbers that appear in [BENCHMARKS.md](BENCHMARKS.md).**
