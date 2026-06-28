# Meeting Recorder

Screen and audio recorder with speaker diarization, transcription, LLM-generated
summary, highlights, and an interactive HTML/web viewer.
Controlled via CLI or a system tray icon.

## Requirements

- **OS:** Windows 10/11 x64
- **Python:** 3.11+
- **ffmpeg:** must be in `PATH`
- **GPU (optional):** NVIDIA with CUDA for WhisperX and local LLM

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install with tray + web dashboard support
pip install -e ".[tray,web]"

# 3. Verify ffmpeg is in PATH
ffmpeg -version
```

> Audio resampling (if sample rates differ): `pip install -e ".[audio]"`

## Configuration

```bash
# Generate a config template
mrec generate-config
```

Edit `config.yaml`. Sensitive values can be set via environment variables:

```bash
set HF_TOKEN=hf_xxxxxxxx       # HuggingFace token for pyannote diarization
set LLM_API_KEY=sk-or-vx-xxxx  # LLM key (if backend: openrouter)
```

Key parameters:

```yaml
output_dir: "C:/Meetings"

recording:
  screen_grabber: "gdigrab"     # gdigrab (compatible) or ddagrab (faster, ffmpeg ≥ 6.1)
  mic_device: "Microphone ..."  # dshow device name
  system_audio_grabber: "soundcard"  # soundcard (recommended) / wasapi / dshow
  record_system_audio: true

transcription:
  model: "large-v3-turbo"       # large-v3-turbo (recommended) / large-v3 / medium
  device: "cuda"                # cuda or cpu
  diarization: true
  hf_token: ""                  # HuggingFace token (or HF_TOKEN env var)
  speaker_names:
    SPEAKER_00: "Alice"
    SPEAKER_01: "Bob"

llm:
  backend: "local"              # local (llama-server) or openrouter
  base_url: "http://127.0.0.1:8080/v1"
  model: "my-model-alias"
  max_tokens: 16384             # increase for thinking models (DeepSeek-R1 etc.)
  timeout: 600
```

### HuggingFace token (diarization)

1. Create a token at <https://huggingface.co/settings/tokens>
2. Accept the license for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`
3. Set `hf_token` in `config.yaml` or `HF_TOKEN` env var

## Usage

### System Tray (recommended)

```bash
mrec tray
```

The icon reflects the current state and shows elapsed time:

| Color | State |
|-------|-------|
| Gray | Idle |
| Red (blinking) | Recording |
| Orange | Processing (transcription / report / mux) |
| Purple | Error |

Tray menu:

| Item | Description |
|------|-------------|
| ▶ Start recording | Start screen and audio capture |
| ⏹ Stop recording | Stop and run the full pipeline |
| ⏹ Stop without processing | Stop and mix audio, skip transcription |
| 💬 Chat about meeting | Open a terminal with `mrec chat` for the last session |
| ⚙️ Transcribe + report | Process the last unprocessed session |
| 📝 Regenerate report | Regenerate protocol and summary for the last session |
| 🎬 Mux video + audio | Merge video and mix audio into a final MP4 |
| 🌐 HTML protocol | Generate interactive HTML protocol and open in browser |
| ⭐ Key moments | Generate 5 highlights with timecodes via LLM |
| 📋 Meetings → | Per-session submenu (see below) |
| 📂 Open meetings folder | Open the root folder of all sessions |
| 📁 Open last session folder | Open the folder of the last (or current) session |

#### Meetings submenu

**📋 Meetings** shows the last 25 sessions (newest first). Each session entry
opens a submenu displaying a 3–4 line summary (if available) and individual
artifact regeneration actions:

| Item | Prerequisite |
|------|-------------|
| ⚙️ Transcribe + report | mix audio exists |
| 📝 Regenerate report | transcript exists |
| ⭐ Key moments | transcript exists |
| 🌐 HTML protocol | transcript exists |
| 🎬 Mux video + audio | video + mix audio exist |

### CLI

```bash
# Start recording
mrec start

# Stop recording + full pipeline (transcription, protocol, summary, highlights)
mrec stop

# Stop recording without processing (mix audio only)
mrec stop-only

# Full pipeline for an existing session (transcription + all reports)
mrec process <session_id>

# Regenerate protocol and summary only
mrec report <session_id>

# Generate 5 key highlights with timecodes via LLM
mrec highlights [session_id]

# Generate interactive HTML protocol and open in browser
mrec html [session_id]

# Merge mix audio with video into a final MP4
mrec mux [session_id]

# Interactive LLM chat about a meeting
mrec chat [session_id]

# List all sessions
mrec list

# Launch web dashboard
mrec serve

# Launch system tray mode
mrec tray

# Generate a config template
mrec generate-config
```

All commands that accept `session_id` default to the last eligible session when omitted.

### Web Dashboard

```bash
mrec serve                          # opens http://127.0.0.1:7070
mrec serve --port 8080 --no-open    # custom port, no auto-open
```

The dashboard provides a browser UI for browsing and replaying meetings:

- **Sessions list** — all sessions with status badge, duration, and speaker list
- **Session detail page:**
  - Video or audio player (streams directly from disk)
  - **📄 Summary** — LLM-generated structured report
  - **⭐ Key moments** — 5 highlights; click ▶ MM:SS to jump to that point in the video
  - **📋 Protocol** — full verbatim protocol
  - **🗒 Transcript** — searchable full transcript; click any line to seek the player

### Meeting Chat

```bash
mrec chat meeting_2026-06-10_09-46-18
```

Interactive REPL that loads the transcript, protocol, and summary into the LLM
context. Ask questions in a multi-turn dialogue — conversation history is kept
for the session. Exit with `exit`, `quit`, or Ctrl+C.

### Programmatic API

```python
from meeting_recorder.config import load_config
from meeting_recorder.pipeline import (
    run_transcribe_only,
    run_report_only,
    run_highlights_only,
    run_html,
)

cfg = load_config()
session_id = "meeting_2026-06-10_09-46-18"

transcript = run_transcribe_only(cfg, session_id)
protocol, summary = run_report_only(cfg, session_id)
highlights_path = run_highlights_only(cfg, session_id)
html_path = run_html(cfg, session_id)
```

## Session Artifacts

Each session is stored in `output_dir/meeting_YYYY-MM-DD_HH-MM-SS/`:

| File | Description |
|------|-------------|
| `*.mp4` | Screen recording (no audio) |
| `*_mic.wav` | Microphone recording |
| `*_system.wav` | System audio recording |
| `*_mix.wav` | Mixed audio (mic + system, end-aligned) |
| `*_final.mp4` | Final video with audio (after `mux`) |
| `*_transcript.json` | Transcript with speaker diarization |
| `*_protocol.md` | Meeting protocol (Markdown) |
| `*_summary.md` | LLM-generated summary report (Markdown) |
| `*_highlights.json` | 5 key moments with timecodes (JSON) |
| `*_protocol.html` | Interactive HTML protocol with media player |
| `*_ffmpeg.log` | ffmpeg log (for diagnostics) |

## Architecture

```
meeting_recorder/
├── __main__.py     # CLI (click): start, stop, stop-only, process, report,
│                   #   highlights, html, serve, mux, chat, tray, list,
│                   #   generate-config
├── config.py       # Pydantic config (YAML)
├── naming.py       # session_id and artifact paths (SessionPaths)
├── recorder.py     # ffmpeg: video + audio capture; mix_audio_files; mux_video
├── transcriber.py  # WhisperX: transcription + pyannote diarization
├── llm_client.py   # OpenAI-compatible HTTP client (local / openrouter)
├── report.py       # Protocol + summary + highlights (map-reduce for long meetings)
├── html_report.py  # Static HTML protocol with embedded media player
├── dashboard.py    # FastAPI web dashboard (session list + detail view)
├── pipeline.py     # Pipeline orchestration (run_*, run_highlights_only, run_html)
├── tray.py         # System tray icon (pystray + Pillow)
├── templates/      # Jinja2 HTML templates for the web dashboard
└── prompts/        # LLM prompt templates (summary, highlights, protocol_clean)
```

## Pipeline

```
Recording (ffmpeg + soundcard)
  → mix_audio_files (end-aligned, averaged)
  → Transcription (WhisperX + pyannote diarization)
  → Protocol (Markdown, verbatim with timestamps)
  → Summary (LLM, map-reduce for long meetings)
  → Highlights (LLM, 5 key moments with timecodes)
  → HTML Protocol (interactive player + searchable transcript)
  → [optional] mux_video (final MP4)
```

Each stage can be run independently using a `session_id`.

## LLM Backends

| Parameter | local (llama-server) | openrouter |
|-----------|----------------------|------------|
| `base_url` | `http://127.0.0.1:8080/v1` | `https://openrouter.ai/api/v1` |
| `api_key` | (empty) | your OpenRouter key |
| `model` | your alias | e.g. `openai/gpt-4o-mini` |
| `timeout` | 300–600 s | 60–120 s |

Start a local llama-server:

```bash
llama-server -m your-model.gguf --host 127.0.0.1 --port 8080 --ctx-size 32768
```

## Privacy

- **`local`** — all data (video, audio, transcript) stays on your machine.
- **`openrouter`** — the transcript is sent to the OpenRouter cloud. The application logs a warning on every LLM call in this mode.

## Tests

```bash
pytest -v
```
