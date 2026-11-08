# Meeting Recorder

Screen and audio recorder with speaker diarization, transcription, LLM-generated
summary, highlights, and an interactive HTML/web viewer.
Controlled via CLI or a system tray icon.

## Requirements

- **OS:** Windows 10/11 x64
- **Python:** 3.11+
- **ffmpeg:** must be in `PATH`
- **GPU (optional):** NVIDIA with CUDA for faster-whisper / pyannote and local LLM

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

Edit `config.yaml` (full annotated example: `config.example.yaml`).
Sensitive values can be set via environment variables:

```bash
set HF_TOKEN=hf_xxxxxxxx       # HuggingFace token for pyannote diarization
set LLM_API_KEY=sk-or-vx-xxxx  # LLM key (if backend: openrouter)
```

Secrets taken from environment variables are **never written** to `config.yaml`
(neither by `mrec generate-config` nor by the tray device menu).

Config file location: `config.yaml` next to the project (editable install), otherwise
`%APPDATA%\MeetingRecorder\config.yaml`; override with `mrec -c path\to\config.yaml …`.

Key parameters:

```yaml
output_dir: "C:/Meetings"

recording:
  screen_grabber: "gdigrab"     # gdigrab (compatible) or ddagrab (faster, ffmpeg ≥ 6.1)
  mic_device: ""                # dshow device name; "" = first capture device
  system_audio_grabber: "soundcard"  # soundcard (recommended) / wasapi / dshow
  system_audio_speaker: ""      # soundcard: output device id or name; "" = default
  record_system_audio: true
  auto_mux: true                # build *_final.mp4 (video + audio) during report

transcription:
  model: "large-v3-turbo"       # large-v3-turbo (recommended) / large-v3 / medium
  device: "cuda"                # cuda or cpu
  diarization: true
  hf_token: ""                  # HuggingFace token (or HF_TOKEN env var)
  keep_models_loaded: false     # false = free GPU memory after each processing
  speaker_names:                # applied when rendering; change and run `mrec report`
    SPEAKER_00: "Alice"
    SPEAKER_01: "Bob"

llm:
  backend: "local"              # local (llama-server) or openrouter
  base_url: "http://127.0.0.1:8080/v1"
  model: "my-model-alias"
  context_tokens: 32768         # model context window (input + output), tokens
  max_tokens: 16384             # increase for thinking models (DeepSeek-R1 etc.)
  max_retries: 2                # retries on 429 / 5xx / network errors
  timeout: 600
```

`context_tokens` must match the real context of the model (e.g. `--ctx-size` of
llama-server). Long meetings are split into parts that fit the window: summary and
highlights use map-reduce, protocol cleaning goes part by part, chat trims the
transcript and the dialogue history.

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
| 🎚️ Recording devices → | Pick microphone and output device (saved to config) |
| 📋 Meetings → | Per-session submenu (see below) |
| 📂 Open meetings folder | Open the root folder of all sessions |
| 📁 Open last session folder | Open the folder of the last (or current) session |

Menu actions are ignored while another operation is running (no double start).

#### Recording devices submenu

- **🎙️ Microphone** — "Auto-detect" or any dshow capture device.
- **🔊 Output device (system audio)** — for `soundcard`: default output device or a
  specific one (stored by device id, so identical headsets are distinguished); for
  `dshow`: the capture device (virtual-audio-capturer, Stereo Mix…); not available
  for `wasapi`.
- **🔄 Refresh device list** — rescan immediately. The list is also rescanned every
  10 s while idle; the menu is rebuilt only when devices actually change.

The selection is written only to the corresponding key of the config file that the
tray was started with (`mrec -c …  tray`); other keys and secrets stay untouched.

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

The dashboard has no authentication. By default it listens on `127.0.0.1` and accepts
only local `Host` headers (DNS-rebinding protection); `--host 0.0.0.0` exposes all
meetings to the network.

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

Interactive REPL that loads the transcript, summary and key moments into the LLM
context. Ask questions in a multi-turn dialogue — conversation history is kept
for the session. If the transcript does not fit `llm.context_tokens`, it is cut
(the model is told where) and the oldest turns of the dialogue are dropped.
Exit with `exit`, `quit`, or Ctrl+C.

### Programmatic API

```python
from meeting_recorder.config import load_config
from meeting_recorder.naming import resolve_session
from meeting_recorder.pipeline import (
    finalize_recording,
    process_session,
    run_highlights_only,
    run_html,
    run_report_only,
    run_transcribe_only,
)

cfg = load_config()
session_id = "meeting_2026-06-10_09-46-18"

transcript = run_transcribe_only(cfg, session_id)
protocol, summary = run_report_only(cfg, session_id)
highlights_path = run_highlights_only(cfg, session_id)
html_path = run_html(cfg, session_id)

# The same steps the CLI and the tray use after a recording stops:
paths = resolve_session(cfg.output_dir, session_id)
finalize_recording(cfg, paths)  # mic + system → *_mix.wav
process_session(cfg, paths)  # transcription + all reports
```

## Session Artifacts

Each session is stored in `output_dir/meeting_YYYY-MM-DD_HH-MM-SS/`:

| File | Description |
|------|-------------|
| `*.mp4` | Screen recording (no audio) |
| `*_mic.wav` | Microphone recording |
| `*_system.wav` | System audio recording |
| `*_mix.wav` | Mixed audio (mic + system, end-aligned) |
| `*_final.mp4` | Final video with audio (`mux` or `recording.auto_mux`) |
| `*_transcript.json` | Transcript with diarization (raw `SPEAKER_XX` labels) |
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
├── transcriber.py  # faster-whisper transcription + pyannote diarization
├── llm_client.py   # OpenAI-compatible HTTP client (local / openrouter)
├── report.py       # Protocol + summary + highlights (map-reduce for long meetings)
├── html_report.py  # Static HTML protocol with embedded media player
├── dashboard.py    # FastAPI web dashboard (session list + detail view)
├── pipeline.py     # Pipeline orchestration (finalize_recording, process_session, run_*)
├── session_utils.py # Shared artifact helpers (atomic writes, transcript, highlights)
├── tray.py         # System tray icon (pystray + Pillow)
├── templates/      # Jinja2 HTML templates for the web dashboard
└── prompts/        # LLM prompt templates (summary, highlights, protocol_clean)
```

## Pipeline

```
Recording (ffmpeg + soundcard)
  → finalize_recording: mix_audio_files (end-aligned, averaged)
  → Transcription (faster-whisper + pyannote diarization)
  → Protocol (Markdown, verbatim with timestamps)
  → Summary (LLM, map-reduce for long meetings)
  → Highlights (LLM, 5 key moments; map-reduce for long meetings)
  → [recording.auto_mux] mux_video (final MP4)
  → HTML Protocol (interactive player + searchable transcript)
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

## Logs and state

- Log: `%APPDATA%\MeetingRecorder\logs\meeting_recorder.log` (rotated, 5 MB × 3);
  override with `MEETING_RECORDER_LOG_FILE`.
- Recording state (`mrec start` / `mrec stop`): `%APPDATA%\MeetingRecorder\.state`;
  override with `MEETING_RECORDER_STATE_DIR`. A running `mrec start` holds a lock
  file, so a state left after a crash is detected and cleaned up automatically.

## Development

```bash
pip install -e ".[tray,web,dev]"
pytest -v --cov            # coverage must stay above 70 %
ruff check . && ruff format --check .
```

Tests never touch the network or your `%APPDATA%` (see `tests/conftest.py`).
