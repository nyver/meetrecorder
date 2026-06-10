# Meeting Recorder

Screen and audio recorder with speaker diarization, transcription, meeting protocol and LLM-generated summary.  
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

# 2. Install dependencies (including tray icon)
pip install -e ".[tray]"

# 3. Verify ffmpeg is in PATH
ffmpeg -version
```

> For audio resampling when sample rates differ: `pip install -e ".[audio]"`

## Configuration

```bash
# Generate a config template
mrec generate-config
```

Override secrets via environment variables:

```bash
set HF_TOKEN=hf_xxxxxxxx       # HuggingFace token for diarization
set LLM_API_KEY=sk-or-vx-xxxx  # LLM key (if backend=openrouter)
```

Key `config.yaml` parameters:

```yaml
recording:
  screen_grabber: "ddagrab"   # ddagrab (recommended) or gdigrab
  mic_device: "Microphone name (dshow)"
  record_system_audio: true

llm:
  backend: "local"            # local or openrouter
  timeout: 600                # LLM request timeout in seconds
  max_tokens: 16384
```

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
| 📝 Regenerate report | Regenerate protocol and summary |
| 🎬 Mux video + audio | Merge video and mix audio into a final MP4 |
| 📂 Open meetings folder | Open the root folder of all sessions |
| 📁 Open last session folder | Open the folder of the last (or current) session |

### CLI

```bash
# Start recording
mrec start

# Stop recording + automatic transcription and report generation
mrec stop

# Stop recording without transcription (process later)
mrec stop-only

# Transcription + report for an existing session
mrec process <session_id>

# Regenerate protocol and summary only
mrec report <session_id>

# Merge mix audio with video into a final MP4
mrec mux [session_id]

# Interactive LLM chat about a meeting
mrec chat [session_id]

# List all sessions
mrec list

# Launch system tray mode
mrec tray
```

All commands that accept `session_id` default to the last session when omitted.

### Meeting Chat

```bash
mrec chat meeting_2026-06-10_09-46-18
```

Interactive REPL: loads the transcript, protocol and summary into the LLM context.
Ask questions in a multi-turn dialogue — conversation history is kept for the duration of the chat session.
Exit with `exit`, `quit`, or Ctrl+C.

### Programmatic API

```python
from meeting_recorder.config import load_config
from meeting_recorder.pipeline import run_transcribe_only, run_report_only

cfg = load_config()
transcript = run_transcribe_only(cfg, "meeting_2026-06-10_09-46-18")
protocol, summary = run_report_only(cfg, "meeting_2026-06-10_09-46-18")
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
| `*_transcript.json` | Transcript with diarization |
| `*_protocol.md` | Meeting protocol (Markdown) |
| `*_summary.md` | LLM-generated summary report |
| `*_ffmpeg.log` | ffmpeg log (for diagnostics) |

## Architecture

```
meeting_recorder/
├── __main__.py     # CLI (click): start, stop, stop-only, process,
│                   #   report, mux, chat, tray, list, generate-config
├── config.py       # pydantic config (yaml)
├── naming.py       # session_id and artifact paths
├── recorder.py     # ffmpeg: video + audio; mix_audio_files; mux_video
├── transcriber.py  # WhisperX: transcription + diarization
├── llm_client.py   # OpenAI-compatible HTTP client
├── report.py       # Protocol + summary (chunking / map-reduce)
├── pipeline.py     # Pipeline orchestration
├── tray.py         # System tray icon (pystray + Pillow)
└── prompts/        # LLM prompt templates
```

## Pipeline

```
Recording (ffmpeg + soundcard)
  → mix_audio_files (end-aligned, averaged)
  → Transcription (WhisperX + pyannote)
  → Protocol (Markdown)
  → Summary (LLM, map-reduce for long meetings)
  → [optional] mux_video (final MP4)
```

Each stage can be run independently using a `session_id`.

## LLM Backends

| Parameter | local (llama-server) | openrouter |
|-----------|----------------------|------------|
| `base_url` | `http://127.0.0.1:8080/v1` | `https://openrouter.ai/api/v1` |
| `api_key` | (empty) | your OpenRouter key |
| `model` | your model | e.g. `openai/gpt-oss-120b:free` |
| `timeout` | 300–600 s | 60–120 s |

Start a local llama-server:

```bash
llama-server -m your-model.gguf --host 127.0.0.1 --port 8080
```

## Privacy

- `local` — all data (video, audio, transcript) stays on your machine.
- `openrouter` — the transcript is sent to the OpenRouter cloud. The application warns about this in the logs.

## Tests

```bash
pytest -v
```
