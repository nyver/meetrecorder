"""Shared helpers for session artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .naming import SessionPaths


TranscriptSource = dict[str, Any] | Path | str


def load_transcript_data(transcript: TranscriptSource) -> dict[str, Any]:
    """Return transcript data from an in-memory dict or a JSON file path."""
    if isinstance(transcript, dict):
        return transcript

    path = Path(transcript)
    if not path.exists():
        raise FileNotFoundError(f"Транскрипт не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_transcript_data(path: Path, data: dict[str, Any]) -> None:
    """Write transcript JSON with the project's standard formatting."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_segments(
    segments: Iterable[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Copy transcript segments and apply configured speaker aliases."""
    aliases = speaker_names or {}
    normalized: list[dict[str, Any]] = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        normalized.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": aliases.get(speaker, speaker),
                "text": seg.get("text", "").strip(),
            }
        )
    return normalized


def transcript_lines(
    segments: Iterable[dict[str, Any]],
    *,
    speaker_names: dict[str, str] | None = None,
    timestamp_formatter,
) -> list[str]:
    """Format transcript segments as timestamped plain-text lines."""
    return [
        f"[{timestamp_formatter(seg['start'])}] {seg['speaker']}: {seg['text']}"
        for seg in normalize_segments(segments, speaker_names)
    ]


def unique_speakers(
    segments: Iterable[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> list[str]:
    """Return sorted speaker names after aliasing."""
    return sorted({seg["speaker"] for seg in normalize_segments(segments, speaker_names)})


def pick_media(paths: SessionPaths, *, mime: bool = True) -> tuple[str, str] | None:
    """Choose the best available media file for a session."""
    if paths.final_video.exists():
        return paths.final_video.name, "video/mp4" if mime else "video"
    if paths.mix_audio.exists():
        return paths.mix_audio.name, "audio/wav" if mime else "audio"
    if paths.video.exists():
        return paths.video.name, "video/mp4" if mime else "video"
    return None


def strip_code_fences(text: str) -> str:
    """Remove wrapping Markdown code fences commonly returned by LLMs."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            line for line in stripped.splitlines() if not line.startswith("```")
        ).strip()
    return stripped
