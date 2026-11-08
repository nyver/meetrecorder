"""Тесты для transcriber.py — mock WhisperModel и pyannote."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from meeting_recorder.config import AppConfig
from meeting_recorder.transcriber import (
    Segment,
    _apply_diarization,
    _get_transcribe_model,
    _model_cache,
    _patch_hf_use_auth_token,
    _patch_speechbrain_lazy_module,
    _patch_torchaudio_compat,
    _trusted_torch_load,
    load_transcript,
    transcribe,
    unload_models,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Очищаем кэш моделей между тестами."""
    _model_cache.clear()
    yield
    _model_cache.clear()


@pytest.fixture
def app_cfg() -> AppConfig:
    return AppConfig()


def _make_wav(path: Path, duration: float = 1.0, sr: int = 48000) -> Path:
    data = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), data, sr)
    return path


def _make_fake_segment(start=0.0, end=1.0, text="Hello"):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_fake_info(language="ru", duration=10.0):
    info = MagicMock()
    info.language = language
    info.duration = duration
    return info


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


class TestSegment:
    def test_to_dict(self):
        s = Segment(start=1.5, end=2.5, speaker="SPEAKER_00", text="Hello")
        d = s.to_dict()
        assert d == {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_00", "text": "Hello"}

    def test_to_dict_rounds_timestamps(self):
        s = Segment(start=1.5555555, end=2.9999999, speaker="S", text="x")
        d = s.to_dict()
        assert d["start"] == 1.56
        assert d["end"] == 3.0


# ---------------------------------------------------------------------------
# _apply_diarization
# ---------------------------------------------------------------------------


class TestApplyDiarization:
    def test_assigns_speakers(self, app_cfg):
        segs = [
            Segment(0.0, 1.0, "UNKNOWN", "Hello"),
            Segment(1.0, 2.0, "UNKNOWN", "World"),
        ]

        # Мокируем диаризацию
        turn1 = MagicMock()
        turn1.start = 0.0
        turn1.end = 1.0
        turn2 = MagicMock()
        turn2.start = 1.0
        turn2.end = 2.0

        mock_pipeline = MagicMock()
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (turn1, None, "speaker_a"),
            (turn2, None, "speaker_b"),
        ]
        mock_pipeline.return_value = mock_diarization

        audio = Path("/tmp/fake.wav")
        result = _apply_diarization(mock_pipeline, audio, segs, app_cfg)
        speakers = {s.speaker for s in result}
        assert "SPEAKER_00" in speakers or "SPEAKER_01" in speakers
        assert "UNKNOWN" not in speakers

    def test_no_match_stays_unknown(self, app_cfg):
        segs = [Segment(5.0, 6.0, "UNKNOWN", "hello")]
        # Диаризация не покрывает [5, 6]
        turn = MagicMock()
        turn.start = 0.0
        turn.end = 1.0
        mock_pipeline = MagicMock()
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [(turn, None, "spk")]
        mock_pipeline.return_value = mock_diarization
        result = _apply_diarization(mock_pipeline, Path("/tmp/f.wav"), segs, app_cfg)
        assert result[0].speaker == "UNKNOWN"


# ---------------------------------------------------------------------------
# load_transcript
# ---------------------------------------------------------------------------


class TestLoadTranscript:
    def test_load_valid(self, tmp_path):
        data = {"session_id": "abc", "segments": []}
        f = tmp_path / "t.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_transcript(f)
        assert result["session_id"] == "abc"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transcript(tmp_path / "nonexistent.json")

    def test_load_from_str_path(self, tmp_path):
        data = {"segments": [{"text": "hi"}]}
        f = tmp_path / "t.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_transcript(str(f))
        assert result["segments"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# _get_transcribe_model (кэш)
# ---------------------------------------------------------------------------


class TestGetTranscribeModel:
    def test_model_cached(self):
        mock_model = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            m1 = _get_transcribe_model("large-v3", "cpu")
            m2 = _get_transcribe_model("large-v3", "cpu")
        assert m1 is m2  # кэш работает

    def test_different_key_different_model(self):
        m_a = MagicMock()
        m_b = MagicMock()
        side_effects = [m_a, m_b]

        def _factory(*a, **kw):
            return side_effects.pop(0)

        with patch("faster_whisper.WhisperModel", side_effect=_factory):
            r1 = _get_transcribe_model("large-v3", "cpu")
            r2 = _get_transcribe_model("medium", "cpu")
        assert r1 is not r2


# ---------------------------------------------------------------------------
# transcribe (основная функция)
# ---------------------------------------------------------------------------


class TestTranscribe:
    @pytest.mark.parametrize("language", ["auto", " AUTO ", ""])
    @pytest.mark.parametrize("batched", [False, True])
    def test_auto_language_passes_none_to_whisper(
        self, app_cfg, tmp_path, language, batched
    ):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.language = language
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"
        app_cfg.transcription.batch_size = 4 if batched else 0
        model = self._mock_model(language="en")
        batch = MagicMock()
        batch.transcribe.return_value = (
            iter([_make_fake_segment()]),
            _make_fake_info(language="en"),
        )
        with (
            patch(
                "meeting_recorder.transcriber._get_transcribe_model", return_value=model
            ),
            patch("faster_whisper.BatchedInferencePipeline", return_value=batch),
        ):
            result = transcribe(audio, app_cfg)
        backend = batch if batched else model
        assert backend.transcribe.call_args.kwargs["language"] is None
        assert backend.transcribe.call_args.kwargs["initial_prompt"] is None
        assert result["language"] == "en"

    def _mock_model(self, texts=("Hello",), language="ru", duration=5.0):
        model = MagicMock()
        fake_segs = [_make_fake_segment(i, i + 1, t) for i, t in enumerate(texts)]
        info = _make_fake_info(language=language, duration=duration)
        model.transcribe.return_value = (iter(fake_segs), info)
        return model

    def test_file_not_found(self, app_cfg, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe(tmp_path / "missing.wav", app_cfg)

    def test_basic_transcription(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"

        model = self._mock_model(["Hi there"])
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            result = transcribe(audio, app_cfg, output_path=tmp_path / "t.json")

        assert result["language"] == "ru"
        assert result["diarization_enabled"] is False
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hi there"
        assert (tmp_path / "t.json").exists()

    def test_cuda_fallback_to_cpu(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.device = "cuda"
        app_cfg.transcription.diarization = False

        model = self._mock_model(["x"])
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ) as mock_get:
            with patch("torch.cuda.is_available", return_value=False):
                transcribe(audio, app_cfg, output_path=tmp_path / "t.json")
        # При отсутствии GPU должен использоваться cpu
        mock_get.assert_called_once()
        assert mock_get.call_args[0][1] == "cpu"

    def test_with_diarization(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = True
        app_cfg.transcription.device = "cpu"
        from pydantic import SecretStr

        app_cfg.transcription.hf_token = SecretStr("fake-token")

        model = self._mock_model(["Hi", "There"])
        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        turn = MagicMock()
        turn.start, turn.end = 0.0, 2.0
        mock_diar.itertracks.return_value = [(turn, None, "spk_a")]
        mock_pipeline.return_value = mock_diar

        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            with patch(
                "meeting_recorder.transcriber._get_diarization_model",
                return_value=mock_pipeline,
            ):
                result = transcribe(audio, app_cfg, output_path=tmp_path / "t.json")

        assert result["diarization_enabled"] is True

    def test_diarization_disabled_in_config(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"

        model = self._mock_model(["Hello"])
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            result = transcribe(audio, app_cfg, output_path=tmp_path / "t.json")

        assert result["diarization_enabled"] is False

    def test_raw_speaker_labels_kept(self, app_cfg, tmp_path):
        """Имена из speaker_names не зашиваются в транскрипт — применяются при показе."""
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = True
        app_cfg.transcription.device = "cpu"
        app_cfg.transcription.speaker_names = {"SPEAKER_00": "Alice"}

        model = self._mock_model(["Hi"])
        diar = MagicMock()
        turn = MagicMock(start=0.0, end=1.0)
        diar.return_value.itertracks.return_value = [(turn, None, "SPK_A")]
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            with patch(
                "meeting_recorder.transcriber._get_diarization_model", return_value=diar
            ):
                result = transcribe(audio, app_cfg, output_path=tmp_path / "t.json")
        assert result["segments"][0]["speaker"] == "SPEAKER_00"

    def test_session_id_written(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"
        model = self._mock_model(["Hi"])
        out = tmp_path / "t.json"
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            transcribe(audio, app_cfg, output_path=out, session_id="meeting_x")
        assert json.loads(out.read_text(encoding="utf-8"))["session_id"] == "meeting_x"

    def test_initial_prompt_none_for_non_ru(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"
        app_cfg.transcription.language = "en"
        model = self._mock_model(["Hi"])
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            transcribe(audio, app_cfg, output_path=tmp_path / "t.json")
        assert model.transcribe.call_args.kwargs["initial_prompt"] is None

    def test_batched_pipeline_used_on_cuda(self, app_cfg, tmp_path):
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cuda"
        app_cfg.transcription.batch_size = 0  # auto → 16 на cuda

        model = self._mock_model(["Hello"])
        mock_batched = MagicMock()
        mock_batched_inst = MagicMock()
        fake_segs = [_make_fake_segment(0, 1, "Hello")]
        mock_batched_inst.transcribe.return_value = (iter(fake_segs), _make_fake_info())
        mock_batched.return_value = mock_batched_inst

        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("faster_whisper.BatchedInferencePipeline", mock_batched):
                    transcribe(audio, app_cfg, output_path=tmp_path / "t.json")

        mock_batched.assert_called_once_with(model=model)

    def test_output_path_auto(self, app_cfg, tmp_path):
        """Если output_path не задан — генерируется автоматически."""
        audio = _make_wav(tmp_path / "audio.wav")
        app_cfg.transcription.diarization = False
        app_cfg.transcription.device = "cpu"

        model = self._mock_model()
        with patch(
            "meeting_recorder.transcriber._get_transcribe_model", return_value=model
        ):
            transcribe(audio, app_cfg)

        expected = tmp_path / "audio_transcript.json"
        assert expected.exists()


# ---------------------------------------------------------------------------
# Monkey-patches (вызов не должен падать)
# ---------------------------------------------------------------------------


class TestCompatPatches:
    def test_patch_hf_auth_token_idempotent(self):
        _patch_hf_use_auth_token()
        _patch_hf_use_auth_token()  # второй вызов не должен упасть

    def test_trusted_torch_load_scoped(self):
        """weights_only=False только внутри блока; снаружи torch.load исходный."""
        import torch

        original = torch.load
        calls = []

        def fake_load(*args, **kwargs):
            calls.append(kwargs.copy())
            return object()

        torch.load = fake_load
        try:
            with _trusted_torch_load():
                torch.load("model.pt", weights_only=True)
                torch.load("model.pt")
            assert torch.load is fake_load  # восстановлен после выхода
            torch.load("model.pt")
        finally:
            torch.load = original

        assert calls[0]["weights_only"] is True  # явное значение сохраняется
        assert calls[1]["weights_only"] is False
        assert "weights_only" not in calls[2]  # вне блока — умолчание torch

    def test_trusted_torch_load_restores_on_error(self):
        import torch

        original = torch.load
        with pytest.raises(RuntimeError):
            with _trusted_torch_load():
                raise RuntimeError("boom")
        assert torch.load is original

    def test_patch_speechbrain_lazy_module_no_speechbrain(self):
        with patch.dict(
            "sys.modules",
            {
                "speechbrain": None,
                "speechbrain.utils": None,
                "speechbrain.utils.importutils": None,
            },
        ):
            _patch_speechbrain_lazy_module()  # не должна упасть при отсутствии модуля

    def test_patch_torchaudio_compat_idempotent(self):
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            _patch_torchaudio_compat()
        _patch_torchaudio_compat()  # повторный вызов — нет-оп


class TestUnloadModels:
    def test_clears_cache(self):
        _model_cache["dummy"] = object()
        unload_models()
        assert "dummy" not in _model_cache

    def test_noop_on_empty_cache(self):
        _model_cache.clear()
        unload_models()  # не должно падать
        assert _model_cache == {}


# ---------------------------------------------------------------------------
# OOM-фолбэк, декодирование, сопоставление диаризации
# ---------------------------------------------------------------------------


class TestOomFallback:
    def test_batch_fallbacks(self):
        from meeting_recorder.transcriber import _batch_fallbacks

        assert _batch_fallbacks(16) == [16, 8, 4, 0]
        assert _batch_fallbacks(2) == [2, 0]

    def test_retries_with_smaller_batch(self):
        from meeting_recorder import transcriber as tr

        calls = []

        def fake_pass(model, audio, size, kwargs):
            calls.append(size)
            if size > 4:
                raise RuntimeError("CUDA failed with error out of memory")
            return ["seg"], "info"

        with patch.object(tr, "_whisper_pass", side_effect=fake_pass):
            with patch.object(tr, "_free_gpu_memory") as free:
                segs, info = tr._transcribe_with_oom_fallback(None, None, 16, {})
        assert calls == [16, 8, 4]
        assert segs == ["seg"]
        assert free.call_count == 2

    def test_non_oom_error_not_retried(self):
        from meeting_recorder import transcriber as tr

        with patch.object(
            tr, "_whisper_pass", side_effect=ValueError("bad audio")
        ) as m:
            with pytest.raises(ValueError):
                tr._transcribe_with_oom_fallback(None, None, 16, {})
        assert m.call_count == 1

    def test_oom_on_last_attempt_raises(self):
        from meeting_recorder import transcriber as tr

        with patch.object(
            tr, "_whisper_pass", side_effect=RuntimeError("out of memory")
        ):
            with patch.object(tr, "_free_gpu_memory"):
                with pytest.raises(RuntimeError):
                    tr._transcribe_with_oom_fallback(None, None, 8, {})

    def test_whisper_pass_consumes_generator(self):
        """Ошибки генератора faster-whisper возникают внутри _whisper_pass."""
        from meeting_recorder.transcriber import _whisper_pass

        def gen():
            yield "a"
            raise RuntimeError("out of memory")

        model = MagicMock()
        model.transcribe.return_value = (gen(), "info")
        with pytest.raises(RuntimeError, match="out of memory"):
            _whisper_pass(model, "x.wav", 0, {})


class TestDecodeAudio:
    def test_decode_real_wav(self, tmp_path):
        from meeting_recorder.transcriber import _decode_audio

        audio = _decode_audio(_make_wav(tmp_path / "a.wav"))
        assert audio is not None and audio.dtype.name == "float32"

    def test_decode_failure_returns_none(self, tmp_path):
        from meeting_recorder.transcriber import _decode_audio

        assert _decode_audio(tmp_path / "missing.wav") is None

    def test_waveform_passed_to_pyannote(self, app_cfg):
        import numpy as np

        segs = [Segment(0.0, 1.0, "UNKNOWN", "a")]
        pipeline = MagicMock()
        pipeline.return_value.itertracks.return_value = []
        _apply_diarization(
            pipeline, Path("x.wav"), segs, app_cfg, waveform=np.zeros(16000, "float32")
        )
        arg = pipeline.call_args.args[0]
        assert arg["sample_rate"] == 16000
        assert tuple(arg["waveform"].shape) == (1, 16000)


class TestSpeakerOverlaps:
    def _brute(self, segments, turns):
        result = []
        for seg in segments:
            acc = {}
            for ts, te, sp in turns:
                ov = min(seg.end, te) - max(seg.start, ts)
                if ov > 0:
                    acc[sp] = acc.get(sp, 0.0) + ov
            result.append(acc)
        return result

    def test_matches_bruteforce_with_overlapping_turns(self):
        import random

        from meeting_recorder.transcriber import _speaker_overlaps

        rnd = random.Random(1)
        segments = []
        t = 0.0
        for _ in range(200):
            d = rnd.uniform(0.5, 8)
            segments.append(Segment(t, t + d, "UNKNOWN", ""))
            t += rnd.uniform(0.2, 6)  # сегменты могут перекрываться
        turns = []
        for _ in range(300):
            st = rnd.uniform(0, t)
            turns.append((st, st + rnd.uniform(0.1, 30), rnd.choice("ABC")))
        turns.sort(key=lambda x: x[0])
        fast = _speaker_overlaps(segments, turns)
        slow = self._brute(segments, turns)
        for f, sl in zip(fast, slow, strict=True):
            assert f.keys() == sl.keys()
            for k in f:
                assert f[k] == pytest.approx(sl[k])
