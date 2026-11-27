"""Client bridge for offloading Harmony Auralis AI models to Modal.com Serverless GPU (L4 24GB).

Reads MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, and USE_MODAL_GPU from `music-backend/.env` (or OS env).
When enabled and deployed, heavy AI models run on Modal NVIDIA L4 GPUs.
An explicitly enabled Modal configuration must be usable; Songscription reports
configuration or remote failures instead of starting models on the local machine.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

APP_NAME = "harmony-auralis-gpu"

_ENV_LOCK = threading.Lock()
_ENV_LOADED = False


class ModalConfigurationError(RuntimeError):
    """Modal was requested but the client configuration is incomplete."""


def _load_backend_dotenv() -> None:
    """Load `music-backend/.env` into `os.environ` (only setting keys not already present)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.is_file():
            try:
                for raw in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            except Exception:
                pass
        _ENV_LOADED = True


def is_modal_enabled() -> bool:
    """Resolve remote execution without silently disabling an explicit request.

    Unconfigured installations keep their existing local mode. Setting
    USE_MODAL_GPU=true requires credentials and the Modal SDK, even if missing.
    """
    _load_backend_dotenv()
    configured_flag = os.environ.get("USE_MODAL_GPU", "").strip().lower()
    flag = configured_flag or "true"
    if flag in ("0", "false", "no", "off"):
        return False
    if not (os.environ.get("MODAL_TOKEN_ID", "").strip() and os.environ.get("MODAL_TOKEN_SECRET", "").strip()):
        if configured_flag:
            raise ModalConfigurationError(
                "USE_MODAL_GPU đang bật nhưng thiếu MODAL_TOKEN_ID hoặc MODAL_TOKEN_SECRET trong music-backend/.env."
            )
        return False
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise ModalConfigurationError(
            "USE_MODAL_GPU đang bật nhưng Python chạy backend chưa có Modal SDK. "
            "Cài bằng python -m pip install modal trong môi trường backend."
        ) from exc
    return True


def _lyrics_worker():
    _load_backend_dotenv()
    import modal

    return modal.Cls.from_name(APP_NAME, "LyricsGPUWorker")()


def _cover_worker():
    _load_backend_dotenv()
    import modal

    return modal.Cls.from_name(APP_NAME, "CoverGPUWorker")()


def modal_extract_lyrics(
    *,
    source_audio: Path,
    vocal_wav_path: Path | None,
    mode: str,
    user_lyrics_text: str,
    requested_language: str,
    model_id: str,
    isolate_vocals: bool,
    chords: list[tuple[float, str]] | None,
    audio_sha1: str | None,
    track_title: str = "",
) -> dict[str, Any]:
    audio_bytes = source_audio.read_bytes()
    vocal_bytes = vocal_wav_path.read_bytes() if (vocal_wav_path and vocal_wav_path.is_file()) else None
    chord_list = [[float(t), str(c)] for t, c in chords] if chords else None
    return _lyrics_worker().extract_lyrics.remote(
        audio_bytes=audio_bytes,
        ext=source_audio.suffix.lower() or ".mp3",
        vocal_wav_bytes=vocal_bytes,
        mode=mode,
        user_lyrics_text=user_lyrics_text,
        requested_language=requested_language,
        model_id=model_id,
        isolate_vocals=isolate_vocals,
        chords=chord_list,
        audio_sha1=audio_sha1,
        track_title=track_title,
    )


def modal_retranscribe_flagged(
    *,
    source_audio: Path,
    vocal_wav_path: Path | None,
    segments: list[dict],
    target_ids: list[int] | None,
    language: str | None,
    model_id: str,
    isolate_vocals: bool,
) -> dict[str, Any]:
    audio_bytes = source_audio.read_bytes()
    vocal_bytes = vocal_wav_path.read_bytes() if (vocal_wav_path and vocal_wav_path.is_file()) else None
    return _lyrics_worker().retranscribe_flagged.remote(
        audio_bytes=audio_bytes,
        ext=source_audio.suffix.lower() or ".mp3",
        vocal_wav_bytes=vocal_bytes,
        segments=segments,
        target_ids=target_ids,
        language=language,
        model_id=model_id,
        isolate_vocals=isolate_vocals,
    )


def modal_separate_instrumental(
    *,
    source_audio: Path,
) -> dict[str, Any]:
    return _lyrics_worker().separate_instrumental.remote(
        audio_bytes=source_audio.read_bytes(),
        ext=source_audio.suffix.lower() or ".mp3",
    )


def modal_transcribe_audio(
    *,
    source_audio: Path,
    sheetsage_model: str | None = None,
) -> dict[str, Any]:
    return _cover_worker().transcribe_audio.remote(
        audio_bytes=source_audio.read_bytes(),
        ext=source_audio.suffix.lower() or ".mp3",
        sheetsage_model=sheetsage_model,
    )


def modal_sheetsage_annotate(
    *,
    source_audio: Path,
    sheetsage_model: str | None = None,
) -> dict[str, Any]:
    return _cover_worker().sheetsage_annotate.remote(
        audio_bytes=source_audio.read_bytes(),
        ext=source_audio.suffix.lower() or ".mp3",
        sheetsage_model=sheetsage_model,
    )


def modal_generate_covers(
    *,
    model_spec: str,
    source_midi: Path,
    source_audio: Path | None,
    sheetsage_json: dict | None,
    frames_npz: Path | None,
    params_dict: dict,
    num_samples: int,
    seed: int,
    prompt_patches: list | None = None,
    style_vector: list[float] | None = None,
    sheetsage_model: str | None = None,
) -> dict[str, Any]:
    return _cover_worker().generate_covers.remote(
        model_spec=model_spec,
        source_midi_bytes=source_midi.read_bytes(),
        audio_bytes=source_audio.read_bytes() if (source_audio and source_audio.is_file()) else None,
        audio_ext=(source_audio.suffix.lower() if source_audio else ".mp3"),
        sheetsage_json=sheetsage_json,
        frames_npz_bytes=frames_npz.read_bytes() if (frames_npz and frames_npz.is_file()) else None,
        params_dict=params_dict,
        num_samples=num_samples,
        seed=seed,
        prompt_patches=prompt_patches,
        style_vector=style_vector,
        sheetsage_model=sheetsage_model,
    )


def modal_generate_score(
    *,
    score_model_spec: str,
    cover_midi: Path,
    source_midi: Path | None,
    sheetsage_json: dict,
    title: str,
) -> dict[str, Any]:
    return _cover_worker().generate_score.remote(
        score_model_spec=score_model_spec,
        cover_midi_bytes=cover_midi.read_bytes(),
        source_midi_bytes=source_midi.read_bytes() if (source_midi and source_midi.is_file()) else None,
        sheetsage_json=sheetsage_json,
        title=title,
    )

