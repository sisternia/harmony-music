"""Unified Songscription service for Qwen3-ASR, Whisper alignment and vocal recovery.

Clean, lean architecture:
  1. Audio I/O & BS-RoFormer Vocal Cache (`LyricsCache` scoped by audio SHA-1 + config hash).
  2. Acoustic Utilities (`_normalize_vocal_rms`, `_blend_vocal_with_mix`, 50ms RMS envelope, valley chunking).
  3. Recognition and recovery (`cover_studio.music_asr_engine`, `cover_studio.vocal_recovery`)
     with shared vocal preservation, recovery and explicit alignment provenance.
  4. Mode 2 Forced Alignment (`parse_user_lyrics`, `align_user_lyrics_to_anchors`), Structure & Chords.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .engine import REPO_ROOT, Cancelled, Engine
from .lyrics_schema import (
    DEFAULT_STREAM_ID,
    EVENT_TYPES,
    HUMMING_TOKENS,
    LEXICAL_ADLIB_PHRASES,
    PIPELINE_CONFIG_VERSION,
    REVIEW_STATUSES,
    SCHEMA_VERSION,
    VOCAL_ROLES,
    VOCABLE_TOKENS,
    assign_overlap_groups,
    classify_vocal_event,
    compute_pipeline_config_hash,
    format_review_reason,
    get_multi_singer_separation_status,
    migrate_segment,
    migrate_transcript,
    normalize_stream_id,
    normalize_vocable_elongation,
    stream_label,
)
from .music_asr_engine import (
    DEFAULT_MODEL_ID,
    SONGSCRIPTION_MODEL_SPECS,
    normalize_model_id,
    refine_segments_with_music_asr,
    run_music_asr_pipeline,
)
from .project import Project, file_sha1, now, read_json, write_json

VOCAL_WORKER = Path(__file__).resolve().parent / "lyrics_vocal_worker.py"
LYRICS_MODELS: tuple[dict[str, str], ...] = SONGSCRIPTION_MODEL_SPECS

LANGUAGE_NAMES: dict[str, str] = {
    "ja": "Tiếng Nhật (Japanese)",
    "en": "Tiếng Anh (English)",
    "vi": "Tiếng Việt (Vietnamese)",
    "ko": "Tiếng Hàn (Korean)",
    "zh": "Tiếng Trung (Chinese)",
    "yue": "Tiếng Quảng Đông (Cantonese)",
    "fr": "Tiếng Pháp (French)",
    "es": "Tiếng Tây Ban Nha (Spanish)",
    "de": "Tiếng Đức (German)",
    "it": "Tiếng Ý (Italian)",
    "pt": "Tiếng Bồ Đào Nha (Portuguese)",
    "ru": "Tiếng Nga (Russian)",
    "th": "Tiếng Thái (Thai)",
    "id": "Tiếng Indonesia (Indonesian)",
}

HALLUCINATION_PATTERNS = re.compile(
    r"^[\s\.,!?;:'\"\-~…♪♫♬♩()\[\]]*(?:"
    r"subtitles?\s+by.*|amara\.org.*|captioned\s+by.*|translated\s+by.*|"
    r"thank\s+you\s+for\s+watching.*|thanks\s+for\s+watching.*|please\s+subscribe.*|"
    r"we(?:'ll|\s+will)\s+be\s+right\s+back.*|stay\s+tuned.*|to\s+be\s+continued.*|"
    r"ご視聴ありがとうございました.*|チャンネル登録.*|字幕[:：].*|提供[:：].*|"
    r"시청해\s*주셔서\s*감사합니다.*|구독과\s*좋아요.*|mbc\s*뉴스.*|"
    r"hãy\s+subscribe.*|cảm\s+ơn\s+các\s+bạn\s+đã\s+xem.*|đăng\s+ký\s+kênh.*|"
    r"请不吝点赞.*|感謝觀看.*|字幕由.*|优优独播剧场.*"
    r")[\s\.,!?;:'\"\-~…♪♫♬♩()\[\]]*$",
    re.IGNORECASE,
)

NOISE_ONLY_PATTERN = re.compile(r"^[\s\.,!?;:'\"\-~…♪♫♬♩()\[\]【】「」『』、。！？・]+$")

SECTION_HEADER_RE = re.compile(
    r"^\s*[\[\(【]\s*"
    r"(intro|verse(?:\s*\d+)?|pre[-\s]?chorus|chorus(?:\s*\d+)?|bridge|outro|hook|interlude|instrumental|"
    r"lời\s*\d*|điệp\s*khúc|kết|mở\s*đầu|"
    r"イントロ|aメロ|bメロ|サビ|大サビ|Cメロ|アウトロ|間奏)"
    r"\s*[\]\)】]\s*$",
    re.IGNORECASE,
)

STREAM_HEADER_RE = re.compile(
    r"^\s*[\[\(【]\s*(?:stream|luồng|luong)\s*([a-z0-9]+)\s*[\]\)】]\s*$",
    re.IGNORECASE,
)

SECTION_CANONICAL: dict[str, str] = {
    "intro": "Intro", "mở đầu": "Intro", "イントロ": "Intro",
    "verse": "Verse", "aメロ": "Verse",
    "pre-chorus": "Pre-Chorus", "pre chorus": "Pre-Chorus", "prechorus": "Pre-Chorus", "bメロ": "Pre-Chorus",
    "chorus": "Chorus", "hook": "Chorus", "điệp khúc": "Chorus", "サビ": "Chorus", "大サビ": "Chorus",
    "bridge": "Bridge", "cメロ": "Bridge",
    "outro": "Outro", "kết": "Outro", "アウトロ": "Outro",
    "interlude": "Instrumental", "instrumental": "Instrumental", "間奏": "Instrumental",
}

VI_DIACRITICS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]")
JA_KANA = re.compile(r"[\u3040-\u30ff]")
KO_HANGUL = re.compile(r"[\uac00-\ud7af]")
ZH_HAN = re.compile(r"[\u4e00-\u9fff]")
CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_DLL_INIT_LOCK = threading.Lock()
_DLL_INITIALIZED = False


# ================================================================== Runtime & Audio I/O
def ensure_windows_runtime() -> Path | None:
    """Ensure bundled FFmpeg is discoverable on Windows."""
    global _DLL_INITIALIZED
    ffmpeg_bin: Path | None = None
    bundled = REPO_ROOT / ".tsumugi" / "ffmpeg-shared" / "bin"
    if (bundled / "ffmpeg.exe").is_file() or (bundled / "ffmpeg").is_file():
        ffmpeg_bin = bundled.resolve()
    else:
        found = shutil.which("ffmpeg")
        if found:
            ffmpeg_bin = Path(found).resolve().parent

    with _DLL_INIT_LOCK:
        if ffmpeg_bin is not None:
            ff_str = str(ffmpeg_bin)
            cur_path = os.environ.get("PATH", "")
            if ff_str not in cur_path.split(os.pathsep):
                os.environ["PATH"] = f"{ff_str}{os.pathsep}{cur_path}"
            if not _DLL_INITIALIZED and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(ff_str)
                except OSError:
                    pass
        _DLL_INITIALIZED = True
    return ffmpeg_bin


def language_label(code: str | None) -> str:
    if not code:
        return "Tự động"
    return LANGUAGE_NAMES.get(code, code.upper())


def infer_segment_script_lang(text: str, fallback: str) -> str:
    if JA_KANA.search(text):
        return "ja"
    if KO_HANGUL.search(text):
        return "ko"
    if VI_DIACRITICS.search(text):
        return "vi"
    if ZH_HAN.search(text) and fallback not in ("ja", "ko"):
        return "zh"
    if re.search(r"[A-Za-z]", text) and not re.search(r"[^\x00-\x7f]", text):
        return "en" if fallback in ("ja", "ko", "zh", "th", "ru") else fallback
    return fallback or "en"


def _ffmpeg_executable() -> str:
    ff = ensure_windows_runtime()
    if ff and (ff / "ffmpeg.exe").is_file():
        return str(ff / "ffmpeg.exe")
    if ff and (ff / "ffmpeg").is_file():
        return str(ff / "ffmpeg")
    return "ffmpeg"


def convert_audio_ffmpeg(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    vocal_bandpass: bool = False,
) -> None:
    cmd = [_ffmpeg_executable(), "-y", "-i", str(src), "-vn"]
    if vocal_bandpass:
        cmd += ["-af", "highpass=f=85,lowpass=f=11500"]
    cmd += ["-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"FFmpeg không thể xử lý tệp âm thanh: {res.stderr[-260:]}")


def load_wav_16k(wav_path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        duration = data.shape[0] / max(sr, 1)
        target_len = max(1, int(round(duration * 16000)))
        x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ================================================================== Caching & Vocal Separation
class LyricsCache:
    """Shared SHA-1 cache for vocal/instrumental stems, stream stems, and config-scoped anchors."""

    def __init__(self, root: Path) -> None:
        self.base = Path(root).resolve() / ".lyrics_cache"
        self.vocals_dir = self.base / "vocals"
        self.instrumentals_dir = self.base / "instrumentals"
        self.anchors_dir = self.base / "anchors"
        self.vocals_dir.mkdir(parents=True, exist_ok=True)
        self.instrumentals_dir.mkdir(parents=True, exist_ok=True)
        self.anchors_dir.mkdir(parents=True, exist_ok=True)

    def vocal_path(self, audio_sha1: str) -> Path:
        return self.vocals_dir / f"{audio_sha1}.wav"

    def has_vocal(self, audio_sha1: str) -> bool:
        p = self.vocal_path(audio_sha1)
        return p.is_file() and p.stat().st_size > 1024

    def instrumental_path(self, audio_sha1: str, ext: str = ".mp3", *, studio_stem: bool = True) -> Path:
        norm_ext = ext if ext.startswith(".") else f".{ext}"
        suffix = f"_bsr{norm_ext.lower()}" if studio_stem else norm_ext.lower()
        return self.instrumentals_dir / f"{audio_sha1}{suffix}"

    def existing_instrumental_path(self, audio_sha1: str, *, require_studio_stem: bool = False) -> Path | None:
        candidates = (
            (self.instrumental_path(audio_sha1, ".mp3", studio_stem=True), self.instrumental_path(audio_sha1, ".wav", studio_stem=True))
            if require_studio_stem
            else (
                self.instrumental_path(audio_sha1, ".mp3", studio_stem=True),
                self.instrumental_path(audio_sha1, ".wav", studio_stem=True),
                self.instrumental_path(audio_sha1, ".mp3", studio_stem=False),
                self.instrumental_path(audio_sha1, ".wav", studio_stem=False),
            )
        )
        for p in candidates:
            if p.is_file() and p.stat().st_size > 1024:
                return p
        return None

    def has_instrumental(self, audio_sha1: str, *, require_studio_stem: bool = False) -> bool:
        return self.existing_instrumental_path(audio_sha1, require_studio_stem=require_studio_stem) is not None

    def save_instrumental_bytes(self, audio_sha1: str, data: bytes, ext: str = ".mp3", *, studio_stem: bool = True) -> Path:
        dst = self.instrumental_path(audio_sha1, ext, studio_stem=studio_stem)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(f".tmp{dst.suffix}")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        return dst

    def stream_vocal_path(self, audio_sha1: str, stream_id: str) -> Path:
        return self.vocals_dir / f"{audio_sha1}_{normalize_stream_id(stream_id)}.wav"

    def has_stream_vocal(self, audio_sha1: str, stream_id: str) -> bool:
        p = self.stream_vocal_path(audio_sha1, stream_id)
        return p.is_file() and p.stat().st_size > 1024

    def available_streams(self, audio_sha1: str) -> list[str]:
        return [sid for sid in ("stream_a", "stream_b", "stream_c") if self.has_stream_vocal(audio_sha1, sid)]

    def anchor_path(
        self,
        audio_sha1: str,
        lang_key: str,
        model_id: str | None = None,
        vocal_isolated: bool | None = None,
        config_hash: str | None = None,
    ) -> Path:
        safe_lang = re.sub(r"[^a-z0-9_-]", "", (lang_key or "auto").lower()) or "auto"
        if not model_id:
            return self.anchors_dir / f"{audio_sha1}_{safe_lang}.json"
        safe_model = re.sub(r"[^a-z0-9_-]", "", model_id.lower()) or "default"
        source = "vocals" if vocal_isolated else "mix" if vocal_isolated is False else "audio"
        if config_hash:
            safe_cfg = re.sub(r"[^a-z0-9_-]", "", config_hash.lower())[:16]
            return self.anchors_dir / f"{audio_sha1}_{safe_lang}_{safe_model}_{source}_{safe_cfg}.json"
        return self.anchors_dir / f"{audio_sha1}_{safe_lang}_{safe_model}_{source}.json"

    def load_anchors(
        self,
        audio_sha1: str,
        lang_key: str,
        model_id: str | None = None,
        vocal_isolated: bool | None = None,
        config_hash: str | None = None,
    ) -> dict | None:
        candidates = (
            (self.anchor_path(audio_sha1, lang_key, model_id, vocal_isolated, config_hash),)
            if model_id
            else (self.anchor_path(audio_sha1, lang_key), self.anchor_path(audio_sha1, "auto"))
        )
        for candidate in candidates:
            if candidate.is_file():
                try:
                    payload = read_json(candidate)
                    if model_id and payload.get("model") != model_id:
                        continue
                    if vocal_isolated is not None and payload.get("vocal_isolated") != vocal_isolated:
                        continue
                    if config_hash and payload.get("config_hash") != config_hash:
                        continue
                    return payload
                except Exception:
                    continue
        return None

    def save_anchors(
        self,
        audio_sha1: str,
        lang_key: str,
        payload: dict,
        model_id: str | None = None,
        vocal_isolated: bool | None = None,
        config_hash: str | None = None,
    ) -> None:
        try:
            stored = dict(payload)
            if model_id:
                stored.setdefault("model", model_id)
            if vocal_isolated is not None:
                stored.setdefault("vocal_isolated", vocal_isolated)
            if config_hash:
                stored["config_hash"] = config_hash
            stored.setdefault("pipeline_version", PIPELINE_CONFIG_VERSION)
            write_json(self.anchor_path(audio_sha1, lang_key, model_id, vocal_isolated, config_hash), stored)
        except Exception:
            pass


def encode_wav_to_mp3(src_wav: Path, dst_mp3: Path, *, bitrate: str = "192k") -> bool:
    """Encode WAV to compact high-quality stereo MP3 via FFmpeg; returns True on success."""
    dst_mp3.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = dst_mp3.with_suffix(".tmp.mp3")
    cmd = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(src_wav),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(tmp_mp3),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0 and tmp_mp3.is_file() and tmp_mp3.stat().st_size > 1024:
        os.replace(tmp_mp3, dst_mp3)
        return True
    tmp_mp3.unlink(missing_ok=True)
    return False


def mix_non_vocal_stems(saved_stems: dict[str, Path], out_wav: Path) -> Path | None:
    """Combine BS-RoFormer's non-vocal stems (`bass`, `drums`, `other`, `guitar`, `piano`)
    into a clean full-fidelity stereo instrumental WAV."""
    import soundfile as sf

    inst_keys = ("bass", "drums", "other", "guitar", "piano", "instrumental", "no_vocals")
    stem_paths = [saved_stems[k] for k in inst_keys if k in saved_stems and saved_stems[k].is_file()]
    if not stem_paths:
        return None

    acc: np.ndarray | None = None
    target_sr = 44100
    for sp in stem_paths:
        arr, sr = sf.read(str(sp), dtype="float32", always_2d=True)
        if not np.isfinite(arr).all():
            continue
        if acc is None:
            acc = arr.astype(np.float64, copy=True)
            target_sr = int(sr)
        else:
            n = min(acc.shape[0], arr.shape[0])
            c = min(acc.shape[1], arr.shape[1])
            acc = acc[:n, :c] + arr[:n, :c].astype(np.float64)

    if acc is None or acc.size == 0:
        return None

    peak = float(np.max(np.abs(acc), initial=0.0))
    if peak > 0.98:
        acc = acc * (0.98 / peak)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_wav.with_suffix(".tmp.wav")
    sf.write(str(tmp_out), np.clip(acc, -1.0, 1.0).astype(np.float32), target_sr, subtype="PCM_16")
    os.replace(tmp_out, out_wav)
    return out_wav


_INSTRUMENTAL_LOCKS_GUARD = threading.Lock()
_INSTRUMENTAL_LOCKS: dict[str, threading.Lock] = {}


def _instrumental_lock(audio_sha1: str) -> threading.Lock:
    with _INSTRUMENTAL_LOCKS_GUARD:
        lock = _INSTRUMENTAL_LOCKS.get(audio_sha1)
        if lock is None:
            lock = threading.Lock()
            _INSTRUMENTAL_LOCKS[audio_sha1] = lock
        return lock


def derive_instrumental_from_vocal_cache(
    source_audio: Path,
    vocal_wav: Path,
    cache: LyricsCache,
    audio_sha1: str,
) -> Path | None:
    """Offline fallback when BS-RoFormer GPU/local separation is unavailable:
    applies a phase-invariant STFT Wiener vocal-suppression mask on BOTH stereo channels so
    even phase-shifted or stereo-reverbed vocal bleed is attenuated cleanly."""
    import soundfile as sf

    if not source_audio.is_file() or not vocal_wav.is_file():
        return None

    with tempfile.TemporaryDirectory(prefix="harmony-inst-") as tmp_str:
        tmp_dir = Path(tmp_str)
        mix_stereo_wav = tmp_dir / "mix_16k_stereo.wav"
        convert_audio_ffmpeg(source_audio, mix_stereo_wav, sample_rate=16000, channels=2, vocal_bandpass=False)
        mix_st, _ = sf.read(str(mix_stereo_wav), dtype="float32", always_2d=True)
        voc_mono = load_wav_16k(vocal_wav)
        n = min(mix_st.shape[0], voc_mono.shape[0])
        if n < 1600:
            return None

        left = mix_st[:n, 0].astype(np.float64, copy=False)
        right = mix_st[:n, 1].astype(np.float64, copy=False) if mix_st.shape[1] >= 2 else left
        voc = voc_mono[:n].astype(np.float64, copy=False)

        # Vectorized STFT Wiener mask: suppresses vocal magnitude on BOTH L & R (immune to resampler phase shift)
        n_fft = 1024
        hop = 256
        n_overlap = n_fft // hop  # 4
        win = np.hanning(n_fft)
        pad = n_fft
        # Pad end so n_frames is a multiple of n_overlap for O(4) vectorized overlap-add
        raw_total = n + 2 * pad
        n_frames = 1 + (raw_total - n_fft) // hop
        rem = n_frames % n_overlap
        if rem:
            n_frames += n_overlap - rem
        total_len = (n_frames - 1) * hop + n_fft
        pad_right = total_len - (pad + n)

        l_pad = np.pad(left, (pad, pad_right))
        r_pad = np.pad(right, (pad, pad_right))
        v_pad = np.pad(voc, (pad, pad_right))

        frames_l = np.lib.stride_tricks.sliding_window_view(l_pad, n_fft)[::hop][:n_frames] * win
        frames_r = np.lib.stride_tricks.sliding_window_view(r_pad, n_fft)[::hop][:n_frames] * win
        frames_v = np.lib.stride_tricks.sliding_window_view(v_pad, n_fft)[::hop][:n_frames] * win

        fl = np.fft.rfft(frames_l, axis=1)
        fr = np.fft.rfft(frames_r, axis=1)
        fv = np.fft.rfft(frames_v, axis=1)

        eps = 1e-10
        v_pow = (np.abs(fv) ** 2) * 1.35
        l_pow = np.abs(fl) ** 2
        r_pow = np.abs(fr) ** 2
        mask_l = np.clip((l_pow - v_pow) / (l_pow + eps), 0.0, 1.0)
        mask_r = np.clip((r_pow - v_pow) / (r_pow + eps), 0.0, 1.0)

        istft_l = np.fft.irfft(fl * mask_l, n=n_fft, axis=1) * win
        istft_r = np.fft.irfft(fr * mask_r, n=n_fft, axis=1) * win

        out_l = np.zeros(total_len, dtype=np.float64)
        out_r = np.zeros(total_len, dtype=np.float64)
        norm = np.zeros(total_len, dtype=np.float64)
        win_sq = win * win
        for k in range(n_overlap):
            sub_l = istft_l[k::n_overlap]
            sub_r = istft_r[k::n_overlap]
            cnt = sub_l.shape[0]
            s = k * hop
            e = s + cnt * n_fft
            out_l[s:e] += sub_l.reshape(-1)
            out_r[s:e] += sub_r.reshape(-1)
            norm[s:e] += np.tile(win_sq, cnt)

        valid = norm > 1e-8
        out_l[valid] /= norm[valid]
        out_r[valid] /= norm[valid]
        out_stereo = np.stack([out_l[pad : pad + n], out_r[pad : pad + n]], axis=1)
        peak = float(np.max(np.abs(out_stereo), initial=0.0))
        if peak > 0.98:
            out_stereo = out_stereo * (0.98 / peak)

        raw_inst_wav = tmp_dir / "instrumental.wav"
        sf.write(str(raw_inst_wav), np.clip(out_stereo, -1.0, 1.0).astype(np.float32), 16000, subtype="PCM_16")

        mp3_dst = cache.instrumental_path(audio_sha1, ".mp3", studio_stem=False)
        if encode_wav_to_mp3(raw_inst_wav, mp3_dst, bitrate="192k"):
            return mp3_dst

        wav_dst = cache.instrumental_path(audio_sha1, ".wav", studio_stem=False)
        wav_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_inst_wav, wav_dst)
        return wav_dst


def ensure_instrumental_cached(
    cache: LyricsCache,
    source_audio: Path | None,
    audio_sha1: str | None,
    *,
    engine: Engine | None = None,
) -> Path | None:
    """Return studio-grade BS-RoFormer 44.1kHz stereo instrumental (`bass + drums + other + guitar + piano`),
    running BS-RoFormer on Modal GPU L4 (or local tsumugi) once per SHA-1 and caching permanently.
    Falls back to STFT spectral suppression only when BS-RoFormer is completely unavailable."""
    if not audio_sha1:
        return None
    sha1 = str(audio_sha1)
    existing_bsr = cache.existing_instrumental_path(sha1, require_studio_stem=True)
    if existing_bsr is not None:
        return existing_bsr
    if source_audio is None or not source_audio.is_file():
        return cache.existing_instrumental_path(sha1, require_studio_stem=False)

    with _instrumental_lock(sha1):
        # Double-check after acquiring per-song lock (prevents duplicate GPU runs on concurrent Range requests)
        existing_bsr = cache.existing_instrumental_path(sha1, require_studio_stem=True)
        if existing_bsr is not None:
            return existing_bsr

        # 1. Preferred: True 6-stem BS-RoFormer on Modal GPU L4 (~6s, zero vocals, 44.1kHz stereo)
        try:
            from .modal_client import is_modal_enabled, modal_separate_instrumental

            if is_modal_enabled():
                m_out = modal_separate_instrumental(source_audio=source_audio)
                inst_bytes = m_out.get("instrumental_audio_bytes")
                if inst_bytes and len(inst_bytes) > 1024:
                    inst_ext = str(m_out.get("instrumental_ext") or ".mp3")
                    saved = cache.save_instrumental_bytes(sha1, inst_bytes, ext=inst_ext, studio_stem=True)
                    voc_bytes = m_out.get("vocal_wav_bytes")
                    if voc_bytes and not cache.has_vocal(sha1):
                        vp = cache.vocal_path(sha1)
                        vp.parent.mkdir(parents=True, exist_ok=True)
                        vp.write_bytes(voc_bytes)
                    return saved
        except Exception as exc:
            logging.getLogger(__name__).warning("Modal BS-RoFormer instrumental separation failed for %s: %s", sha1[:10], exc)

        # 2. Local BS-RoFormer via .tsumugi worker when available
        tsumugi = getattr(engine, "tsumugi", None) if engine is not None else None
        if tsumugi is not None and getattr(tsumugi, "python", None) and Path(tsumugi.python).is_file():
            try:
                with tempfile.TemporaryDirectory(prefix="harmony-inst-local-") as tmp_str:
                    tmp_dir = Path(tmp_str)
                    isolate_vocals_cached(
                        engine=engine,
                        cache=cache,
                        source_audio=source_audio,
                        audio_sha1=sha1,
                        tmp_dir=tmp_dir,
                        want_isolate=True,
                        force_separate=True,
                        progress=lambda _p, _m: None,
                        should_stop=lambda: False,
                        log=lambda _m: None,
                    )
                local_bsr = cache.existing_instrumental_path(sha1, require_studio_stem=True)
                if local_bsr is not None:
                    return local_bsr
            except Exception as exc:
                logging.getLogger(__name__).warning("Local BS-RoFormer instrumental separation failed for %s: %s", sha1[:10], exc)

        # 3. Offline fallback: STFT spectral suppression on both stereo channels
        existing_any = cache.existing_instrumental_path(sha1, require_studio_stem=False)
        if existing_any is not None:
            return existing_any
        if cache.has_vocal(sha1):
            return derive_instrumental_from_vocal_cache(source_audio, cache.vocal_path(sha1), cache, sha1)
        return None


def isolate_vocals_cached(
    engine: Engine,
    cache: LyricsCache,
    source_audio: Path,
    audio_sha1: str,
    tmp_dir: Path,
    want_isolate: bool,
    force_separate: bool,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> tuple[Path, bool, bool]:
    cached_wav = cache.vocal_path(audio_sha1)
    if want_isolate and not force_separate and cache.has_vocal(audio_sha1):
        log("Đã tìm thấy kênh giọng hát đã tách trong bộ nhớ đệm — tái sử dụng ngay.")
        progress(0.35, "separating_vocals")
        return cached_wav, True, True

    tsumugi = getattr(engine, "tsumugi", None)
    if want_isolate and tsumugi is not None and getattr(tsumugi, "python", None) and Path(tsumugi.python).is_file():
        progress(0.14, "separating_vocals")
        stereo_44k = tmp_dir / "input_44k_stereo.wav"
        convert_audio_ffmpeg(source_audio, stereo_44k, sample_rate=44100, channels=2, vocal_bandpass=False)
        if should_stop():
            raise Cancelled()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        env.pop("VIRTUAL_ENV", None)
        cmd = [
            str(tsumugi.python), "-X", "utf8", str(VOCAL_WORKER),
            "--tsumugi-dir", str(tsumugi.dir), "--audio", str(stereo_44k),
            "--out", str(cached_wav),
            "--instrumental-out", str(tmp_dir / "instrumental_44k.wav"),
            "--device", str(getattr(engine, "device", "auto")),
        ]
        if getattr(tsumugi, "ffmpeg_dir", None):
            cmd += ["--ffmpeg-dir", str(tsumugi.ffmpeg_dir)]
        log("Đang tách riêng giọng hát khỏi nhạc đệm (BS-RoFormer Local)...")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, cwd=str(tsumugi.dir),
        )
        stopped = threading.Event()

        def watch() -> None:
            while proc.poll() is None:
                if should_stop():
                    stopped.set()
                    proc.kill()
                    return
                stopped.wait(0.35)

        threading.Thread(target=watch, daemon=True).start()
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if line.startswith("@@progress "):
                try:
                    info = json.loads(line[len("@@progress ") :])
                    progress(round(0.15 + 0.24 * (int(info.get("done", 0)) / max(1, int(info.get("total", 1)))), 3), "separating_vocals")
                except Exception:
                    pass
            elif line and not line.startswith("@@"):
                log(line[-150:])
        code = proc.wait()
        stopped.set()
        if should_stop():
            raise Cancelled()
        if code == 0 and cache.has_vocal(audio_sha1):
            inst_44k = tmp_dir / "instrumental_44k.wav"
            if inst_44k.is_file() and inst_44k.stat().st_size > 1024:
                mp3_dst = cache.instrumental_path(audio_sha1, ".mp3", studio_stem=True)
                if not encode_wav_to_mp3(inst_44k, mp3_dst, bitrate="192k"):
                    wav_dst = cache.instrumental_path(audio_sha1, ".wav", studio_stem=True)
                    shutil.copy2(inst_44k, wav_dst)
            progress(0.40, "separating_vocals")
            return cached_wav, True, False
        log("Không thể tách kênh vocals bằng BS-RoFormer, dùng âm thanh gốc.")

    progress(0.30, "separating_vocals")
    raw_16k = tmp_dir / "raw_16k.wav"
    convert_audio_ffmpeg(source_audio, raw_16k, sample_rate=16000, channels=1, vocal_bandpass=False)
    return raw_16k, False, False


# ================================================================== Acoustic Conditioning & Phrase Chunking
def compute_vocal_envelope(audio_16k: np.ndarray, hop_samples: int = 800) -> tuple[np.ndarray, float]:
    """Compute 50ms (20 fps) RMS envelope and adaptive vocal floor threshold."""
    n = audio_16k.shape[0]
    if n == 0:
        return np.zeros(1, dtype=np.float32), 0.002
    win = hop_samples * 2
    n_frames = max(1, math.ceil(n / hop_samples))
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_samples
        chunk = audio_16k[s : min(n, s + win)]
        if chunk.size:
            rms[i] = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
    p95 = float(np.percentile(rms, 95)) if rms.size else 0.0
    floor = max(0.0012, min(0.0060, p95 * 0.04))
    return rms, floor


def segment_vocal_activity(rms_env: np.ndarray, floor: float, start_s: float, end_s: float) -> tuple[float, float]:
    if rms_env.size == 0 or end_s <= start_s:
        return 0.0, 0.0
    i0 = max(0, min(rms_env.size - 1, int(math.floor(start_s * 20.0))))
    i1 = max(i0 + 1, min(rms_env.size, int(math.ceil(end_s * 20.0))))
    sl = rms_env[i0:i1]
    return (float(np.mean(sl)) if sl.size else 0.0), (float(np.mean(sl >= floor)) if sl.size else 0.0)


def _normalize_vocal_rms(audio_16k: np.ndarray, target_rms: float = 0.095) -> np.ndarray:
    """Linear, pump-free RMS normalization (never boosts background bleed or reverb tails)."""
    if audio_16k.size == 0:
        return audio_16k
    abs_a = np.abs(audio_16k)
    p85 = float(np.percentile(abs_a, 85))
    active = abs_a >= max(0.008, p85 * 0.25)
    eff_rms = float(
        np.sqrt(np.mean(np.square(audio_16k[active] if np.any(active) else audio_16k, dtype=np.float64)) + 1e-12)
    )
    if eff_rms < 1e-4:
        return audio_16k
    out = (audio_16k * float(np.clip(target_rms / eff_rms, 0.55, 2.2))).astype(np.float32)
    peak = float(np.max(np.abs(out), initial=0.0))
    if peak > 0.94:
        out = (out * (0.94 / peak)).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _blend_vocal_with_mix(vocal_16k: np.ndarray, mix_16k: np.ndarray, mix_weight: float = 0.20) -> np.ndarray:
    """RMS-matched blend restoring sharp consonants (s, t, k, p, h) without letting instrumental drown vocals."""
    n = min(vocal_16k.shape[0], mix_16k.shape[0])
    if n == 0:
        return vocal_16k
    v, m = vocal_16k[:n], mix_16k[:n]
    v_rms = float(np.sqrt(np.mean(np.square(v, dtype=np.float64)) + 1e-12))
    m_rms = float(np.sqrt(np.mean(np.square(m, dtype=np.float64)) + 1e-12))
    if v_rms < 0.0018 or m_rms < 1e-5:
        return _normalize_vocal_rms(v)
    blended = ((1.0 - mix_weight) * v + mix_weight * (m * (v_rms / m_rms))).astype(np.float32)
    return _normalize_vocal_rms(blended)


def _split_long_regions_at_pauses(
    regions: list[tuple[float, float]],
    rms_env: np.ndarray,
    duration: float,
    *,
    max_chunk_sec: float = 28.0,
    min_chunk_sec: float = 14.0,
    overlap_sec: float = 0.0,
) -> list[tuple[float, float]]:
    """Split long singing regions at deepest vocal energy valleys with ZERO chunk overlap by default,
    staying cleanly inside `openai/whisper-large-v3`'s native 30s encoder window."""
    if not regions:
        return []
    out: list[tuple[float, float]] = []
    n_frames = rms_env.size
    for reg_s, reg_e in regions:
        cur_s, reg_end = max(0.0, float(reg_s)), min(duration, float(reg_e))
        while reg_end - cur_s > max_chunk_sec:
            lo, hi = cur_s + min_chunk_sec, min(reg_end - 3.0, cur_s + max_chunk_sec)
            if hi <= lo or n_frames == 0:
                cut_t = min(reg_end, cur_s + 20.0)
            else:
                i0, i1 = max(0, min(n_frames - 1, int(round(lo * 20.0)))), max(1, min(n_frames, int(round(hi * 20.0))))
                w_rms = rms_env[i0:i1]
                best = i0 + int(np.argmin(np.convolve(w_rms, np.ones(5, dtype=np.float32) / 5.0, mode="same") if w_rms.size >= 5 else w_rms))
                cut_t = round(best * 0.05, 2)
                if cut_t <= cur_s + 4.0:
                    cut_t = min(reg_end, cur_s + 20.0)
            out.append((round(cur_s, 2), round(min(reg_end, cut_t + overlap_sec), 2)))
            cur_s = max(cur_s + 4.0, cut_t - overlap_sec * 0.5)
        if reg_end - cur_s >= 0.4:
            out.append((round(cur_s, 2), round(reg_end, 2)))
    return out


def _extract_vocal_regions(
    rms_env: np.ndarray,
    floor: float,
    duration: float,
    *,
    pad_sec: float = 0.65,
    merge_gap_sec: float = 2.4,
    min_region_sec: float = 0.6,
) -> list[tuple[float, float]]:
    if rms_env.size == 0 or duration <= 0:
        return [(0.0, max(0.1, duration))]
    active = rms_env >= floor
    if not np.any(active):
        return []
    raw_spans: list[tuple[float, float]] = []
    in_seg, seg_start = False, 0
    for idx, val in enumerate(active):
        if val and not in_seg:
            in_seg, seg_start = True, idx
        elif not val and in_seg:
            in_seg = False
            s_t, e_t = max(0.0, seg_start * 0.05 - pad_sec), min(duration, idx * 0.05 + pad_sec)
            if e_t - s_t >= min_region_sec:
                raw_spans.append((s_t, e_t))
    if in_seg:
        s_t = max(0.0, seg_start * 0.05 - pad_sec)
        if duration - s_t >= min_region_sec:
            raw_spans.append((s_t, duration))

    if not raw_spans:
        return []
    merged: list[tuple[float, float]] = [raw_spans[0]]
    for s_t, e_t in raw_spans[1:]:
        ps, pe = merged[-1]
        if s_t - pe <= merge_gap_sec:
            merged[-1] = (ps, max(pe, e_t))
        else:
            merged.append((s_t, e_t))
    return _split_long_regions_at_pauses(merged, rms_env, duration)


# ================================================================== Vocalise, Bleed, Validation & Timing Helpers
def _norm_text_for_cmp(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower()).strip())


_BARE_VOWEL_LOOP_TOKENS: set[str] = {
    "ah", "ahh", "aah", "aaah", "oh", "ohh", "ooh", "oooh", "uh", "uhh",
    "eh", "ehh", "ee", "oo", "aa", "a", "o", "u", "e", "i", "hm", "hmm", "mm", "mmm",
    "啊", "哦", "噢", "喔", "呃", "呜", "嗯", "哼",
    "あ", "お", "う", "아", "오", "우",
}
_LOOP_TOKEN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[^\W_]+", flags=re.UNICODE)


def _analyze_token_repetition(text: str) -> dict[str, Any]:
    """Measure consecutive identical token runs and short n-gram loops for ASR degeneration detection."""
    matches = list(_LOOP_TOKEN_RE.finditer(text))
    toks = [normalize_vocable_elongation(m.group().lower()) for m in matches]
    n = len(toks)
    if n == 0:
        return {
            "toks": [], "matches": [], "max_run": 0, "max_run_tok": "", "max_run_start": 0,
            "trailing_run": 0, "trailing_tok": "", "trailing_start": 0,
            "max_ngram_reps": 0, "max_ngram_k": 0, "max_ngram_start": 0,
        }

    # Longest consecutive identical single-token run + trailing run
    max_run, max_run_tok, max_run_start = 1, toks[0], 0
    cur_run, cur_start = 1, 0
    for idx in range(1, n):
        if toks[idx] == toks[idx - 1]:
            cur_run += 1
        else:
            if cur_run > max_run:
                max_run, max_run_tok, max_run_start = cur_run, toks[cur_start], cur_start
            cur_run, cur_start = 1, idx
    if cur_run > max_run:
        max_run, max_run_tok, max_run_start = cur_run, toks[cur_start], cur_start
    trailing_run, trailing_tok, trailing_start = cur_run, toks[cur_start], cur_start

    # Longest consecutive short k-gram (k=2..4) repetition run
    max_ngram_reps, max_ngram_k, max_ngram_start = 1, 0, 0
    for k in (2, 3, 4):
        if n < k * 4:
            continue
        i = 0
        while i <= n - k * 2:
            pat = toks[i : i + k]
            reps = 1
            while i + (reps + 1) * k <= n and toks[i + reps * k : i + (reps + 1) * k] == pat:
                reps += 1
            if reps > max_ngram_reps:
                max_ngram_reps, max_ngram_k, max_ngram_start = reps, k, i
            i += reps * k if reps > 1 else 1

    return {
        "toks": toks,
        "matches": matches,
        "max_run": max_run,
        "max_run_tok": max_run_tok,
        "max_run_start": max_run_start,
        "trailing_run": trailing_run,
        "trailing_tok": trailing_tok,
        "trailing_start": trailing_start,
        "max_ngram_reps": max_ngram_reps,
        "max_ngram_k": max_ngram_k,
        "max_ngram_start": max_ngram_start,
    }


def _is_runaway_token_loop(
    text: str,
    comp_ratio: float = 1.0,
    *,
    duration_sec: float | None = None,
) -> bool:
    """Detect genuine ASR autoregressive degeneration loops (e.g. 'On wide horizons, ah ah ah ...' x500)
    WITHOUT ever dropping or deduplicating real repeated vocalises/scat (even 12-40 authentic beats)."""
    if not text or not text.strip():
        return False
    stats = _analyze_token_repetition(text)
    toks = stats["toks"]
    n = len(toks)
    if n < 12:
        return False

    vocal_tokens = VOCABLE_TOKENS | HUMMING_TOKENS | {t for t in LEXICAL_ADLIB_PHRASES if " " not in t}
    max_run = stats["max_run"]
    max_tok = stats["max_run_tok"]
    trailing_run = stats["trailing_run"]
    trailing_tok = stats["trailing_tok"]
    trailing_start = stats["trailing_start"]
    max_ng_reps = stats["max_ngram_reps"]
    max_ng_k = stats["max_ngram_k"]

    # 1. Acoustic syllable-rate impossibility when clip/chunk duration is known:
    #    Genuine singing (even rapid rhythmic scat) stays <= ~5.5 repeated tokens/sec;
    #    autoregressive decoder loops emit 12-40 tokens/sec (e.g. 504 "ah" in 24.4s = 20.7/s).
    if duration_sec is not None and duration_sec > 0.1:
        if max_run >= 14 and max_run > max(16.0, duration_sec * 5.5):
            return True
        if max_ng_reps >= 10 and (max_ng_reps * max_ng_k) > max(24.0, duration_sec * 6.5):
            return True

    # 2. Lexical sentence suddenly degenerating into a long trailing bare-vowel loop
    #    (e.g. "On wide horizons, ah ah ah ..." x25+ — never flags normal 4-16 beat sung tails).
    if trailing_run >= 25 and trailing_tok in _BARE_VOWEL_LOOP_TOKENS and trailing_start >= 1:
        prefix_lexical = sum(1 for t in toks[:trailing_start] if t not in vocal_tokens)
        if prefix_lexical >= 1:
            return True

    # 3. Extreme unbroken token flood regardless of duration (preserves up to 40 real "Oh" / 48 real scat)
    if max_run >= 45 and max_tok in _BARE_VOWEL_LOOP_TOKENS:
        return True
    if max_run >= 60:
        return True
    if max_ng_reps >= 40:
        return True

    # 4. Non-vocable lexical token runaway loop (never touches authentic vocables or lexical ad-libs)
    ngram_start = stats["max_ngram_start"]
    ngram_pat = " ".join(toks[ngram_start : ngram_start + max_ng_k]) if max_ng_k > 0 else ""
    is_adlib_ngram = (
        ngram_pat in LEXICAL_ADLIB_PHRASES
        or (max_ng_k > 0 and all(t in vocal_tokens for t in toks[ngram_start : ngram_start + max_ng_k]))
    )
    if max_tok not in vocal_tokens and max_run >= 20:
        return True
    if not is_adlib_ngram and max_ng_reps >= 12:
        return True

    # 5. Legacy decoder compression diagnostic for non-vocable lexical spam
    if _is_mostly_vocalise(text):
        return False
    if comp_ratio > 5.0:
        return True
    return bool(n >= 12 and len(set(toks)) <= 2 and comp_ratio > 4.2)


def sanitize_runaway_repetition(
    text: str,
    *,
    duration_sec: float | None = None,
) -> tuple[str, bool]:
    """Trim an ASR decoder degeneration tail while preserving genuine preceding lyrics.

    Returns `(cleaned_text, was_trimmed)`. If the entire string was pure degeneration spam with
    no reliable lexical content, returns `("", True)` so callers can retry or emit an honest
    untranscribed review region instead of fabricating syllable counts.
    """
    raw = (text or "").strip()
    if not raw or not _is_runaway_token_loop(raw, duration_sec=duration_sec):
        return raw, False

    stats = _analyze_token_repetition(raw)
    toks = stats["toks"]
    matches = stats["matches"]
    vocal_tokens = VOCABLE_TOKENS | HUMMING_TOKENS | {t for t in LEXICAL_ADLIB_PHRASES if " " not in t}

    # Case A: Clean lexical prefix followed by a runaway repeated tail -> keep the lexical prefix cleanly
    cut_token_idx: int | None = None
    if stats["trailing_run"] >= 14 and stats["trailing_start"] > 0:
        cut_token_idx = stats["trailing_start"]
    elif stats["max_run"] >= 14 and stats["max_run_start"] > 0:
        cut_token_idx = stats["max_run_start"]
    elif stats["max_ngram_reps"] >= 10 and stats["max_ngram_start"] > 0:
        cut_token_idx = stats["max_ngram_start"] + stats["max_ngram_k"]

    if cut_token_idx is not None and cut_token_idx < len(matches):
        prefix_lexical = sum(1 for t in toks[:cut_token_idx] if t not in vocal_tokens)
        if prefix_lexical >= 1:
            char_cut = matches[cut_token_idx].start()
            trimmed = re.sub(r"[\s,;:\-—、，]+$", "", raw[:char_cut]).strip()
            if trimmed and not _is_runaway_token_loop(trimmed, duration_sec=duration_sec):
                return trimmed, True

    # Case B: Pure degeneration spam (no recoverable lexical prefix)
    return "", True


def _is_mostly_vocalise(text: str) -> bool:
    event_type, is_adlib, _ = classify_vocal_event(text)
    if event_type in ("vocable", "humming") or is_adlib is True:
        return True
    tokens = re.findall(r"[^\W_]+", normalize_vocable_elongation(text).lower(), flags=re.UNICODE)
    vocal_tokens = VOCABLE_TOKENS | HUMMING_TOKENS | {t for t in LEXICAL_ADLIB_PHRASES if " " not in t}
    return bool(tokens) and sum(token in vocal_tokens for token in tokens) / len(tokens) >= 0.75


def _clean_vocalise_text(
    text: str,
    primary_lang: str,
    comp_ratio: float = 1.0,
    *,
    allow_rhythmic: bool = False,
    is_mixed: bool = False,
) -> tuple[str | None, bool]:
    """Normalize stretched spelling without deleting articulated syllables or scripts.

    Keep the legacy keyword arguments for callers, but never use a song's language
    to replace foreign-script lyrics with guessed Latin vocables. Repetition and
    acoustic timing are reviewed separately from textual normalization.
    """
    stripped = normalize_vocable_elongation(text.strip())
    if not stripped or stripped in ("\u266a (Vocalise)", "(Vocalise)"):
        return None, bool(stripped)
    if _is_runaway_token_loop(stripped, comp_ratio):
        return None, True
    event_type, is_adlib, _ = classify_vocal_event(stripped)
    return stripped, bool(is_adlib is True or event_type in ("vocable", "humming"))


def _align_vocalise_to_envelope(
    seg: dict[str, Any],
    rms_env: np.ndarray,
    vocal_floor: float,
    duration: float,
) -> list[dict[str, Any]]:
    """Refine acoustic boundaries without deleting or multiplying recognized vocals.

    A low-energy/short vocable remains a reviewable hypothesis. Splitting across
    pauses requires word timing evidence; an energy envelope cannot tell which
    syllables belong to each burst.
    """
    event_type = seg.get("event_type")
    if event_type not in ("vocable", "humming") and not seg.get("is_adlib"):
        return [seg]
    start = max(0.0, float(seg.get("start") or 0.0))
    end = min(duration, max(start + 0.05, float(seg.get("end") or (start + 0.5))))

    def review(reason: str) -> list[dict[str, Any]]:
        out = dict(seg)
        reasons = list(out.get("review_reasons") or [])
        if reason not in reasons:
            reasons.append(reason)
        out["review_reasons"] = reasons
        out["review_reason"] = format_review_reason(reasons)
        out["low_confidence"] = True
        if out.get("review_status") not in ("verified", "edited", "skipped"):
            out["review_status"] = "needs_review"
        return [out]

    if rms_env.size == 0:
        return review("vocal_timing_unresolved")
    i0 = max(0, min(rms_env.size - 1, int(math.floor(start * 20.0))))
    i1 = max(i0 + 1, min(rms_env.size, int(math.ceil(end * 20.0))))
    active = rms_env[i0:i1] >= max(vocal_floor * 1.45, 0.0065)
    if not np.any(active):
        return review("weak_vocal_activity")
    if (end - start) < 0.25:
        return [seg]

    # Bridge micro-dips only; retain real breath pauses for word-based grouping.
    bridged = active.copy()
    idx = 0
    while idx < bridged.size:
        if bridged[idx]:
            idx += 1
            continue
        stop = idx
        while stop < bridged.size and not bridged[stop]:
            stop += 1
        if idx > 0 and stop < bridged.size and stop - idx < 9:
            bridged[idx:stop] = True
        idx = stop

    edges = np.diff(np.concatenate(([False], bridged, [False])).astype(np.int8))
    spans = [
        (max(start, (i0 + int(left)) * 0.05 - 0.04), min(end, (i0 + int(right)) * 0.05 + 0.06))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]
    words = seg.get("words") or []
    if len(spans) == 1:
        out = dict(seg)
        left, right = spans[0]
        # Word evidence takes precedence over the RMS threshold for quiet syllables.
        if words:
            left = min(left, min(float(word["start"]) for word in words))
            right = max(right, max(float(word["end"]) for word in words))
        out["start"] = round(max(start, left), 2)
        out["end"] = round(min(end, right), 2)
        return [out]

    if not words:
        return review("vocal_timing_unresolved")

    def join_words(items: list[dict[str, Any]]) -> str:
        text = ""
        for word in items:
            part = str(word.get("word") or "")
            if text and part and not part[0].isspace() and not CJK_CHAR.search(part):
                if not re.match(r"^[,.;:!?)]", part):
                    text += " "
            text += part
        return text.strip()

    # Only split when the word list accounts for the whole recognized phrase.
    if _norm_text_for_cmp(join_words(words)) != _norm_text_for_cmp(str(seg.get("text") or "")):
        return review("vocal_timing_unresolved")
    groups: list[list[dict[str, Any]]] = [[] for _ in spans]
    for word in words:
        ws, we = float(word["start"]), float(word["end"])
        midpoint = (ws + we) / 2.0
        best = min(
            range(len(spans)),
            key=lambda idx: (
                -max(0.0, min(we, spans[idx][1]) - max(ws, spans[idx][0])),
                abs(midpoint - (spans[idx][0] + spans[idx][1]) / 2.0),
            ),
        )
        groups[best].append(dict(word))
    result = []
    for items in groups:
        if not items:
            continue
        out = dict(seg)
        out["text"] = join_words(items)
        out["words"] = items
        out["start"] = round(max(start, min(float(word["start"]) for word in items)), 2)
        out["end"] = round(min(end, max(float(word["end"]) for word in items)), 2)
        result.append(out)
    return result


def _validate_words(raw_words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    if not raw_words:
        return []
    cleaned: list[dict] = []
    prev_end = seg_start - 0.05
    short_cnt = 0
    dup_cnt = 0
    for w in raw_words:
        txt = str(w.get("word") or "")
        if not txt.strip():
            continue
        ws, we, prob = w.get("start"), w.get("end"), w.get("probability")
        if ws is None or we is None or not (math.isfinite(ws) and math.isfinite(we)):
            return []
        ws_f = max(seg_start, float(ws))
        we_f = max(ws_f, min(seg_end + 0.25, float(we)))
        if ws_f < prev_end - 0.15:
            return []
        if we_f - ws_f < 0.03:
            short_cnt += 1
        if cleaned and abs(ws_f - float(cleaned[-1]["start"])) < 0.015 and abs(we_f - float(cleaned[-1]["end"])) < 0.015:
            dup_cnt += 1
        prev_end = ws_f
        p_val = float(prob) if prob is not None and math.isfinite(prob) else None
        if (
            cleaned and not txt[0].isspace()
            and not CJK_CHAR.search(txt) and not CJK_CHAR.search(cleaned[-1]["word"])
            and _is_mostly_vocalise(txt) and _is_mostly_vocalise(cleaned[-1]["word"])
        ):
            # ASR word entries are separate articulated syllables, not word pieces.
            # Keep display spacing as well as each syllable's timing for LRC export.
            txt = " " + txt
        if (
            cleaned
            and not txt[0].isspace()
            and not cleaned[-1]["word"][-1].isspace()
            and not CJK_CHAR.search(txt)
            and not CJK_CHAR.search(cleaned[-1]["word"])
            and re.match(r"^[A-Za-z0-9'’\-]", txt)
            and re.search(r"[A-Za-z0-9'’\-]$", cleaned[-1]["word"])
            and (ws_f - float(cleaned[-1]["end"])) <= 0.18
        ):
            cleaned[-1]["word"] += txt
            cleaned[-1]["end"] = round(max(float(cleaned[-1]["end"]), we_f), 2)
            continue
        p_round = round(p_val, 3) if p_val is not None else None
        cleaned.append(
            {
                "word": txt,
                "start": round(ws_f, 2),
                "end": round(max(ws_f + 0.02, we_f), 2),
                "probability": p_round,
                "uncertain": bool(p_round is not None and p_round < 0.30),
                **({"timing_source": w["timing_source"]} if w.get("timing_source") else {}),
            }
        )
    if not cleaned or dup_cnt >= 2 or (len(cleaned) >= 3 and short_cnt / len(cleaned) > 0.45):
        return []
    return cleaned


def _split_segment_into_lyric_lines(
    text: str,
    s_time: float,
    e_time: float,
    valid_words: list[dict],
) -> list[tuple[str, float, float, list[dict]]]:
    """Unified lyric line-splitter delegating to `music_asr_engine._group_words_into_lyric_lines`."""
    from .music_asr_engine import _group_words_into_lyric_lines

    return _group_words_into_lyric_lines(
        valid_words,
        text,
        s_time,
        e_time,
        is_vocalise_chunk=_is_mostly_vocalise(text),
    )


# ================================================================== Mode 2: User Reference Lyrics Alignment
def _normalize_char_for_match(ch: str) -> str:
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    if 0xAC00 <= code <= 0xD7A3:
        return unicodedata.normalize("NFD", ch)
    if ch in ("đ", "Đ"):
        return "d"
    return "".join(c for c in unicodedata.normalize("NFD", ch.lower()) if unicodedata.category(c) != "Mn")


def _tokenize_line_for_alignment(text: str, lang: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if CJK_CHAR.search(text):
        return [t for t in re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][ー々ゝゞ]*\s*|[A-Za-z0-9']+\s*|[^\s]", text) if t]
    return [p for p in re.findall(r"\S+\s*", text) if p.strip()]


def parse_user_lyrics(lyrics_text: str) -> list[dict[str, str | None]]:
    lines_out: list[dict[str, str | None]] = []
    current_section: str | None = None
    current_stream: str = DEFAULT_STREAM_ID

    def add_line(raw: str, copies: int = 1) -> None:
        nonlocal current_section, current_stream
        s = raw.strip().lstrip("\ufeff")
        if not s:
            return
        sm = STREAM_HEADER_RE.match(s)
        if sm:
            current_stream = normalize_stream_id(sm.group(1))
            return
        m = SECTION_HEADER_RE.match(s)
        if m:
            raw_tag = re.sub(r"\s+", " ", m.group(1).strip().lower())
            base_tag = re.sub(r"\s*\d+$", "", raw_tag)
            canon = SECTION_CANONICAL.get(base_tag)
            if canon:
                suf = re.search(r"(\d+)$", raw_tag)
                current_section = f"{canon} {suf.group(1)}" if suf else canon
            else:
                current_section = m.group(1).strip().title()
            return
        lines_out.extend({"text": s, "section": current_section, "stream_id": current_stream} for _ in range(copies))

    normalized = lyrics_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    srt_time = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$")
    if any(srt_time.match(line) for line in normalized.splitlines()):
        for block in re.split(r"\n\s*\n", normalized.strip()):
            content = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().isdigit() and not srt_time.match(l)]
            if content:
                add_line(" ".join(content))
    else:
        lrc_time = re.compile(r"\[(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,:]\d{1,3})?\]")
        lrc_meta = re.compile(r"^\[(?:ti|ar|al|by|la|length|offset|re|ve):.*\]$", re.IGNORECASE)
        for raw in normalized.splitlines():
            s = raw.strip()
            if not s or lrc_meta.match(s):
                continue
            stamps = 0
            while (match := lrc_time.match(s)) is not None:
                stamps += 1
                s = s[match.end() :].lstrip()
            add_line(s, copies=stamps or 1)
    return lines_out


def align_user_lyrics_to_anchors(
    user_lines: list[dict[str, str | None]],
    anchors: list[dict],
    rms_env: np.ndarray,
    vocal_floor: float,
    duration: float,
    primary_lang: str,
) -> tuple[list[dict], str]:
    if not user_lines:
        return [], "line"
    user_words_meta: list[dict] = []
    line_word_meta: list[list[dict]] = [[] for _ in user_lines]
    for li, item in enumerate(user_lines):
        lt = str(item["text"])
        for tok in _tokenize_line_for_alignment(lt, infer_segment_script_lang(lt, primary_lang)):
            norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in tok) if c.isalnum())
            wm = {"line_idx": li, "display": tok, "norm": norm}
            user_words_meta.append(wm)
            line_word_meta[li].append(wm)

    ref_has_cjk = any(CJK_CHAR.search(str(item.get("text") or "")) for item in user_lines)
    anchor_meta: list[dict] = []
    for a in anchors:
        wt = str(a.get("word") or "")
        norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in wt) if c.isalnum())
        if not norm:
            continue
        ws, we = float(a["start"]), max(float(a["start"]) + 0.02, float(a["end"]))
        p_raw = a.get("probability")
        prob = min(1.0, max(0.0, float(p_raw))) if (p_raw is not None and math.isfinite(p_raw)) else 0.78
        if ref_has_cjk and CJK_CHAR.search(wt) and len(norm) > 1:
            step = (we - ws) / len(norm)
            for ci, ch in enumerate(norm):
                anchor_meta.append({"norm": ch, "start": ws + ci * step, "end": ws + (ci + 1) * step, "probability": prob})
        else:
            anchor_meta.append({"norm": norm, "start": ws, "end": we, "probability": prob})

    sim_cache: dict[tuple[str, str], float] = {}

    def token_sim(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        k = (left, right)
        if k in sim_cache:
            return sim_cache[k]
        if 2 * min(len(left), len(right)) / (len(left) + len(right)) < 0.45 or min(len(left), len(right)) <= 1:
            return 0.0
        ratio = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
        if left.startswith(right) or right.startswith(left):
            ratio = max(ratio, 0.72)
        if len(sim_cache) < 100_000:
            sim_cache[k] = ratio
        return ratio

    uc, ac = len(user_words_meta), len(anchor_meta)
    user_to_anchor: list[int | None] = [None] * uc
    match_quality: list[float] = [0.0] * uc
    if uc and ac:
        user_gap = 1.05
        anchor_gaps = [0.16 + 0.62 * float(a["probability"]) for a in anchor_meta]

        def pair_cost(sim: float, prob: float) -> float:
            cp = 0.46 * (1.0 - prob)
            if sim >= 0.999:
                return 0.04 + cp
            if sim >= 0.72:
                return 0.18 + (1.0 - sim) * 0.90 + cp
            if sim >= 0.45:
                return 0.85 + (1.0 - sim) * 0.80 + cp
            return 1.75 + cp

        trace_dp = [bytearray(ac + 1) for _ in range(uc + 1)]
        prev = [0.0] * (ac + 1)
        for j in range(1, ac + 1):
            prev[j] = prev[j - 1] + anchor_gaps[j - 1]
            trace_dp[0][j] = 3
        for i in range(1, uc + 1):
            cur = [0.0] * (ac + 1)
            cur[0] = prev[0] + user_gap
            trace_dp[i][0] = 2
            ln = str(user_words_meta[i - 1]["norm"])
            for j in range(1, ac + 1):
                sim = token_sim(ln, str(anchor_meta[j - 1]["norm"]))
                best, op = prev[j - 1] + pair_cost(sim, float(anchor_meta[j - 1]["probability"])), 1
                sa, su = cur[j - 1] + anchor_gaps[j - 1], prev[j] + user_gap
                if sa < best - 1e-9:
                    best, op = sa, 3
                if su < best - 1e-9:
                    best, op = su, 2
                cur[j], trace_dp[i][j] = best, op
            prev = cur

        i, j = uc, ac
        while i > 0 or j > 0:
            op = trace_dp[i][j]
            if i > 0 and j > 0 and op == 1:
                sim = token_sim(str(user_words_meta[i - 1]["norm"]), str(anchor_meta[j - 1]["norm"]))
                if sim >= 0.45:
                    user_to_anchor[i - 1], match_quality[i - 1] = j - 1, sim
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or op == 3):
                j -= 1
            elif i > 0:
                i -= 1
            else:
                break

    line_words: list[list[dict | None]] = [[] for _ in user_lines]
    line_matched_times: list[list[tuple[float, float, float]]] = [[] for _ in user_lines]
    total_words, anchored_words = 0, 0
    for idx_u, wm in enumerate(user_words_meta):
        li = wm["line_idx"]
        total_words += 1
        ai = user_to_anchor[idx_u]
        sim = match_quality[idx_u]
        if ai is not None and ai < ac:
            anc = anchor_meta[ai]
            ws, we = float(anc["start"]), float(anc["end"])
            mp = round(float(anc["probability"]) * (0.55 + 0.45 * sim), 3)
            line_words[li].append(
                {"word": wm["display"], "start": round(ws, 2), "end": round(max(ws + 0.04, we), 2),
                 "probability": mp, "uncertain": mp < 0.48 or sim < 0.72, "timing_source": "user_aligned"}
            )
            line_matched_times[li].append((ws, we, mp))
            anchored_words += 1
        else:
            line_words[li].append(None)

    active_idxs = np.where(rms_env >= vocal_floor)[0]
    first_voc = float(active_idxs[0] * 0.05) if active_idxs.size else 0.0
    last_voc = float(min(duration, (active_idxs[-1] + 1) * 0.05)) if active_idxs.size else max(1.0, duration)
    line_bounds = [
        (m[0][0], max(m[0][0] + 0.25, m[-1][1]), round(float(np.mean([x[2] for x in m])), 3), True) if m else (-1.0, -1.0, 0.35, False)
        for m in line_matched_times
    ]
    n_lines, i = len(user_lines), 0
    while i < n_lines:
        if line_bounds[i][3]:
            i += 1
            continue
        j = i
        while j < n_lines and not line_bounds[j][3]:
            j += 1
        left_t = line_bounds[i - 1][1] + 0.15 if i > 0 else first_voc
        right_t = line_bounds[j][0] - 0.10 if j < n_lines else max(left_t + 0.5 * (j - i), last_voc)
        if right_t <= left_t:
            right_t = min(max(duration, left_t + 0.8), left_t + 1.6 * (j - i))
        step = max(0.4, right_t - left_t) / (j - i)
        for k in range(j - i):
            ls = left_t + k * step
            line_bounds[i + k] = (ls, max(ls + 0.2, left_t + (k + 1) * step - 0.06), 0.35, False)
        i = j

    for li, lw in enumerate(line_words):
        matched = [idx for idx, w in enumerate(lw) if w is not None]
        if not matched:
            continue
        lmeta = line_word_meta[li]
        char_rate = min(0.12, max(0.025, float(np.median([max(0.0, float(lw[idx]["end"]) - float(lw[idx]["start"])) / max(1, len(lmeta[idx]["norm"])) for idx in matched]))))
        weights = [max(1, len(m["norm"])) for m in lmeta]

        def add_est(idxs: list[int], start: float, end: float) -> None:
            if not idxs or end - start < 0.04 * len(idxs):
                return
            tw, cur, cum = sum(weights[k] for k in idxs), start, 0
            for k in idxs:
                cum += weights[k]
                we = min(end, max(cur + 0.02, start + (end - start) * cum / tw))
                lw[k] = {"word": lmeta[k]["display"], "start": round(cur, 2), "end": round(we, 2), "probability": 0.12, "uncertain": True, "timing_source": "interpolated"}
                cur = we

        for l_idx, r_idx in zip(matched, matched[1:]):
            if r_idx - l_idx > 1:
                add_est(list(range(l_idx + 1, r_idx)), float(lw[l_idx]["end"]), float(lw[r_idx]["start"]))
        f_idx, l_idx = matched[0], matched[-1]
        l_floor = max(0.0, line_bounds[li - 1][1] if li > 0 else 0.0)
        l_ceil = min(duration, line_bounds[li + 1][0] if li + 1 < n_lines else duration)
        if f_idx > 0:
            idxs = list(range(f_idx))
            add_est(idxs, max(l_floor, float(lw[f_idx]["start"]) - char_rate * sum(weights[k] for k in idxs)), float(lw[f_idx]["start"]))
        if l_idx + 1 < len(lw):
            idxs = list(range(l_idx + 1, len(lw)))
            add_est(idxs, float(lw[l_idx]["end"]), min(l_ceil, float(lw[l_idx]["end"]) + char_rate * sum(weights[k] for k in idxs)))
        inferred = [w for w in lw if w is not None]
        if inferred:
            ls, le = min(float(w["start"]) for w in inferred), max(float(w["end"]) for w in inferred)
            line_bounds[li] = (ls, max(ls + 0.25, le), round(float(np.mean([float(w.get("probability") or 0.0) for w in inferred])), 3), line_bounds[li][3])

    segments: list[dict] = []
    any_word_level = False
    for li, item in enumerate(user_lines):
        ls, le, conf, has_anchor = line_bounds[li]
        lw = line_words[li]
        words_clean = _validate_words([w for w in lw if w is not None], ls, le) if (lw and all(w is not None for w in lw)) else []
        if words_clean:
            any_word_level = True
        txt = str(item["text"])
        ev_type, ev_adlib, lang_hint = classify_vocal_event(txt, fallback_lang=infer_segment_script_lang(txt, primary_lang))
        if ev_type in ("humming", "unknown_vocal"):
            words_clean = []
        seg_lang = None if ev_type in ("humming", "unknown_vocal") else (lang_hint or infer_segment_script_lang(txt, primary_lang))
        reasons = ["unmatched_reference_line"] if not has_anchor else (["uncertain_words"] if (conf < 0.50 or any(w.get("uncertain") for w in words_clean)) else [])
        segments.append(
            {
                "id": li + 1, "start": round(max(0.0, ls), 2), "end": round(max(ls + 0.15, le), 2),
                "text": txt, "text_raw": txt, "text_edited": None, "event_type": ev_type,
                "vocal_role": "unknown", "is_adlib": ev_adlib, "stream_id": normalize_stream_id(item.get("stream_id")),
                "language": seg_lang, "lang": str(seg_lang or ""), "section": item.get("section"),
                "confidence": conf if has_anchor else None,
                "low_confidence": not has_anchor or conf < 0.50 or any(w.get("uncertain") for w in words_clean),
                "review_status": "needs_review" if (not has_anchor or conf < 0.50) else None,
                "review_reasons": reasons, "review_reason": format_review_reason(reasons),
                "timing_source": "user_aligned" if (has_anchor and words_clean) else ("interpolated" if has_anchor else "acoustic_region"),
                "timing_granularity": "word" if words_clean else ("region" if ev_type in ("humming", "unknown_vocal") else "line"),
                "source_audio_variant": "vocal_stem", "words": words_clean, "candidates": [],
            }
        )
    return assign_overlap_groups(segments), ("word" if any_word_level and anchored_words >= max(1, total_words // 3) else "line")


# ================================================================== Structure, Chords & Project Store
def _phrase_trigrams(text: str) -> set[str]:
    norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in text) if c.isalnum())
    return ({norm[i : i + 3] for i in range(len(norm) - 2)} if len(norm) >= 3 else ({norm} if norm else set()))


def annotate_conservative_structure(segments: list[dict], duration: float) -> list[dict]:
    if len(segments) < 5 or any(s.get("section") for s in segments):
        return segments
    stanzas: list[list[int]] = []
    cur: list[int] = [0]
    for idx in range(1, len(segments)):
        if float(segments[idx]["start"]) - float(segments[idx - 1]["end"]) >= 2.2 or len(cur) >= 6:
            stanzas.append(cur)
            cur = [idx]
        else:
            cur.append(idx)
    if cur:
        stanzas.append(cur)
    if len(stanzas) < 3:
        return segments
    grams = [_phrase_trigrams(" ".join(segments[i]["text"] for i in st)) for st in stanzas]
    chorus_stanzas: set[int] = set()
    for i in range(len(stanzas)):
        if len(stanzas[i]) < 2:
            continue
        for j in range(i + 1, len(stanzas)):
            if len(stanzas[j]) < 2:
                continue
            if float(segments[stanzas[j][0]]["start"]) - float(segments[stanzas[i][0]]["start"]) >= 14.0:
                a, b = grams[i], grams[j]
                if a and b and (2.0 * len(a & b) / (len(a) + len(b))) >= 0.72:
                    chorus_stanzas.update((i, j))
    if not chorus_stanzas:
        return segments
    out = [dict(s) for s in segments]
    v_num, last_ch = 1, max(chorus_stanzas)
    for si, st in enumerate(stanzas):
        fi = st[0]
        if si in chorus_stanzas:
            out[fi]["section"] = "Chorus"
        elif si == len(stanzas) - 1 and si > last_ch and (duration - float(out[fi]["start"])) <= 35.0:
            out[fi]["section"] = "Outro"
        elif len(st) >= 2 and si < last_ch:
            out[fi]["section"] = f"Verse {v_num}"
            v_num += 1
    return out


def attach_chords(segments: list[dict], chords: list[tuple[float, str]] | None) -> list[dict]:
    if not chords:
        return segments
    chord_list = sorted(chords, key=lambda c: c[0])

    def chord_at(sec: float) -> str | None:
        if not chord_list or sec < chord_list[0][0] - 0.25:
            return None
        lo, hi = 0, len(chord_list) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if chord_list[mid][0] <= sec + 0.15:
                lo = mid
            else:
                hi = mid - 1
        return chord_list[lo][1]

    return [{**seg, "chord": seg.get("chord") or chord_at(float(seg["start"]))} for seg in segments]


def build_full_text(segments: list[dict], include_sections: bool = False) -> str:
    lines: list[str] = []
    for seg in segments:
        sec = seg.get("section")
        if include_sections and sec:
            if lines:
                lines.append("")
            lines.append(f"[{sec}]")
        lines.append(str(seg.get("text") or "").strip())
    return "\n".join(lines).strip()


class ProjectLyricsStore:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.dir = project.dir / "lyrics"
        self.state_file = self.dir / "state.json"
        self.transcript_file = self.dir / "transcript.json"

    def get(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {
                "project_id": self.project.id, "title": self.project.data.get("title", ""),
                "has_audio": bool(self.project.audio_path and self.project.audio_path.is_file()),
                "state": "idle", "mode": "extract", "error": None, "language": None, "language_name": None,
                "language_confidence": None, "detected_languages": [], "vocal_isolated": False,
                "vocal_cached": False, "alignment_method": None, "segment_count": 0, "duration": None,
                "updated": None, "transcript": None,
            }
        data = read_json(self.state_file)
        data["project_id"] = self.project.id
        data["title"] = self.project.data.get("title", data.get("title", ""))
        data["has_audio"] = bool(self.project.audio_path and self.project.audio_path.is_file())
        if self.transcript_file.is_file():
            try:
                data["transcript"] = migrate_transcript(read_json(self.transcript_file))
            except Exception:
                data["transcript"] = None
        else:
            data["transcript"] = None
        return data

    def summary(self) -> dict[str, Any] | None:
        if not self.state_file.is_file():
            return None
        try:
            s = read_json(self.state_file)
            return {k: s.get(k) for k in ("state", "mode", "language", "language_name", "segment_count", "error", "updated")}
        except Exception:
            return None

    def update(self, transcript: dict | None = None, **fields: Any) -> dict[str, Any]:
        self.dir.mkdir(parents=True, exist_ok=True)
        cur = read_json(self.state_file) if self.state_file.is_file() else {"created": now()}
        cur.update(fields)
        cur["updated"] = now()
        write_json(self.state_file, cur)
        if transcript is not None:
            write_json(self.transcript_file, transcript)
        return self.get()

    def clear(self) -> None:
        if self.dir.is_dir():
            shutil.rmtree(self.dir, ignore_errors=True)

    def recover(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            s = read_json(self.state_file)
            if s.get("state") in ("queued", "running"):
                s["state"], s["error"] = "cancelled", "Máy chủ khởi động lại nên tác vụ đã dừng."
                write_json(self.state_file, s)
        except Exception:
            pass


# ================================================================== Main Pipeline & Editing Entrypoints
def execute_lyrics_pipeline(
    *,
    engine: Engine,
    cache: LyricsCache,
    source_audio: Path,
    audio_sha1: str | None = None,
    track_title: str = "",
    mode: str = "extract",
    user_lyrics_text: str = "",
    requested_language: str = "auto",
    model_id: str = DEFAULT_MODEL_ID,
    isolate_vocals: bool = True,
    force_separate: bool = False,
    chords: list[tuple[float, str]] | None = None,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> dict[str, Any]:
    ensure_windows_runtime()
    if not source_audio.is_file():
        raise FileNotFoundError("Không tìm thấy tệp âm thanh gốc để xử lý lời bài hát.")
    if source_audio.suffix.lower() in (".mid", ".midi"):
        raise ValueError("Không thể trích xuất lời hát từ tệp Piano MIDI. Cần tệp âm thanh gốc chứa giọng hát.")
    if should_stop():
        raise Cancelled()
    if hasattr(engine, "unload"):
        engine.unload()

    model_id = normalize_model_id(model_id)
    req_lang = (requested_language or "auto").strip().lower()
    progress(0.05, "preparing_audio")
    log(f"Bước 1/5 · Chuẩn bị âm thanh & khởi tạo mô hình Songscription ({model_id})...")
    sha1 = audio_sha1 or file_sha1(source_audio)

    # GPU execution: separation, recognition, recovery and alignment share one worker.
    from .modal_client import is_modal_enabled, modal_extract_lyrics

    use_modal = is_modal_enabled()
    try:
        if use_modal:
            # The synchronous worker returns its stage logs on completion. While
            # waiting, do not claim that separation is the stage currently running.
            progress(None, "processing_gpu")
            log(f"☁ Đang kết nối Modal.com GPU NVIDIA L4 ({model_id})...")
            cached_voc = cache.vocal_path(sha1) if (isolate_vocals and not force_separate and cache.has_vocal(sha1)) else None
            modal_res = modal_extract_lyrics(
                source_audio=source_audio, vocal_wav_path=cached_voc, mode=mode, user_lyrics_text=user_lyrics_text,
                requested_language=req_lang, model_id=model_id, isolate_vocals=isolate_vocals, chords=chords,
                audio_sha1=sha1, track_title=track_title,
            )
            if should_stop():
                raise Cancelled()
            for m_log in modal_res.get("logs") or []:
                log(str(m_log))
            voc_bytes = modal_res.get("vocal_wav_bytes")
            if voc_bytes:
                vp = cache.vocal_path(sha1)
                vp.parent.mkdir(parents=True, exist_ok=True)
                vp.write_bytes(voc_bytes)
            inst_bytes = modal_res.get("instrumental_audio_bytes")
            if inst_bytes:
                inst_ext = str(modal_res.get("instrumental_ext") or ".mp3")
                cache.save_instrumental_bytes(sha1, inst_bytes, ext=inst_ext)
            elif isolate_vocals and cache.has_vocal(sha1) and not cache.has_instrumental(sha1):
                ensure_instrumental_cached(cache, source_audio, sha1)
            for st_id, st_bytes in (modal_res.get("stream_wav_bytes") or {}).items():
                if st_bytes:
                    sp = cache.stream_vocal_path(sha1, st_id)
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_bytes(st_bytes)
            tr = modal_res["transcript"]
            if modal_res.get("anchors"):
                cfg_hash = tr.get("config_hash") or compute_pipeline_config_hash(
                    model_id=model_id, requested_lang=req_lang, vocal_isolated=bool(tr.get("vocal_isolated", isolate_vocals))
                )
                cache.save_anchors(
                    sha1, req_lang,
                    {
                        "model": model_id, "vocal_isolated": tr.get("vocal_isolated", isolate_vocals),
                        "language": tr.get("language"), "language_confidence": tr.get("language_confidence"),
                        "detected_languages": tr.get("detected_languages") or [], "anchors": modal_res["anchors"],
                    },
                    model_id=model_id, vocal_isolated=bool(tr.get("vocal_isolated", isolate_vocals)), config_hash=cfg_hash,
                )
            progress(0.96, "saving_transcript")
            log(f"☁ Hoàn tất trên Modal GPU L4 ({len(tr.get('segments') or [])} sự kiện · {tr.get('language_name')})!")
            return tr
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Songscription trên Modal thất bại ({type(exc).__name__}). "
            "Kiểm tra kết nối, quyền truy cập và bản triển khai harmony-auralis-gpu trong Modal. "
            "GPU máy không được sử dụng."
        ) from exc

    # Local execution uses the same selected recognizer and recovery path as Modal.
    with tempfile.TemporaryDirectory(prefix="harmony-lyrics-") as tmp_str:
        tmp_dir = Path(tmp_str)
        mix_wav = tmp_dir / "mix_16k.wav"
        convert_audio_ffmpeg(source_audio, mix_wav, sample_rate=16000, channels=1, vocal_bandpass=False)
        mix_16k = load_wav_16k(mix_wav)
        duration = round(float(mix_16k.shape[0]) / 16000.0, 2)
        if should_stop():
            raise Cancelled()

        asr_wav, vocal_isolated, vocal_cached = isolate_vocals_cached(
            engine=engine, cache=cache, source_audio=source_audio, audio_sha1=sha1, tmp_dir=tmp_dir,
            want_isolate=isolate_vocals, force_separate=force_separate, progress=progress, should_stop=should_stop, log=log,
        )
        full_vocal = load_wav_16k(asr_wav)
        n_min = min(full_vocal.shape[0], mix_16k.shape[0])
        full_vocal, mix_16k = full_vocal[:n_min], mix_16k[:n_min]
        pure_rms_env, pure_vocal_floor = compute_vocal_envelope(full_vocal)
        pure_vocal_norm = _normalize_vocal_rms(full_vocal) if vocal_isolated else None
        vocal_rms = float(np.sqrt(np.mean(np.square(full_vocal, dtype=np.float64)) + 1e-12))
        mix_rms = float(np.sqrt(np.mean(np.square(mix_16k, dtype=np.float64)) + 1e-12))
        if vocal_isolated and vocal_rms >= 0.0018:
            asr_audio = pure_vocal_norm
        elif mode == "align" and vocal_isolated and vocal_rms < 0.0018 and mix_rms > 0.008:
            asr_audio = _blend_vocal_with_mix(full_vocal, mix_16k, mix_weight=0.25)
        else:
            asr_audio = pure_vocal_norm if vocal_isolated else _normalize_vocal_rms(mix_16k)

        cfg_hash = compute_pipeline_config_hash(model_id=model_id, requested_lang=req_lang, vocal_isolated=vocal_isolated)
        cached_anchor_data = (
            cache.load_anchors(sha1, req_lang, model_id=model_id, vocal_isolated=vocal_isolated, config_hash=cfg_hash)
            if (mode == "align" and not force_separate) else None
        )
        raw_asr_segments: list[dict] = []
        diagnostics: dict[str, Any] = {}
        if cached_anchor_data is not None:
            primary_lang = req_lang if (req_lang and req_lang != "auto") else str(cached_anchor_data.get("language") or "en")
            lang_conf = float(cached_anchor_data.get("language_confidence") or 1.0)
            detected_languages = list(cached_anchor_data.get("detected_languages") or [])
            anchors = list(cached_anchor_data.get("anchors") or [])
            rms_env, vocal_floor, raw_segments = pure_rms_env, pure_vocal_floor, []
        else:
            asr_out = run_music_asr_pipeline(
                asr_audio=asr_audio, duration=duration, vocal_isolated=vocal_isolated,
                rms_env=pure_rms_env, vocal_floor=pure_vocal_floor, requested_lang=req_lang,
                model_id=model_id, progress=progress, should_stop=should_stop, log=log,
                pure_vocal_audio=pure_vocal_norm,
                original_audio=mix_16k, audio_sha1=sha1, track_title=track_title,
                mode=mode, user_lyrics_text=user_lyrics_text,
            )
            primary_lang, lang_conf = asr_out["language"], asr_out["language_confidence"]
            detected_languages, raw_segments, anchors = asr_out["detected_languages"], asr_out["raw_segments"], asr_out["anchors"]
            raw_asr_segments = list(asr_out.get("raw_asr_segments") or [])
            diagnostics = dict(asr_out.get("diagnostics") or {})
            diagnostics["vocal_separation"] = {
                "isolated": vocal_isolated, "cached": vocal_cached,
                "provider": "bs-roformer-local" if vocal_isolated else "raw-mix-16k",
                "vocal_floor": round(float(pure_vocal_floor), 5), "duration_sec": duration,
            }
            rms_env, vocal_floor = asr_out["rms_env"], asr_out["vocal_floor"]
            cache.save_anchors(
                sha1, req_lang,
                {"model": model_id, "vocal_isolated": vocal_isolated, "language": primary_lang,
                 "language_confidence": lang_conf, "detected_languages": detected_languages, "anchors": anchors},
                model_id=model_id, vocal_isolated=vocal_isolated, config_hash=cfg_hash,
            )

        progress(0.88, "aligning_timestamps")
        if mode == "align" and not diagnostics.get("reference_lyrics"):
            parsed = parse_user_lyrics(user_lyrics_text)
            if not parsed:
                raise ValueError("Nội dung lời bài hát trống. Vui lòng nhập lời bài hát để căn chỉnh.")
            if req_lang in ("", "auto"):
                primary_lang = infer_segment_script_lang(" ".join(x["text"] or "" for x in parsed[:8]), primary_lang or "en")
            segments, alignment_method = align_user_lyrics_to_anchors(parsed, anchors, rms_env, vocal_floor, duration, primary_lang)
            segments = annotate_conservative_structure(segments, duration)
        else:
            segments = [migrate_segment({**seg, "id": idx + 1}, fallback_lang=primary_lang) for idx, seg in enumerate(raw_segments)]
            segments = assign_overlap_groups(annotate_conservative_structure(segments, duration))
            alignment_method = "word" if any(bool(s.get("words")) for s in segments) else "line"

        segments = attach_chords(segments, chords)
        streams_present = sorted({str(s.get("stream_id") or DEFAULT_STREAM_ID) for s in segments}) or [DEFAULT_STREAM_ID]
        progress(0.96, "saving_transcript")
        return {
            "schema_version": SCHEMA_VERSION, "mode": mode, "model": model_id, "config_hash": cfg_hash, "audio_sha1": sha1,
            "language": primary_lang, "language_name": language_label(primary_lang),
            "language_confidence": round(float(lang_conf), 4), "detected_languages": detected_languages,
            "is_multilingual": len({s.get("language") for s in segments if s.get("language")}) > 1,
            "vocal_isolated": vocal_isolated, "vocal_cached": vocal_cached, "alignment_method": alignment_method,
            "duration": duration,
            "streams": [{"id": sid, "label": stream_label(sid), "has_isolated_audio": cache.has_stream_vocal(sha1, sid)} for sid in streams_present],
            "multi_singer_separation": get_multi_singer_separation_status(),
            "segments": segments, "raw_asr_segments": raw_asr_segments, "diagnostics": diagnostics, "edit_history": [],
            "full_text": build_full_text(segments, include_sections=False),
            "structured_text": build_full_text(segments, include_sections=True),
        }


def sanitize_edited_segments(raw_segments: list[dict], fallback_lang: str = "en") -> list[dict]:
    cleaned: list[dict] = []
    for seg in raw_segments:
        txt = str(seg.get("text") or "").strip()
        ev_type = seg.get("event_type") if seg.get("event_type") in EVENT_TYPES else None
        if not txt and ev_type != "unknown_vocal":
            continue
        try:
            s_time = max(0.0, round(float(seg.get("start") or 0.0), 2))
        except (TypeError, ValueError):
            s_time = 0.0
        try:
            e_time = max(round(s_time + 0.05, 2), round(float(seg.get("end") or (s_time + 2.0)), 2))
        except (TypeError, ValueError):
            e_time = round(s_time + 2.0, 2)

        text_raw_str = str(seg.get("text_raw") if seg.get("text_raw") is not None else txt)
        text_edited = txt if txt != text_raw_str else seg.get("text_edited")

        if ev_type is None:
            ev_type, inferred_adlib, inferred_lang = classify_vocal_event(
                txt, is_adlib_hint=seg.get("is_adlib") if isinstance(seg.get("is_adlib"), bool) else None, fallback_lang=fallback_lang
            )
        else:
            inferred_adlib = seg.get("is_adlib") if isinstance(seg.get("is_adlib"), bool) else None
            inferred_lang = None

        raw_language = seg.get("language") if "language" in seg else seg.get("lang")
        language = (
            None if (raw_language in ("", None) and ev_type in ("humming", "unknown_vocal"))
            else (inferred_lang or infer_segment_script_lang(txt, fallback_lang)) if raw_language in ("", None)
            else str(raw_language).strip().lower()
        )
        orig_words = seg.get("words") if isinstance(seg.get("words"), list) else []
        valid_words: list[dict] = []
        if orig_words and ev_type not in ("humming", "unknown_vocal"):
            if "".join(str(w.get("word") or "").strip() for w in orig_words) == re.sub(r"\s+", "", txt):
                valid_words = _validate_words(orig_words, s_time - 0.05, e_time + 0.25)

        review_status = seg.get("review_status") if seg.get("review_status") in REVIEW_STATUSES else None
        review_reasons = [str(r) for r in (seg.get("review_reasons") or []) if str(r).strip()]
        timing_source = str(seg.get("timing_source") or ("forced_aligner" if valid_words else "manual"))
        cleaned.append(
            {
                "id": len(cleaned) + 1, "start": s_time, "end": e_time,
                "text": txt, "text_raw": text_raw_str, "text_edited": str(text_edited) if text_edited is not None else None,
                "event_type": ev_type,
                "vocal_role": seg.get("vocal_role") if seg.get("vocal_role") in VOCAL_ROLES else "unknown",
                "is_adlib": seg.get("is_adlib") if isinstance(seg.get("is_adlib"), bool) else inferred_adlib,
                "stream_id": normalize_stream_id(seg.get("stream_id")), "overlap_group_id": None,
                "language": language, "lang": str(language or ""),
                "section": str(seg["section"]).strip() if seg.get("section") else None,
                "chord": seg.get("chord") or None,
                "confidence": None if (ev_type == "unknown_vocal" and timing_source == "acoustic_region") else seg.get("confidence"),
                "low_confidence": bool(seg.get("low_confidence", False)),
                "review_status": review_status, "review_reasons": review_reasons,
                "review_reason": str(seg.get("review_reason") or "").strip() or format_review_reason(review_reasons),
                "timing_source": timing_source,
                "timing_granularity": str(seg.get("timing_granularity") or ("word" if valid_words else ("region" if ev_type in ("humming", "unknown_vocal") else "line"))),
                "source_audio_variant": str(seg.get("source_audio_variant") or "vocal_stem"),
                "words": valid_words,
                "candidates": seg.get("candidates") if isinstance(seg.get("candidates"), list) else [],
            }
        )
    return assign_overlap_groups(cleaned)


def retranscribe_flagged_segments(
    *,
    cache: LyricsCache,
    source_audio: Path,
    audio_sha1: str | None,
    segments: list[dict],
    target_ids: list[int] | None = None,
    language: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    isolate_vocals: bool = True,
) -> tuple[list[dict], list[dict]]:
    ensure_windows_runtime()
    if not segments:
        return [], []
    target_set = set(target_ids) if target_ids else None
    target_indices = [
        idx
        for idx, seg in enumerate(segments)
        if (
            (int(seg.get("id") or (idx + 1)) in target_set)
            if target_set is not None
            else (
                seg.get("review_status") == "needs_review"
                or (seg.get("review_status") not in ("verified", "edited", "skipped") and bool(seg.get("low_confidence")))
            )
        )
    ]
    if not target_indices:
        return list(segments), []

    eff_model = normalize_model_id(model_id)
    sha1 = audio_sha1 or file_sha1(source_audio)
    decode_lang = language if (language and language != "auto") else None

    from .modal_client import is_modal_enabled, modal_retranscribe_flagged

    use_modal = is_modal_enabled()
    if use_modal:
        try:
            cached_voc = cache.vocal_path(sha1) if (isolate_vocals and cache.has_vocal(sha1)) else None
            m_out = modal_retranscribe_flagged(
                source_audio=source_audio, vocal_wav_path=cached_voc, segments=segments,
                target_ids=target_ids, language=language, model_id=eff_model, isolate_vocals=isolate_vocals,
            )
            voc_bytes = m_out.get("vocal_wav_bytes")
            if voc_bytes and not cache.has_vocal(sha1):
                vp = cache.vocal_path(sha1)
                vp.parent.mkdir(parents=True, exist_ok=True)
                vp.write_bytes(voc_bytes)
            inst_bytes = m_out.get("instrumental_audio_bytes")
            if inst_bytes and not cache.has_instrumental(sha1, require_studio_stem=True):
                inst_ext = str(m_out.get("instrumental_ext") or ".mp3")
                cache.save_instrumental_bytes(sha1, inst_bytes, ext=inst_ext, studio_stem=True)
            return m_out["updated_segments"], m_out["results"]
        except Cancelled:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Đọc lại lời hát trên Modal thất bại ({type(exc).__name__}). "
                "Kiểm tra kết nối và bản triển khai harmony-auralis-gpu trong Modal. "
                "GPU máy không được sử dụng."
            ) from exc

    with tempfile.TemporaryDirectory(prefix="lyrics-refine-") as tmp:
        tmp_dir = Path(tmp)
        orig_wav = tmp_dir / "orig_16k.wav"
        convert_audio_ffmpeg(source_audio, orig_wav, sample_rate=16000)
        orig_16k = load_wav_16k(orig_wav)
        duration = float(orig_16k.shape[0]) / 16000.0
        pure_vocal, crisp_blend, has_vocal_stem = _normalize_vocal_rms(orig_16k), _normalize_vocal_rms(orig_16k), False
        if isolate_vocals and cache.has_vocal(sha1):
            voc_wav = tmp_dir / "voc_16k.wav"
            convert_audio_ffmpeg(cache.vocal_path(sha1), voc_wav, sample_rate=16000)
            voc_16k = load_wav_16k(voc_wav)
            n = min(voc_16k.shape[0], orig_16k.shape[0])
            if n > 0:
                voc_16k, orig_16k, duration = voc_16k[:n], orig_16k[:n], float(n) / 16000.0
                pure_vocal = _normalize_vocal_rms(voc_16k)
                crisp_blend = _blend_vocal_with_mix(voc_16k, orig_16k, mix_weight=0.08)
                has_vocal_stem = True
        return refine_segments_with_music_asr(
            segments=segments, target_indices=target_indices,
            pure_vocal=pure_vocal, crisp_blend=crisp_blend, duration=duration,
            has_vocal_stem=has_vocal_stem, decode_lang=decode_lang, model_id=eff_model,
        )
