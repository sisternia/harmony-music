"""Isolated AI Lyrics Transcription & Alignment Service for Harmony Auralis.

Pipeline:
  Audio -> Vocal Separation (BS-RoFormer cached by SHA-1 / FFmpeg vocal-band fallback)
        -> Speech Recognition (Hybrid FP16-Storage / FP32-Compute Whisper on CUDA + Faster-Whisper INT8 fallback)
        -> Word & Line Alignment (DTW word timing in Mode 1; Acoustic Anchor DP Alignment in Mode 2)
        -> Conservative Structure Detection & Persistent Per-Project Storage

Guarantees:
  - Never modifies original source audio or Tsumugi / Piano Cover pipelines.
  - Reuses cached vocal stems and acoustic anchors keyed by audio SHA-1.
  - Eliminates GTX 1650 (Turing sm_75, 4 GB VRAM) FP16 NaN logit crashes (`Categorical(logits: nan)`)
    by storing weights in FP16 (~1.76 GB VRAM) while executing all Linear/Conv1d/Decoder logits in FP32
    with finite logit clamping.
  - Suppresses hallucinations on instrumental sections via isolated-vocal RMS gating while preserving
    vocals that enter after an instrumental intro.
  - Never presents estimated word timestamps as confident: tokens missing from
    ASR are interpolated only inside measured gaps and marked `uncertain`, while
    lines without anchors remain line-level.
"""

from __future__ import annotations

import difflib
import gc
import json
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
from .project import Project, file_sha1, now, read_json, safe_name, write_json

VOCAL_WORKER = Path(__file__).resolve().parent / "lyrics_vocal_worker.py"

LYRICS_MODELS: tuple[dict[str, str], ...] = (
    {"id": "turbo", "label": "Whisper Large-v3 Turbo (Nhanh & Chuẩn · Khuyên dùng 4GB VRAM)"},
    {"id": "large-v3", "label": "Whisper Large-v3 (Độ chính xác tối đa)"},
    {"id": "medium", "label": "Whisper Medium (Nhẹ VRAM)"},
    {"id": "small", "label": "Whisper Small (Siêu nhanh / Tương thích CPU)"},
)

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

SECTION_CANONICAL: dict[str, str] = {
    "intro": "Intro",
    "mở đầu": "Intro",
    "イントロ": "Intro",
    "verse": "Verse",
    "aメロ": "Verse",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "prechorus": "Pre-Chorus",
    "bメロ": "Pre-Chorus",
    "chorus": "Chorus",
    "hook": "Chorus",
    "điệp khúc": "Chorus",
    "サビ": "Chorus",
    "大サビ": "Chorus",
    "bridge": "Bridge",
    "cメロ": "Bridge",
    "outro": "Outro",
    "kết": "Outro",
    "アウトロ": "Outro",
    "interlude": "Instrumental",
    "instrumental": "Instrumental",
    "間奏": "Instrumental",
}

VI_DIACRITICS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]")
JA_KANA = re.compile(r"[\u3040-\u30ff]")
KO_HANGUL = re.compile(r"[\uac00-\ud7af]")
ZH_HAN = re.compile(r"[\u4e00-\u9fff]")
CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_DLL_INIT_LOCK = threading.Lock()
_DLL_INITIALIZED = False
_WHISPER_PATCHED = False


def ensure_windows_runtime() -> Path | None:
    """Ensure bundled FFmpeg and PyTorch CUDA/cuBLAS DLLs are discoverable on Windows."""
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

        if not _DLL_INITIALIZED:
            try:
                import sys
                import torch

                extra_dirs: list[Path] = []
                torch_lib = Path(torch.__file__).resolve().parent / "lib"
                if torch_lib.is_dir():
                    extra_dirs.append(torch_lib)
                # Register nvidia/*/bin (cublas64_12.dll, cudnn, etc.) for CTranslate2 / Faster-Whisper on CUDA
                nv_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
                if nv_root.is_dir():
                    for bin_dir in nv_root.glob("*/bin"):
                        if bin_dir.is_dir():
                            extra_dirs.append(bin_dir.resolve())

                for d in extra_dirs:
                    d_str = str(d)
                    cur_path = os.environ.get("PATH", "")
                    if d_str not in cur_path.split(os.pathsep):
                        os.environ["PATH"] = f"{d_str}{os.pathsep}{cur_path}"
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(d_str)
                        except OSError:
                            pass
            except Exception:
                pass
            _DLL_INITIALIZED = True

    return ffmpeg_bin


def _patch_openai_whisper_for_turing_fp32() -> None:
    """Patch openai-whisper so weights can reside in FP16 in VRAM (~1.76 GB for turbo) while
    all Linear, Conv1d, Attention, and Decoder logit operations execute in safe FP32 without NaNs.
    """
    global _WHISPER_PATCHED
    if _WHISPER_PATCHED:
        return
    import torch
    import torch.nn.functional as F
    import whisper.model

    def _safe_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(
            x.float(),
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
        )
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    def _safe_conv1d_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv1d(
            x.float(),
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    orig_decoder_forward = whisper.model.TextDecoder.forward

    def _safe_decoder_forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        logits = orig_decoder_forward(self, x, xa.float(), kv_cache=kv_cache)
        return torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)

    whisper.model.Linear.forward = _safe_linear_forward
    whisper.model.Conv1d.forward = _safe_conv1d_forward
    whisper.model.TextDecoder.forward = _safe_decoder_forward
    whisper.model.MultiHeadAttention.use_sdpa = False
    _WHISPER_PATCHED = True


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
    """Shared SHA-1 cache for separated vocal stems and acoustic alignment anchors."""

    def __init__(self, root: Path) -> None:
        self.base = Path(root).resolve() / ".lyrics_cache"
        self.vocals_dir = self.base / "vocals"
        self.anchors_dir = self.base / "anchors"
        self.vocals_dir.mkdir(parents=True, exist_ok=True)
        self.anchors_dir.mkdir(parents=True, exist_ok=True)

    def vocal_path(self, audio_sha1: str) -> Path:
        return self.vocals_dir / f"{audio_sha1}.wav"

    def has_vocal(self, audio_sha1: str) -> bool:
        p = self.vocal_path(audio_sha1)
        return p.is_file() and p.stat().st_size > 1024

    def anchor_path(
        self,
        audio_sha1: str,
        lang_key: str,
        model_id: str | None = None,
        vocal_isolated: bool | None = None,
    ) -> Path:
        safe_lang = re.sub(r"[^a-z0-9_-]", "", (lang_key or "auto").lower()) or "auto"
        if not model_id:
            # Legacy path retained for callers that do not select an ASR model.
            return self.anchors_dir / f"{audio_sha1}_{safe_lang}.json"
        safe_model = re.sub(r"[^a-z0-9_-]", "", model_id.lower()) or "default"
        source = "vocals" if vocal_isolated else "mix" if vocal_isolated is False else "audio"
        return self.anchors_dir / f"{audio_sha1}_{safe_lang}_{safe_model}_{source}.json"

    def load_anchors(
        self,
        audio_sha1: str,
        lang_key: str,
        model_id: str | None = None,
        vocal_isolated: bool | None = None,
    ) -> dict | None:
        if model_id:
            candidates = (self.anchor_path(audio_sha1, lang_key, model_id, vocal_isolated),)
        else:
            candidates = (self.anchor_path(audio_sha1, lang_key), self.anchor_path(audio_sha1, "auto"))
        for candidate in candidates:
            if candidate.is_file():
                try:
                    payload = read_json(candidate)
                    # A cache generated by another recognizer (or another vocal source)
                    # must never silently become the timing source for this run.
                    if model_id and payload.get("model") != model_id:
                        continue
                    if vocal_isolated is not None and payload.get("vocal_isolated") != vocal_isolated:
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
    ) -> None:
        try:
            stored = dict(payload)
            if model_id:
                stored.setdefault("model", model_id)
            if vocal_isolated is not None:
                stored.setdefault("vocal_isolated", vocal_isolated)
            write_json(self.anchor_path(audio_sha1, lang_key, model_id, vocal_isolated), stored)
        except Exception:
            pass


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
    """Return `(asr_16k_wav_path, vocal_isolated, used_cache)`."""
    cached_wav = cache.vocal_path(audio_sha1)
    if want_isolate and not force_separate and cache.has_vocal(audio_sha1):
        log("Đã tìm thấy kênh giọng hát đã tách trong bộ nhớ đệm — tái sử dụng ngay.")
        progress(0.38, "separating_vocals")
        return cached_wav, True, True

    tsumugi = getattr(engine, "tsumugi", None)
    if want_isolate and tsumugi is not None and getattr(tsumugi, "python", None) and Path(tsumugi.python).is_file():
        progress(0.14, "separating_vocals")
        log("Đang chuẩn hóa âm thanh stereo 44.1kHz để tách giọng hát AI (BS-RoFormer)...")
        stereo_44k = tmp_dir / "input_44k_stereo.wav"
        convert_audio_ffmpeg(source_audio, stereo_44k, sample_rate=44100, channels=2, vocal_bandpass=False)
        if should_stop():
            raise Cancelled()

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        env.pop("VIRTUAL_ENV", None)
        device_str = str(getattr(engine, "device", "auto"))
        cmd = [
            str(tsumugi.python),
            "-X",
            "utf8",
            str(VOCAL_WORKER),
            "--tsumugi-dir",
            str(tsumugi.dir),
            "--audio",
            str(stereo_44k),
            "--out",
            str(cached_wav),
            "--device",
            device_str,
        ]
        if getattr(tsumugi, "ffmpeg_dir", None):
            cmd += ["--ffmpeg-dir", str(tsumugi.ffmpeg_dir)]

        log("Đang tách riêng giọng hát khỏi nhạc đệm (BS-RoFormer)...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(tsumugi.dir),
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
            if not line:
                continue
            if line.startswith("@@progress "):
                try:
                    info = json.loads(line[len("@@progress ") :])
                    done = int(info.get("done", 0))
                    total = max(1, int(info.get("total", 1)))
                    frac = 0.15 + 0.26 * (done / total)
                    progress(round(frac, 3), "separating_vocals")
                except Exception:
                    pass
            elif not line.startswith("@@"):
                log(line[-150:])

        code = proc.wait()
        stopped.set()
        if should_stop():
            raise Cancelled()
        if code == 0 and cache.has_vocal(audio_sha1):
            progress(0.42, "separating_vocals")
            return cached_wav, True, False
        log("Không thể tách kênh vocals bằng BS-RoFormer, chuyển sang bộ lọc dải tần giọng hát FFmpeg.")

    progress(0.30, "separating_vocals")
    filtered_16k = tmp_dir / "filtered_16k.wav"
    convert_audio_ffmpeg(source_audio, filtered_16k, sample_rate=16000, channels=1, vocal_bandpass=True)
    return filtered_16k, False, False


# ================================================================== Vocal Energy & Instrumental Gate
def compute_vocal_envelope(audio_16k: np.ndarray, hop_samples: int = 800) -> tuple[np.ndarray, float]:
    """Compute 50ms RMS envelope and adaptive vocal floor threshold."""
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
    """Return `(mean_rms, active_frame_ratio)` for `[start_s, end_s]` (50ms frames -> 20 fps)."""
    if rms_env.size == 0 or end_s <= start_s:
        return 0.0, 0.0
    i0 = max(0, min(rms_env.size - 1, int(math.floor(start_s * 20.0))))
    i1 = max(i0 + 1, min(rms_env.size, int(math.ceil(end_s * 20.0))))
    sl = rms_env[i0:i1]
    mean_r = float(np.mean(sl)) if sl.size else 0.0
    active_ratio = float(np.mean(sl >= floor)) if sl.size else 0.0
    return mean_r, active_ratio


def select_vocal_windows(audio_16k: np.ndarray, sr: int = 16000, window_sec: int = 30, max_windows: int = 5) -> list[np.ndarray]:
    """Pick up to `max_windows` non-overlapping windows with highest vocal RMS (skipping instrumental intros)."""
    win_samples = window_sec * sr
    if audio_16k.shape[0] <= win_samples:
        return [audio_16k]
    hop = 10 * sr
    scored: list[tuple[float, int]] = []
    for start in range(0, max(1, audio_16k.shape[0] - win_samples // 2), hop):
        chunk = audio_16k[start : start + win_samples]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
        scored.append((rms, start))
    scored.sort(reverse=True)
    chosen: list[int] = []
    for rms, start in scored:
        if rms < 1e-4:
            break
        if all(abs(start - prev) >= 16 * sr for prev in chosen):
            chosen.append(start)
            if len(chosen) >= max_windows:
                break
    if not chosen:
        chosen = [0]
    chosen.sort()
    return [audio_16k[s : s + win_samples] for s in chosen]


def _level_vocal_dynamics(audio_16k: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply gentle singing-voice dynamic range leveling so quiet verses and loud choruses
    both sit at optimal Whisper mel-spectrogram energy without boosting silent noise floor.
    """
    n = audio_16k.shape[0]
    if n < sr // 2:
        return audio_16k
    hop = sr // 10  # 100ms
    win = sr // 2   # 500ms smoothing window
    n_frames = max(1, math.ceil(n / hop))
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        c = audio_16k[s : min(n, s + win)]
        if c.size:
            env[i] = float(np.sqrt(np.mean(np.square(c, dtype=np.float64)) + 1e-12))

    # Smooth the envelope across ~700ms so gain changes never pump mid-syllable
    kernel = np.ones(7, dtype=np.float32) / 7.0
    smooth_env = np.convolve(env, kernel, mode="same")

    # Target vocal RMS ~0.12; ignore silent frames below noise gate 0.004
    target_rms = 0.12
    gains = np.ones_like(smooth_env)
    active = smooth_env >= 0.0045
    if np.any(active):
        raw_gain = target_rms / np.maximum(smooth_env, 1e-4)
        # Boost quiet verses up to 3.2x (+10 dB), gently attenuate very hot choruses down to 0.65x
        gains[active] = np.clip(raw_gain[active], 0.65, 3.2)

    # Interpolate frame gains to sample level
    x_frames = np.linspace(0.0, float(n - 1), n_frames, dtype=np.float32)
    sample_gains = np.interp(np.arange(n, dtype=np.float32), x_frames, gains).astype(np.float32)
    leveled = audio_16k * sample_gains
    # Soft-knee tanh limiter to prevent hard clipping on loud belting peaks
    peak = float(np.max(np.abs(leveled), initial=0.0))
    if peak > 0.92:
        leveled = (np.tanh(leveled * 1.15) / math.tanh(1.15)).astype(np.float32)
    return np.clip(leveled, -1.0, 1.0).astype(np.float32)


def _extract_vocal_regions(
    rms_env: np.ndarray,
    floor: float,
    duration: float,
    *,
    pad_sec: float = 0.65,
    merge_gap_sec: float = 2.6,
    min_region_sec: float = 0.6,
) -> list[tuple[float, float]]:
    """Extract contiguous active singing regions `(start_s, end_s)` from the 20fps vocal RMS envelope.
    Skips long instrumental intros/solos/outros so Whisper aligns its 30s windows with actual vocals.
    """
    if rms_env.size == 0 or duration <= 0:
        return [(0.0, max(0.1, duration))]

    active = rms_env >= floor
    if not np.any(active):
        return []

    raw_spans: list[tuple[float, float]] = []
    in_seg = False
    seg_start = 0
    for idx, val in enumerate(active):
        if val and not in_seg:
            in_seg = True
            seg_start = idx
        elif not val and in_seg:
            in_seg = False
            s_t = max(0.0, seg_start * 0.05 - pad_sec)
            e_t = min(duration, idx * 0.05 + pad_sec)
            if e_t - s_t >= min_region_sec:
                raw_spans.append((s_t, e_t))
    if in_seg:
        s_t = max(0.0, seg_start * 0.05 - pad_sec)
        e_t = duration
        if e_t - s_t >= min_region_sec:
            raw_spans.append((s_t, e_t))

    if not raw_spans:
        return []

    merged: list[tuple[float, float]] = [raw_spans[0]]
    for s_t, e_t in raw_spans[1:]:
        prev_s, prev_e = merged[-1]
        if s_t - prev_e <= merge_gap_sec:
            merged[-1] = (prev_s, max(prev_e, e_t))
        else:
            merged.append((s_t, e_t))

    return _split_long_regions_at_pauses(merged, rms_env, duration)


def _split_long_regions_at_pauses(
    regions: list[tuple[float, float]],
    rms_env: np.ndarray,
    duration: float,
    *,
    max_chunk_sec: float = 23.0,
    min_chunk_sec: float = 13.0,
    overlap_sec: float = 0.28,
) -> list[tuple[float, float]]:
    """Subdivide any vocal region longer than `max_chunk_sec` into 13-23s phrase-aligned windows
    split at local RMS minima (natural breath pauses between sung lines).
    Guarantees every chunk fits cleanly inside a single 30s Whisper window so Whisper's internal
    30-second seek loop never truncates or skips quiet lines near window boundaries.
    """
    if not regions:
        return []
    out: list[tuple[float, float]] = []
    n_frames = rms_env.size
    for reg_s, reg_e in regions:
        cur_s = max(0.0, float(reg_s))
        reg_end = min(duration, float(reg_e))
        while reg_end - cur_s > max_chunk_sec:
            search_lo = cur_s + min_chunk_sec
            search_hi = min(reg_end - 3.5, cur_s + max_chunk_sec)
            if search_hi <= search_lo or n_frames == 0:
                cut_t = min(reg_end, cur_s + 20.0)
            else:
                i0 = max(0, min(n_frames - 1, int(round(search_lo * 20.0))))
                i1 = max(i0 + 1, min(n_frames, int(round(search_hi * 20.0))))
                window_rms = rms_env[i0:i1]
                if window_rms.size >= 5:
                    smooth = np.convolve(window_rms, np.ones(5, dtype=np.float32) / 5.0, mode="same")
                    best_idx = i0 + int(np.argmin(smooth))
                else:
                    best_idx = i0 + int(np.argmin(window_rms)) if window_rms.size else i0
                cut_t = round(best_idx * 0.05, 2)
                if cut_t <= cur_s + 4.0:
                    cut_t = min(reg_end, cur_s + 20.0)
            out.append((round(cur_s, 2), round(min(reg_end, cut_t + overlap_sec), 2)))
            cur_s = max(cur_s + 4.0, cut_t - overlap_sec * 0.5)
        if reg_end - cur_s >= 0.4:
            out.append((round(cur_s, 2), round(reg_end, 2)))
    return out


def _split_segment_into_lyric_lines(
    text: str,
    s_time: float,
    e_time: float,
    valid_words: list[dict],
) -> list[tuple[str, float, float, list[dict]]]:
    """Split multi-clause Whisper segments (in English, Vietnamese, Japanese, Chinese, Korean, etc.)
    at natural clause boundaries (punctuation, full-width spaces, or breath pauses) so each sung
    phrase becomes its own synchronized lyric line.
    """
    n_words = len(valid_words)
    seg_dur = e_time - s_time
    is_cjk = bool(CJK_CHAR.search(text))
    min_tokens = 3 if is_cjk else 3
    if n_words < (5 if is_cjk else 7) or seg_dur < (4.2 if is_cjk else 5.0):
        return [(text, s_time, e_time, valid_words)]

    split_points: list[int] = []
    last_split = 0
    for i in range(n_words - 1):
        left_count = (i + 1) - last_split
        right_count = n_words - (i + 1)
        if left_count < min_tokens or right_count < min_tokens:
            continue
        w_cur = valid_words[i]
        w_nxt = valid_words[i + 1]
        raw_cur_word = str(w_cur.get("word") or "")
        w_txt = raw_cur_word.strip()
        gap = max(0.0, float(w_nxt["start"]) - float(w_cur["end"]))
        left_dur = float(w_cur["end"]) - float(valid_words[last_split]["start"])
        has_clause_punct = w_txt.endswith((",", ";", ".", "!", "?", "、", "。", "！", "？", "…")) or (
            is_cjk and raw_cur_word.endswith((" ", "　"))
        )

        should_split = False
        if has_clause_punct and (gap >= 0.08 or left_dur >= 2.0 or left_count >= (4 if is_cjk else 5)):
            should_split = True
        elif gap >= (0.30 if is_cjk else 0.36) and left_dur >= (1.9 if is_cjk else 2.2) and left_count >= (3 if is_cjk else 4):
            should_split = True

        if should_split:
            split_points.append(i + 1)
            last_split = i + 1

    if not split_points:
        return [(text, s_time, e_time, valid_words)]

    bounds = [0] + split_points + [n_words]
    sub_lines: list[tuple[str, float, float, list[dict]]] = []
    for b_idx in range(len(bounds) - 1):
        i0, i1 = bounds[b_idx], bounds[b_idx + 1]
        sub_w = valid_words[i0:i1]
        if not sub_w:
            continue
        sub_txt = "".join(str(w.get("word") or "") for w in sub_w).strip()
        sub_txt = re.sub(r"[,;、]\s*$", "", sub_txt).strip()
        if not sub_txt:
            continue
        if len(sub_txt) >= 2 and sub_txt[0].islower() and sub_txt[0].isalpha():
            sub_txt = sub_txt[0].upper() + sub_txt[1:]
        sub_s = s_time if b_idx == 0 else float(sub_w[0]["start"])
        sub_e = e_time if b_idx == len(bounds) - 2 else float(sub_w[-1]["end"])
        sub_e = max(sub_s + 0.15, sub_e)
        sub_lines.append((sub_txt, round(sub_s, 2), round(sub_e, 2), sub_w))

    return sub_lines if sub_lines else [(text, s_time, e_time, valid_words)]


def _validate_words(raw_words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Validate DTW word timestamps. Return `[]` if word-level timing is unreliable (never fabricate)."""
    if not raw_words:
        return []
    cleaned: list[dict] = []
    prev_end = seg_start - 0.05
    zero_span_count = 0
    for w in raw_words:
        txt = str(w.get("word") or "")
        if not txt.strip():
            continue
        ws = w.get("start")
        we = w.get("end")
        prob = w.get("probability")
        if ws is None or we is None or not (math.isfinite(ws) and math.isfinite(we)):
            return []
        ws_f = max(seg_start, float(ws))
        we_f = max(ws_f, min(seg_end + 0.25, float(we)))
        if ws_f < prev_end - 0.15:
            return []
        if we_f - ws_f < 0.015:
            zero_span_count += 1
        prev_end = ws_f
        p_val = round(float(prob), 3) if prob is not None and math.isfinite(prob) else None
        cleaned.append(
            {
                "word": txt,
                "start": round(ws_f, 2),
                "end": round(max(ws_f + 0.02, we_f), 2),
                "probability": p_val,
                # Calibrated threshold for sung vocals (0.30 avoids false-flagging normal melismatic words)
                "uncertain": bool(p_val is not None and p_val < 0.30),
            }
        )

    if not cleaned:
        return []
    if len(cleaned) >= 3 and zero_span_count / len(cleaned) > 0.55:
        return []
    return cleaned


# ================================================================== Speech Recognition Engines
def _run_openai_whisper_safe_cuda(
    asr_audio: np.ndarray,
    duration: float,
    vocal_isolated: bool,
    rms_env: np.ndarray,
    vocal_floor: float,
    requested_lang: str,
    model_id: str,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Run OpenAI Whisper on CUDA with FP16 weight storage (~1.76 GB VRAM for turbo) and FP32 compute.
    Completely prevents Turing sm_75 (GTX 1650) FP16 overflow / Categorical(logits: nan) errors!
    """
    import torch
    import whisper

    _patch_openai_whisper_for_turing_fp32()

    eff_model = model_id
    progress(0.45, "recognizing_lyrics")
    log(f"Đang nạp mô hình Whisper ({eff_model}) trên CUDA (chế độ FP16-Storage / FP32-Compute an toàn cho GTX 1650)...")

    w_model = whisper.load_model(eff_model, device="cpu")
    for m in w_model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            m.half()
    w_model = w_model.to("cuda")

    try:
        if should_stop():
            raise Cancelled()

        # Language detection across vocal windows in safe FP32
        progress(0.50, "recognizing_lyrics")
        if requested_lang and requested_lang != "auto":
            primary_lang = requested_lang
            lang_conf = 1.0
            detected_languages = [{"code": primary_lang, "name": language_label(primary_lang), "probability": 1.0}]
        elif w_model.is_multilingual:
            log("Đang tự động nhận diện ngôn ngữ bài hát trên các phân đoạn giọng hát...")
            windows = select_vocal_windows(asr_audio)
            accum: dict[str, float] = {}
            total_w = 0.0
            n_mels = w_model.dims.n_mels
            for chunk in windows:
                if should_stop():
                    raise Cancelled()
                w_rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
                weight = max(0.05, w_rms)
                padded = whisper.pad_or_trim(chunk)
                mel = whisper.log_mel_spectrogram(padded, n_mels=n_mels).to(device="cuda", dtype=torch.float32)
                _, probs = w_model.detect_language(mel)
                for code, prob in probs.items():
                    if math.isfinite(prob):
                        accum[code] = accum.get(code, 0.0) + float(prob) * weight
                total_w += weight

            ranked = sorted(
                ((code, val / max(total_w, 1e-9)) for code, val in accum.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            primary_lang, lang_conf = ranked[0] if ranked else ("en", 0.0)
            detected_languages = [
                {"code": c, "name": language_label(c), "probability": round(float(p), 4)}
                for c, p in ranked[:4]
                if p >= 0.03
            ]
        else:
            primary_lang = "en"
            lang_conf = 1.0
            detected_languages = [{"code": "en", "name": language_label("en"), "probability": 1.0}]

        is_mixed = (
            requested_lang in ("", "auto")
            and len(detected_languages) >= 2
            and detected_languages[1]["probability"] >= 0.18
        )
        decode_lang = None if is_mixed else primary_lang
        log(
            f"Ngôn ngữ nhận diện: {language_label(primary_lang)} ({lang_conf * 100:.1f}%)"
            + (" · Đa ngôn ngữ" if is_mixed else "")
        )

        if should_stop():
            raise Cancelled()

        progress(0.56, "recognizing_lyrics")
        log("Đang trích xuất lời hát & căn dấu thời gian từng câu/từ (Beam Search x5 + Cân bằng giọng hát)...")

        # Always chunk into <=23s phrase-aligned windows at local RMS minima (both when vocal_isolated=True
        # and vocal_isolated=False) so Whisper's internal 30s seek loop never drops quiet lines near 30s boundaries!
        regions = (
            _extract_vocal_regions(rms_env, vocal_floor, duration)
            if vocal_isolated
            else _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)
        )
        if not regions:
            regions = _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)

        raw_segments: list[dict] = []
        anchors: list[dict] = []
        repeat_streak = 0
        last_norm_text = ""

        total_reg_dur = max(0.1, sum(max(0.1, e - s) for s, e in regions))
        done_reg_dur = 0.0

        for reg_start, reg_end in regions:
            if should_stop():
                raise Cancelled()
            s_idx = max(0, int(reg_start * 16000))
            e_idx = min(asr_audio.shape[0], int(reg_end * 16000))
            if e_idx - s_idx < 3200:  # < 0.2s
                continue
            chunk_audio = asr_audio[s_idx:e_idx]

            # Run transcribe with fp16=False so mel spectrograms & activations stay in FP32
            # while weights stay in FP16 via our patched Linear/Conv1d!
            res = w_model.transcribe(
                chunk_audio,
                language=decode_lang,
                task="transcribe",
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=(0.0, 0.2),
                beam_size=5,
                best_of=5,
                patience=1.2,
                compression_ratio_threshold=2.3,
                logprob_threshold=-1.0,
                no_speech_threshold=0.78 if vocal_isolated else 0.85,
                fp16=False,
                verbose=False,
            )

            done_reg_dur += max(0.1, reg_end - reg_start)
            progress(round(0.56 + 0.30 * min(1.0, done_reg_dur / total_reg_dur), 3), "recognizing_lyrics")

            for seg in res.get("segments") or []:
                if should_stop():
                    raise Cancelled()
                text = str(seg.get("text") or "").strip()
                if not text or NOISE_ONLY_PATTERN.match(text):
                    continue

                s_time = max(0.0, reg_start + float(seg.get("start") or 0.0))
                e_time = max(s_time + 0.08, min(duration + 0.5, reg_start + float(seg.get("end") or 0.2)))
                mean_r, active_ratio = segment_vocal_activity(rms_env, vocal_floor, s_time, e_time)
                no_speech = float(seg.get("no_speech_prob") or 0.0)
                avg_lp = float(seg.get("avg_logprob") or -0.5)
                comp_ratio = float(seg.get("compression_ratio") or 1.0)

                if vocal_isolated and active_ratio < 0.08 and mean_r < vocal_floor * 0.9:
                    continue
                if not vocal_isolated and mean_r < vocal_floor * 0.65 and no_speech > 0.40:
                    continue
                if HALLUCINATION_PATTERNS.match(text):
                    continue
                if comp_ratio > 2.65 and avg_lp < -0.75:
                    continue

                raw_w_list = [
                    {
                        "word": str(w.get("word") or ""),
                        "start": reg_start + float(w.get("start") or 0.0),
                        "end": reg_start + float(w.get("end") or 0.2),
                        "probability": float(w.get("probability") or 0.0),
                    }
                    for w in (seg.get("words") or [])
                ]
                valid_words = _validate_words(raw_w_list, s_time, e_time)
                sub_lines = _split_segment_into_lyric_lines(text, s_time, e_time, valid_words)

                for sub_text, sub_s, sub_e, sub_words in sub_lines:
                    if not sub_text or NOISE_ONLY_PATTERN.match(sub_text) or HALLUCINATION_PATTERNS.match(sub_text):
                        continue
                    sub_mean_r, sub_active_ratio = segment_vocal_activity(rms_env, vocal_floor, sub_s, sub_e)
                    if vocal_isolated and sub_active_ratio < 0.08 and sub_mean_r < vocal_floor * 0.9:
                        continue
                    norm_t = re.sub(r"[^\w\s]", "", sub_text.lower()).strip()
                    norm_t = re.sub(r"\s+", " ", norm_t)
                    if not norm_t:
                        continue

                    # Boundary deduplication across adjacent sub-23s chunks (0.28s overlap)
                    if raw_segments:
                        prev_seg = raw_segments[-1]
                        prev_norm = re.sub(r"[^\w\s]", "", prev_seg["text"].lower()).strip()
                        prev_norm = re.sub(r"\s+", " ", prev_norm)
                        if sub_s < prev_seg["end"] + 0.35 and (
                            norm_t == prev_norm
                            or (len(norm_t) >= 4 and len(prev_norm) >= 10 and norm_t in prev_norm)
                        ):
                            continue
                        if sub_s < prev_seg["end"] - 0.04:
                            sub_s = round(max(prev_seg["start"] + 0.15, prev_seg["end"]), 2)
                            sub_e = round(max(sub_s + 0.12, sub_e), 2)

                    if norm_t == last_norm_text and (sub_e - sub_s) < 1.2 and (
                        raw_segments and sub_s - raw_segments[-1]["end"] < 0.25
                    ):
                        repeat_streak += 1
                        if repeat_streak >= 2:
                            continue
                    else:
                        repeat_streak = 0
                    last_norm_text = norm_t

                    for vw in sub_words:
                        anchors.append(vw)

                    if sub_words:
                        probs_w = [w["probability"] for w in sub_words if w["probability"] is not None]
                        w_conf = float(np.mean(probs_w)) if probs_w else math.exp(max(-3.0, min(0.0, avg_lp)))
                    else:
                        w_conf = math.exp(max(-3.0, min(0.0, avg_lp)))

                    # Calibrated singing confidence & review flagging (avoid false-flagging clear lines)
                    conf = round(max(0.05, min(0.99, 0.72 * w_conf + 0.28 * (1.0 - min(1.0, no_speech)))), 3)
                    uncertain_words = sum(1 for w in sub_words if w.get("uncertain"))
                    uncertain_ratio = (uncertain_words / len(sub_words)) if sub_words else 0.0
                    low_conf = bool(
                        conf < 0.56
                        or avg_lp < -0.82
                        or (uncertain_words >= 2 and uncertain_ratio >= 0.28)
                    )

                    raw_segments.append(
                        {
                            "start": round(sub_s, 2),
                            "end": round(sub_e, 2),
                            "text": sub_text,
                            "lang": infer_segment_script_lang(sub_text, primary_lang),
                            "confidence": conf,
                            "low_confidence": low_conf,
                            "words": sub_words,
                        }
                    )

        return {
            "language": primary_lang,
            "language_name": language_label(primary_lang),
            "language_confidence": round(float(lang_conf), 4),
            "detected_languages": detected_languages,
            "raw_segments": raw_segments,
            "anchors": anchors,
            "rms_env": rms_env,
            "vocal_floor": vocal_floor,
        }
    finally:
        del w_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_faster_whisper_cpu_fallback(
    asr_audio: np.ndarray,
    duration: float,
    vocal_isolated: bool,
    rms_env: np.ndarray,
    vocal_floor: float,
    requested_lang: str,
    model_id: str,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
    *,
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict[str, Any]:
    """Run Faster-Whisper (CTranslate2) on CUDA (int8_float32) or CPU (int8) honoring the exact requested model."""
    from faster_whisper import WhisperModel

    eff_model = model_id
    progress(0.46, "recognizing_lyrics")
    if device == "cuda":
        log(f"Đang nạp Faster-Whisper ({eff_model}, CUDA {compute_type} — giữ nguyên mô hình đã chọn, tối ưu 4GB VRAM)...")
        fw_model = WhisperModel(eff_model, device="cuda", compute_type=compute_type)
    else:
        log(f"Đang nạp Faster-Whisper ({eff_model}, CPU INT8)...")
        fw_model = WhisperModel(eff_model, device="cpu", compute_type="int8", cpu_threads=max(4, (os.cpu_count() or 8) // 2))
    try:
        if requested_lang and requested_lang != "auto":
            primary_lang = requested_lang
            lang_conf = 1.0
            detected_languages = [{"code": primary_lang, "name": language_label(primary_lang), "probability": 1.0}]
        else:
            progress(0.50, "recognizing_lyrics")
            windows = select_vocal_windows(asr_audio)
            accum: dict[str, float] = {}
            total_w = 0.0
            for chunk in windows:
                if should_stop():
                    raise Cancelled()
                w_rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
                weight = max(0.05, w_rms)
                lang_code, lang_prob, all_probs = fw_model.detect_language(chunk)
                if all_probs:
                    for code_raw, prob in all_probs:
                        code = str(code_raw).strip("<|>")
                        accum[code] = accum.get(code, 0.0) + float(prob) * weight
                elif lang_code:
                    code = str(lang_code).strip("<|>")
                    accum[code] = accum.get(code, 0.0) + float(lang_prob) * weight
                total_w += weight

            ranked = sorted(
                ((code, val / max(total_w, 1e-9)) for code, val in accum.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            primary_lang, lang_conf = ranked[0] if ranked else ("en", 0.0)
            detected_languages = [
                {"code": c, "name": language_label(c), "probability": round(float(p), 4)}
                for c, p in ranked[:4]
                if p >= 0.03
            ]

        is_mixed = (
            requested_lang in ("", "auto")
            and len(detected_languages) >= 2
            and detected_languages[1]["probability"] >= 0.18
        )
        decode_lang = None if is_mixed else primary_lang

        progress(0.55, "recognizing_lyrics")
        regions = (
            _extract_vocal_regions(rms_env, vocal_floor, duration)
            if vocal_isolated
            else _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)
        )
        if not regions:
            regions = _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)

        raw_segments: list[dict] = []
        anchors: list[dict] = []
        repeat_streak = 0
        last_norm_text = ""
        total_reg_dur = max(0.1, sum(max(0.1, e - s) for s, e in regions))
        done_reg_dur = 0.0

        for reg_start, reg_end in regions:
            if should_stop():
                raise Cancelled()
            s_idx = max(0, int(reg_start * 16000))
            e_idx = min(asr_audio.shape[0], int(reg_end * 16000))
            if e_idx - s_idx < 3200:
                continue
            chunk_audio = asr_audio[s_idx:e_idx]
            seg_iter, _ = fw_model.transcribe(
                chunk_audio,
                language=decode_lang,
                task="transcribe",
                beam_size=5 if device == "cuda" else 4,
                patience=1.2,
                temperature=[0.0, 0.2],
                compression_ratio_threshold=2.3,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.82,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=False,
            )

            for seg in seg_iter:
                if should_stop():
                    raise Cancelled()
                s_time = max(0.0, reg_start + float(seg.start))
                e_time = max(s_time + 0.08, min(duration + 0.5, reg_start + float(seg.end)))
                text = (seg.text or "").strip()
                if not text or NOISE_ONLY_PATTERN.match(text):
                    continue

                mean_r, active_ratio = segment_vocal_activity(rms_env, vocal_floor, s_time, e_time)
                no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                avg_lp = float(getattr(seg, "avg_logprob", -0.5) or -0.5)
                if vocal_isolated and active_ratio < 0.08 and mean_r < vocal_floor * 0.9:
                    continue
                if not vocal_isolated and mean_r < vocal_floor * 0.65 and no_speech > 0.40:
                    continue
                if HALLUCINATION_PATTERNS.match(text):
                    continue

                raw_w_list = [
                    {
                        "word": w.word,
                        "start": reg_start + float(w.start),
                        "end": reg_start + float(w.end),
                        "probability": float(w.probability),
                    }
                    for w in (seg.words or [])
                ]
                valid_words = _validate_words(raw_w_list, s_time, e_time)
                sub_lines = _split_segment_into_lyric_lines(text, s_time, e_time, valid_words)

                for sub_text, sub_s, sub_e, sub_words in sub_lines:
                    if not sub_text or NOISE_ONLY_PATTERN.match(sub_text) or HALLUCINATION_PATTERNS.match(sub_text):
                        continue
                    sub_mean_r, sub_active_ratio = segment_vocal_activity(rms_env, vocal_floor, sub_s, sub_e)
                    if vocal_isolated and sub_active_ratio < 0.08 and sub_mean_r < vocal_floor * 0.9:
                        continue
                    norm_t = re.sub(r"[^\w\s]", "", sub_text.lower()).strip()
                    norm_t = re.sub(r"\s+", " ", norm_t)
                    if not norm_t:
                        continue
                    if raw_segments:
                        prev_seg = raw_segments[-1]
                        prev_norm = re.sub(r"[^\w\s]", "", prev_seg["text"].lower()).strip()
                        prev_norm = re.sub(r"\s+", " ", prev_norm)
                        if sub_s < prev_seg["end"] + 0.35 and (
                            norm_t == prev_norm
                            or (len(norm_t) >= 4 and len(prev_norm) >= 10 and norm_t in prev_norm)
                        ):
                            continue
                        if sub_s < prev_seg["end"] - 0.04:
                            sub_s = round(max(prev_seg["start"] + 0.15, prev_seg["end"]), 2)
                            sub_e = round(max(sub_s + 0.12, sub_e), 2)

                    if norm_t == last_norm_text and (sub_e - sub_s) < 1.2 and (
                        raw_segments and sub_s - raw_segments[-1]["end"] < 0.25
                    ):
                        repeat_streak += 1
                        if repeat_streak >= 2:
                            continue
                    else:
                        repeat_streak = 0
                    last_norm_text = norm_t

                    for vw in sub_words:
                        anchors.append(vw)

                    probs_w = [w["probability"] for w in sub_words if w["probability"] is not None]
                    w_conf = float(np.mean(probs_w)) if probs_w else math.exp(max(-3.0, min(0.0, avg_lp)))
                    conf = round(max(0.05, min(0.99, 0.72 * w_conf + 0.28 * (1.0 - min(1.0, no_speech)))), 3)
                    uncertain_words = sum(1 for w in sub_words if w.get("uncertain"))
                    uncertain_ratio = (uncertain_words / len(sub_words)) if sub_words else 0.0
                    low_conf = bool(
                        conf < 0.56
                        or avg_lp < -0.82
                        or (uncertain_words >= 2 and uncertain_ratio >= 0.28)
                    )
                    raw_segments.append(
                        {
                            "start": round(sub_s, 2),
                            "end": round(sub_e, 2),
                            "text": sub_text,
                            "lang": infer_segment_script_lang(sub_text, primary_lang),
                            "confidence": conf,
                            "low_confidence": low_conf,
                            "words": sub_words,
                        }
                    )

            done_reg_dur += max(0.1, reg_end - reg_start)
            progress(round(0.55 + 0.30 * min(1.0, done_reg_dur / total_reg_dur), 3), "recognizing_lyrics")

        return {
            "language": primary_lang,
            "language_name": language_label(primary_lang),
            "language_confidence": round(float(lang_conf), 4),
            "detected_languages": detected_languages,
            "raw_segments": raw_segments,
            "anchors": anchors,
            "rms_env": rms_env,
            "vocal_floor": vocal_floor,
        }
    finally:
        del fw_model
        gc.collect()
        if device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


def run_asr_recognition(
    engine: Engine,
    audio_16k: np.ndarray,
    duration: float,
    vocal_isolated: bool,
    requested_lang: str,
    model_id: str,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
    *,
    rms_env: np.ndarray | None = None,
    vocal_floor: float | None = None,
) -> dict[str, Any]:
    """Run speech recognition on isolated vocals with GTX 1650 NaN protection and CPU fallback."""
    if rms_env is None or vocal_floor is None:
        rms_env, vocal_floor = compute_vocal_envelope(audio_16k)
    active_frames = int(np.sum(rms_env >= vocal_floor))
    active_seconds = active_frames * 0.05

    if vocal_isolated and active_seconds < 0.75 and float(np.max(rms_env, initial=0.0)) < 0.0035:
        log("Không phát hiện tín hiệu giọng hát trên kênh vocals (bản nhạc hòa tấu không lời).")
        lang_code = requested_lang if (requested_lang and requested_lang != "auto") else "auto"
        return {
            "language": lang_code,
            "language_name": language_label(lang_code),
            "language_confidence": 0.0,
            "detected_languages": [],
            "raw_segments": [],
            "anchors": [],
            "rms_env": rms_env,
            "vocal_floor": vocal_floor,
        }

    # Apply singing-voice dynamic range leveling so quiet verses and loud choruses
    # both sit at optimal Whisper energy levels without clipping or boosting silence.
    asr_audio = _level_vocal_dynamics(audio_16k)

    import torch

    want_dev = str(getattr(engine, "device", "auto"))
    if want_dev != "cpu" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024**3)
        total_gb = total_b / (1024**3)
        # For large-v3 on 4 GB VRAM GPUs (e.g. GTX 1650), use Faster-Whisper CUDA int8_float32 (~1.6 GB VRAM)
        # so the EXACT requested large-v3 model runs natively on GPU without downgrading to turbo or OOM!
        if model_id == "large-v3" and (total_gb <= 4.5 or free_gb < 3.4) and free_gb >= 1.65:
            try:
                return _run_faster_whisper_cpu_fallback(
                    asr_audio=asr_audio,
                    duration=duration,
                    vocal_isolated=vocal_isolated,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    requested_lang=requested_lang,
                    model_id=model_id,
                    progress=progress,
                    should_stop=should_stop,
                    log=log,
                    device="cuda",
                    compute_type="int8_float32",
                )
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"Cảnh báo Faster-Whisper CUDA ({exc}) — chuyển sang Faster-Whisper CPU INT8 (giữ nguyên {model_id})...")
                return _run_faster_whisper_cpu_fallback(
                    asr_audio=asr_audio,
                    duration=duration,
                    vocal_isolated=vocal_isolated,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    requested_lang=requested_lang,
                    model_id=model_id,
                    progress=progress,
                    should_stop=should_stop,
                    log=log,
                    device="cpu",
                    compute_type="int8",
                )
        if free_gb >= 1.65:
            try:
                return _run_openai_whisper_safe_cuda(
                    asr_audio=asr_audio,
                    duration=duration,
                    vocal_isolated=vocal_isolated,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    requested_lang=requested_lang,
                    model_id=model_id,
                    progress=progress,
                    should_stop=should_stop,
                    log=log,
                )
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"Cảnh báo CUDA ({exc}) — chuyển sang Faster-Whisper CPU INT8 dự phòng...")

    return _run_faster_whisper_cpu_fallback(
        asr_audio=asr_audio,
        duration=duration,
        vocal_isolated=vocal_isolated,
        rms_env=rms_env,
        vocal_floor=vocal_floor,
        requested_lang=requested_lang,
        model_id=model_id,
        progress=progress,
        should_stop=should_stop,
        log=log,
    )


# ================================================================== Mode 2: Forced Alignment of User Lyrics
def _normalize_char_for_match(ch: str) -> str:
    """Language-aware phonetic/script character normalizer for monotonic alignment scoring ONLY."""
    code = ord(ch)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    if 0xAC00 <= code <= 0xD7A3:
        return unicodedata.normalize("NFD", ch)
    if ch in ("đ", "Đ"):
        return "d"
    decomp = unicodedata.normalize("NFD", ch.lower())
    return "".join(c for c in decomp if unicodedata.category(c) != "Mn")


def _tokenize_line_for_alignment(text: str, lang: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if CJK_CHAR.search(text):
        # Japanese and Chinese references do not have reliable whitespace word
        # boundaries. Tokenize their characters even when a mixed-script line
        # contains spaces, so they can match Whisper's acoustic token slices.
        tokens = re.findall(
            r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][ー々ゝゞ]*\s*|[A-Za-z0-9']+\s*|[^\s]", text
        )
        return [t for t in tokens if t]
    parts = re.findall(r"\S+\s*", text)
    return [p for p in parts if p.strip()]


def parse_user_lyrics(lyrics_text: str) -> list[dict[str, str | None]]:
    lines_out: list[dict[str, str | None]] = []
    current_section: str | None = None

    def add_lyric_line(raw: str, copies: int = 1) -> None:
        nonlocal current_section
        s = raw.strip().lstrip("\ufeff")
        if not s:
            return
        m = SECTION_HEADER_RE.match(s)
        if m:
            raw_tag = re.sub(r"\s+", " ", m.group(1).strip().lower())
            base_tag = re.sub(r"\s*\d+$", "", raw_tag)
            canon = SECTION_CANONICAL.get(base_tag)
            if canon:
                suffix_m = re.search(r"(\d+)$", raw_tag)
                current_section = f"{canon} {suffix_m.group(1)}" if suffix_m else canon
            else:
                current_section = m.group(1).strip().title()
            return
        lines_out.extend({"text": s, "section": current_section} for _ in range(copies))

    normalized_input = lyrics_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    srt_time = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$")
    if any(srt_time.match(line) for line in normalized_input.splitlines()):
        # SRT cue wrapping is presentational: one cue is one lyric line even if
        # the subtitle editor wrapped it across multiple physical lines.
        for block in re.split(r"\n\s*\n", normalized_input.strip()):
            content = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
                and not line.strip().isdigit()
                and not srt_time.match(line)
            ]
            if content:
                add_lyric_line(" ".join(content))
    else:
        lrc_time = re.compile(r"\[(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,:]\d{1,3})?\]")
        lrc_metadata = re.compile(r"^\[(?:ti|ar|al|by|la|length|offset|re|ve):.*\]$", re.IGNORECASE)
        for raw in normalized_input.splitlines():
            s = raw.strip()
            if not s or lrc_metadata.match(s):
                continue
            stamps = 0
            while (match := lrc_time.match(s)) is not None:
                stamps += 1
                s = s[match.end() :].lstrip()
            # Multiple LRC tags encode the same lyric at multiple occurrences.
            # Preserve those repetitions while recomputing every timestamp from
            # the selected audio.
            add_lyric_line(s, copies=stamps or 1)
    return lines_out


def align_user_lyrics_to_anchors(
    user_lines: list[dict[str, str | None]],
    anchors: list[dict],
    rms_env: np.ndarray,
    vocal_floor: float,
    duration: float,
    primary_lang: str,
) -> tuple[list[dict], str]:
    """Align reference text to acoustic anchors without changing its wording.

    Direct anchor matches retain measured word timings. A reference token that
    falls between two measured anchors may receive an explicitly uncertain
    interpolation; an entirely unanchored line stays line-level.
    """
    if not user_lines:
        return [], "line"

    # Align at the token level.  The previous implementation flattened both
    # streams into one long character string and fed it to SequenceMatcher.
    # That is attractive because it handles spelling errors, but it can jump
    # between repeated phrases (a very common pattern in choruses) and then
    # assign every following line to the wrong take.  A monotonic token DP
    # keeps the time axis intact while still allowing Whisper to omit or
    # insert words.
    user_words_meta: list[dict] = []
    line_word_meta: list[list[dict]] = [[] for _ in user_lines]
    for li, item in enumerate(user_lines):
        line_txt = str(item["text"])
        seg_lang = infer_segment_script_lang(line_txt, primary_lang)
        tokens = _tokenize_line_for_alignment(line_txt, seg_lang)
        for tok in tokens:
            norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in tok) if c.isalnum())
            word_meta = {"line_idx": li, "display": tok, "norm": norm}
            user_words_meta.append(word_meta)
            line_word_meta[li].append(word_meta)

    reference_has_cjk = any(CJK_CHAR.search(str(item.get("text") or "")) for item in user_lines)
    anchor_meta: list[dict] = []
    for a in anchors:
        w_txt = str(a.get("word") or "")
        norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in w_txt) if c.isalnum())
        if not norm:
            continue
        ws = float(a["start"])
        we = max(ws + 0.02, float(a["end"]))
        prob = float(a.get("probability") if a.get("probability") is not None else 0.7)
        prob = min(1.0, max(0.0, prob)) if math.isfinite(prob) else 0.7
        if reference_has_cjk and CJK_CHAR.search(w_txt) and len(norm) > 1:
            # Whisper often groups a spoken Japanese phrase into one token,
            # while the reference has no spaces. Give each reference
            # character its share of the token's actual acoustic interval.
            step = (we - ws) / len(norm)
            for char_idx, char in enumerate(norm):
                anchor_meta.append(
                    {
                        "norm": char,
                        "start": ws + char_idx * step,
                        "end": ws + (char_idx + 1) * step,
                        "probability": prob,
                    }
                )
        else:
            anchor_meta.append({"norm": norm, "start": ws, "end": we, "probability": prob})

    similarity_cache: dict[tuple[str, str], float] = {}
    def token_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        key = (left, right)
        cached = similarity_cache.get(key)
        if cached is not None:
            return cached
        if left == right:
            return 1.0
        # Reject impossible/very weak pairs cheaply before invoking the
        # character matcher for every cell in the token DP matrix.
        if 2 * min(len(left), len(right)) / (len(left) + len(right)) < 0.45:
            return 0.0
        # A normalized edit ratio catches the small errors Whisper makes in
        # sung vowels and Vietnamese/Japanese diacritics without treating an
        # unrelated one-letter word as a match.
        ratio = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
        if min(len(left), len(right)) <= 1:
            return 0.0
        if left.startswith(right) or right.startswith(left):
            ratio = max(ratio, 0.72)
        if len(similarity_cache) < 100_000:
            similarity_cache[key] = ratio
        return ratio

    user_count = len(user_words_meta)
    anchor_count = len(anchor_meta)
    user_to_anchor: list[int | None] = [None] * user_count
    match_quality: list[float] = [0.0] * user_count
    if user_count and anchor_count:
        # Costs are deliberately asymmetric: a missing Whisper token is less
        # surprising than an invented reference token, while a poor fuzzy
        # substitution is more expensive than skipping both tokens.  This
        # prevents common words such as "I"/"a" from pulling a whole chorus
        # onto the wrong repetition.
        user_gap = 1.05
        def anchor_gap(probability: float) -> float:
            # Low-confidence ASR insertions should be cheap to discard.  A
            # strong acoustic word is more useful when repeated text gives
            # several otherwise-equivalent alignment paths.
            return 0.16 + 0.62 * probability

        def pair_cost(sim: float, probability: float) -> float:
            confidence_penalty = 0.46 * (1.0 - probability)
            if sim >= 0.999:
                return 0.04 + confidence_penalty
            if sim >= 0.72:
                return 0.18 + (1.0 - sim) * 0.90 + confidence_penalty
            if sim >= 0.45:
                return 0.85 + (1.0 - sim) * 0.80 + confidence_penalty
            return 1.75 + confidence_penalty

        # Keep only the predecessor operation for traceback.  A byte per cell
        # is substantially smaller than retaining Python tuple objects.
        trace = [bytearray(anchor_count + 1) for _ in range(user_count + 1)]
        anchor_gaps = [anchor_gap(float(a["probability"])) for a in anchor_meta]
        prev = [0.0] * (anchor_count + 1)
        for j in range(1, anchor_count + 1):
            prev[j] = prev[j - 1] + anchor_gaps[j - 1]
            trace[0][j] = 3  # skip anchor
        for i in range(1, user_count + 1):
            cur = [0.0] * (anchor_count + 1)
            cur[0] = prev[0] + user_gap
            trace[i][0] = 2  # skip user token
            left_norm = str(user_words_meta[i - 1]["norm"])
            for j in range(1, anchor_count + 1):
                sim = token_similarity(left_norm, str(anchor_meta[j - 1]["norm"]))
                probability = float(anchor_meta[j - 1]["probability"])
                diag = prev[j - 1] + pair_cost(sim, probability)
                skip_user = prev[j] + user_gap
                skip_anchor = cur[j - 1] + anchor_gaps[j - 1]
                # Prefer a real match, then skipping an acoustic insertion,
                # then skipping a reference token when costs tie.
                best = diag
                op = 1
                if skip_anchor < best - 1e-9:
                    best, op = skip_anchor, 3
                if skip_user < best - 1e-9:
                    best, op = skip_user, 2
                cur[j] = best
                trace[i][j] = op
            prev = cur

        i, j = user_count, anchor_count
        while i > 0 or j > 0:
            op = trace[i][j] if i >= 0 and j >= 0 else 0
            if i > 0 and j > 0 and op == 1:
                sim = token_similarity(
                    str(user_words_meta[i - 1]["norm"]),
                    str(anchor_meta[j - 1]["norm"]),
                )
                if sim >= 0.45:
                    user_to_anchor[i - 1] = j - 1
                    match_quality[i - 1] = sim
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
    total_words = 0
    anchored_words = 0

    user_cursor = 0
    for wm in user_words_meta:
        li = wm["line_idx"]
        total_words += 1
        anchor_idx = user_to_anchor[user_cursor] if user_cursor < len(user_to_anchor) else None
        sim = match_quality[user_cursor] if user_cursor < len(match_quality) else 0.0
        user_cursor += 1
        if anchor_idx is not None and anchor_idx < len(anchor_meta):
            anchor = anchor_meta[anchor_idx]
            ws, we = float(anchor["start"]), float(anchor["end"])
            # Similarity is part of confidence: an exact anchor at 0.92 is
            # stronger than a fuzzy anchor at the same acoustic probability.
            mp = round(float(anchor["probability"]) * (0.55 + 0.45 * sim), 3)
            w_entry = {
                # Keep the source spacing in English token displays; enhanced
                # LRC concatenates timed tokens without injecting separators.
                "word": wm["display"],
                "start": round(ws, 2),
                "end": round(max(ws + 0.04, we), 2),
                "probability": mp,
                "uncertain": mp < 0.48 or sim < 0.72,
            }
            line_words[li].append(w_entry)
            line_matched_times[li].append((ws, we, mp))
            anchored_words += 1
        else:
            line_words[li].append(None)

    active_idxs = np.where(rms_env >= vocal_floor)[0]
    first_vocal_sec = float(active_idxs[0] * 0.05) if active_idxs.size else 0.0
    last_vocal_sec = float(min(duration, (active_idxs[-1] + 1) * 0.05)) if active_idxs.size else max(1.0, duration)

    line_bounds: list[tuple[float, float, float, bool]] = []
    for li in range(len(user_lines)):
        matches = line_matched_times[li]
        if matches:
            ls = matches[0][0]
            le = max(ls + 0.25, matches[-1][1])
            conf = round(float(np.mean([m[2] for m in matches])), 3)
            line_bounds.append((ls, le, conf, True))
        else:
            line_bounds.append((-1.0, -1.0, 0.35, False))

    n_lines = len(user_lines)
    i = 0
    while i < n_lines:
        if line_bounds[i][3]:
            i += 1
            continue
        j = i
        while j < n_lines and not line_bounds[j][3]:
            j += 1
        left_t = line_bounds[i - 1][1] + 0.15 if i > 0 else first_vocal_sec
        right_t = line_bounds[j][0] - 0.10 if j < n_lines else max(left_t + 0.5 * (j - i), last_vocal_sec)
        if right_t <= left_t:
            right_t = min(max(duration, left_t + 0.8), left_t + 1.6 * (j - i))
        span = max(0.4, right_t - left_t)
        count = j - i
        step = span / count
        for k in range(count):
            ls = left_t + k * step
            le = max(ls + 0.2, left_t + (k + 1) * step - 0.06)
            line_bounds[i + k] = (ls, le, 0.35, False)
        i = j

    # Preserve every supplied reference token in word-sync exports.  Whisper
    # occasionally omits a short sung word; when it falls between two trusted
    # anchors, distribute only that real acoustic gap across the missing words.
    # Such estimates are explicitly low confidence so consumers can distinguish
    # them from directly aligned acoustic words.
    for li, lw in enumerate(line_words):
        matched = [idx for idx, word in enumerate(lw) if word is not None]
        if not matched:
            continue
        line_meta = line_word_meta[li]

        matched_durations = [
            max(0.0, float(lw[idx]["end"]) - float(lw[idx]["start"])) / max(1, len(line_meta[idx]["norm"]))
            for idx in matched
        ]
        char_rate = float(np.median(matched_durations)) if matched_durations else 0.06
        char_rate = min(0.12, max(0.025, char_rate))
        weights = [max(1, len(meta["norm"])) for meta in line_meta]

        def add_estimated_run(indices: list[int], start: float, end: float) -> None:
            if not indices or end - start < 0.04 * len(indices):
                return
            total_weight = sum(weights[idx] for idx in indices)
            cursor = start
            cumulative_weight = 0
            for idx in indices:
                cumulative_weight += weights[idx]
                next_time = start + (end - start) * cumulative_weight / total_weight
                word_end = min(end, max(cursor + 0.02, next_time))
                lw[idx] = {
                    "word": line_meta[idx]["display"],
                    "start": round(cursor, 2),
                    "end": round(word_end, 2),
                    "probability": 0.12,
                    "uncertain": True,
                }
                cursor = word_end

        # Missing tokens between aligned neighbors use the actual unclaimed
        # gap, so their inferred spans never move the trusted words.
        for left_idx, right_idx in zip(matched, matched[1:]):
            if right_idx - left_idx <= 1:
                continue
            left_end = float(lw[left_idx]["end"])
            right_start = float(lw[right_idx]["start"])
            add_estimated_run(list(range(left_idx + 1, right_idx)), left_end, right_start)

        # At line edges, use the local median character duration and stop at
        # the neighboring line, song boundary, or nearest trusted word.
        first_idx, last_idx = matched[0], matched[-1]
        line_floor = max(0.0, line_bounds[li - 1][1] if li > 0 else 0.0)
        line_ceiling = min(duration, line_bounds[li + 1][0] if li + 1 < n_lines else duration)
        if first_idx > 0:
            idxs = list(range(first_idx))
            required = char_rate * sum(weights[idx] for idx in idxs)
            add_estimated_run(idxs, max(line_floor, float(lw[first_idx]["start"]) - required), float(lw[first_idx]["start"]))
        if last_idx + 1 < len(lw):
            idxs = list(range(last_idx + 1, len(lw)))
            required = char_rate * sum(weights[idx] for idx in idxs)
            add_estimated_run(idxs, float(lw[last_idx]["end"]), min(line_ceiling, float(lw[last_idx]["end"]) + required))

        inferred = [word for word in lw if word is not None]
        if inferred:
            old_start, old_end, old_conf, anchored = line_bounds[li]
            ls = min(float(word["start"]) for word in inferred)
            le = max(float(word["end"]) for word in inferred)
            mean_prob = float(np.mean([float(word.get("probability") or 0.0) for word in inferred]))
            line_bounds[li] = (ls, max(ls + 0.25, le), round(mean_prob, 3), anchored)

    segments: list[dict] = []
    any_word_level = False
    for li, item in enumerate(user_lines):
        ls, le, conf, has_anchor = line_bounds[li]
        lw = line_words[li]
        all_words_valid = bool(lw) and all(w is not None for w in lw)
        words_clean: list[dict] = []
        if all_words_valid:
            typed_words = [w for w in lw if w is not None]
            words_clean = _validate_words(typed_words, ls, le)
            if words_clean:
                any_word_level = True

        txt = str(item["text"])
        segments.append(
            {
                "id": li + 1,
                "start": round(max(0.0, ls), 2),
                "end": round(max(ls + 0.15, le), 2),
                "text": txt,
                "lang": infer_segment_script_lang(txt, primary_lang),
                "section": item.get("section"),
                "confidence": conf if has_anchor else None,
                "low_confidence": not has_anchor or conf < 0.50 or any(w.get("uncertain") for w in words_clean),
                "words": words_clean,
            }
        )

    return segments, ("word" if any_word_level and anchored_words >= max(1, total_words // 3) else "line")


# ================================================================== Step D: Structure Preservation
def _phrase_trigrams(text: str) -> set[str]:
    norm = "".join(c for c in (_normalize_char_for_match(ch) for ch in text) if c.isalnum())
    if len(norm) < 3:
        return {norm} if norm else set()
    return {norm[i : i + 3] for i in range(len(norm) - 2)}


def _dice_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def annotate_conservative_structure(segments: list[dict], duration: float) -> list[dict]:
    """Assign conservative section labels (`Verse`, `Chorus`, `Outro`) ONLY when clear repetition exists."""
    if len(segments) < 5 or any(s.get("section") for s in segments):
        return segments

    stanzas: list[list[int]] = []
    cur: list[int] = [0]
    for idx in range(1, len(segments)):
        gap = float(segments[idx]["start"]) - float(segments[idx - 1]["end"])
        if gap >= 2.2 or len(cur) >= 6:
            stanzas.append(cur)
            cur = [idx]
        else:
            cur.append(idx)
    if cur:
        stanzas.append(cur)

    if len(stanzas) < 3:
        return segments

    stanza_grams = [_phrase_trigrams(" ".join(segments[i]["text"] for i in st)) for st in stanzas]
    chorus_stanzas: set[int] = set()
    for i in range(len(stanzas)):
        if len(stanzas[i]) < 2:
            continue
        for j in range(i + 1, len(stanzas)):
            if len(stanzas[j]) < 2:
                continue
            time_sep = float(segments[stanzas[j][0]]["start"]) - float(segments[stanzas[i][0]]["start"])
            if time_sep >= 14.0 and _dice_similarity(stanza_grams[i], stanza_grams[j]) >= 0.72:
                chorus_stanzas.add(i)
                chorus_stanzas.add(j)

    if not chorus_stanzas:
        return segments

    out = [dict(s) for s in segments]
    verse_num = 1
    last_chorus = max(chorus_stanzas)
    for si, st in enumerate(stanzas):
        first_idx = st[0]
        if si in chorus_stanzas:
            out[first_idx]["section"] = "Chorus"
        elif si == len(stanzas) - 1 and si > last_chorus and (duration - float(out[first_idx]["start"])) <= 35.0:
            out[first_idx]["section"] = "Outro"
        elif len(st) >= 2 and si < last_chorus:
            out[first_idx]["section"] = f"Verse {verse_num}"
            verse_num += 1
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


# ================================================================== Per-Project Store
class ProjectLyricsStore:
    """Persist AI Lyrics state & transcript per project at `<project_dir>/lyrics/`."""

    def __init__(self, project: Project) -> None:
        self.project = project
        self.dir = project.dir / "lyrics"
        self.state_file = self.dir / "state.json"
        self.transcript_file = self.dir / "transcript.json"

    def get(self) -> dict[str, Any]:
        if not self.state_file.is_file():
            return {
                "project_id": self.project.id,
                "title": self.project.data.get("title", ""),
                "has_audio": bool(self.project.audio_path and self.project.audio_path.is_file()),
                "state": "idle",
                "mode": "extract",
                "error": None,
                "language": None,
                "language_name": None,
                "language_confidence": None,
                "detected_languages": [],
                "vocal_isolated": False,
                "vocal_cached": False,
                "alignment_method": None,
                "segment_count": 0,
                "duration": None,
                "updated": None,
                "transcript": None,
            }
        data = read_json(self.state_file)
        data["project_id"] = self.project.id
        data["title"] = self.project.data.get("title", data.get("title", ""))
        data["has_audio"] = bool(self.project.audio_path and self.project.audio_path.is_file())
        if self.transcript_file.is_file():
            try:
                data["transcript"] = read_json(self.transcript_file)
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
            return {
                "state": s.get("state", "idle"),
                "mode": s.get("mode", "extract"),
                "language": s.get("language"),
                "language_name": s.get("language_name"),
                "segment_count": s.get("segment_count", 0),
                "error": s.get("error"),
                "updated": s.get("updated"),
            }
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
                s["state"] = "cancelled"
                s["error"] = "Máy chủ khởi động lại nên tác vụ đã dừng."
                write_json(self.state_file, s)
        except Exception:
            pass


# ================================================================== Main Pipeline Execution
def execute_lyrics_pipeline(
    *,
    engine: Engine,
    cache: LyricsCache,
    source_audio: Path,
    audio_sha1: str | None = None,
    mode: str = "extract",  # "extract" | "align"
    user_lyrics_text: str = "",
    requested_language: str = "auto",
    model_id: str = "turbo",
    isolate_vocals: bool = True,
    force_separate: bool = False,
    chords: list[tuple[float, str]] | None = None,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Execute the complete 6-stage AI Lyrics pipeline and return the structured transcript dict."""
    ensure_windows_runtime()
    if not source_audio.is_file():
        raise FileNotFoundError("Không tìm thấy tệp âm thanh gốc để xử lý lời bài hát.")
    if source_audio.suffix.lower() in (".mid", ".midi"):
        raise ValueError("Không thể trích xuất lời hát từ tệp Piano MIDI. Cần tệp âm thanh gốc chứa giọng hát.")

    if should_stop():
        raise Cancelled()

    if hasattr(engine, "unload"):
        engine.unload()

    valid_ids = {m["id"] for m in LYRICS_MODELS}
    if model_id not in valid_ids:
        model_id = "turbo"
    req_lang = (requested_language or "auto").strip().lower()

    progress(0.05, "preparing_audio")
    log("Bước 1/5 · Đang kiểm tra và chuẩn bị tệp âm thanh gốc...")
    sha1 = audio_sha1 or file_sha1(source_audio)

    with tempfile.TemporaryDirectory(prefix="harmony-lyrics-") as tmp_str:
        tmp_dir = Path(tmp_str)
        mix_16k_wav = tmp_dir / "mix_16k.wav"
        convert_audio_ffmpeg(source_audio, mix_16k_wav, sample_rate=16000, channels=1, vocal_bandpass=True)
        mix_16k = load_wav_16k(mix_16k_wav)
        duration = round(float(mix_16k.shape[0]) / 16000.0, 2)

        if should_stop():
            raise Cancelled()

        asr_wav, vocal_isolated, vocal_cached = isolate_vocals_cached(
            engine=engine,
            cache=cache,
            source_audio=source_audio,
            audio_sha1=sha1,
            tmp_dir=tmp_dir,
            want_isolate=isolate_vocals,
            force_separate=force_separate,
            progress=progress,
            should_stop=should_stop,
            log=log,
        )
        vocal_16k = load_wav_16k(asr_wav)
        # Compute pure vocal envelope BEFORE blending any mix audio so instrumental intros/outros stay at 0 RMS
        pure_rms_env, pure_vocal_floor = compute_vocal_envelope(vocal_16k)

        vocal_rms = float(np.sqrt(np.mean(np.square(vocal_16k, dtype=np.float64)) + 1e-12))
        mix_rms = float(np.sqrt(np.mean(np.square(mix_16k, dtype=np.float64)) + 1e-12))
        if vocal_isolated and vocal_rms >= 0.0018:
            # Blend 86% BS-RoFormer isolated vocals + 14% vocal-bandpass mix so sharp unvoiced
            # consonants (p, t, k, s, b) remain crisp while 86% of accompaniment/drums stay suppressed.
            n_min = min(vocal_16k.shape[0], mix_16k.shape[0])
            vocal_16k = (vocal_16k[:n_min] * 0.86 + mix_16k[:n_min] * 0.14).astype(np.float32)
        elif mode == "align" and vocal_isolated and vocal_rms < 0.0018 and mix_rms > 0.008:
            log("Kênh hát rất nhỏ — kết hợp nhẹ tín hiệu gốc để hỗ trợ căn chỉnh dấu thời gian.")
            n_min = min(vocal_16k.shape[0], mix_16k.shape[0])
            vocal_16k = (vocal_16k[:n_min] * 0.65 + mix_16k[:n_min] * 0.35).astype(np.float32)

        if should_stop():
            raise Cancelled()

        cached_anchor_data = (
            cache.load_anchors(sha1, req_lang, model_id=model_id, vocal_isolated=vocal_isolated)
            if (mode == "align" and not force_separate)
            else None
        )
        if cached_anchor_data is not None:
            progress(0.68, "recognizing_lyrics")
            log("Tái sử dụng điểm neo âm học từ bộ nhớ đệm để đồng bộ lại nhanh...")
            primary_lang = (
                req_lang if (req_lang and req_lang != "auto") else str(cached_anchor_data.get("language") or "en")
            )
            lang_conf = float(cached_anchor_data.get("language_confidence") or 1.0)
            detected_languages = list(cached_anchor_data.get("detected_languages") or [])
            anchors = list(cached_anchor_data.get("anchors") or [])
            rms_env, vocal_floor = pure_rms_env, pure_vocal_floor
            raw_segments = []
        else:
            asr_out = run_asr_recognition(
                engine=engine,
                audio_16k=vocal_16k,
                duration=duration,
                vocal_isolated=vocal_isolated,
                requested_lang=req_lang,
                model_id=model_id,
                progress=progress,
                should_stop=should_stop,
                log=log,
                rms_env=pure_rms_env,
                vocal_floor=pure_vocal_floor,
            )
            primary_lang = asr_out["language"]
            lang_conf = asr_out["language_confidence"]
            detected_languages = asr_out["detected_languages"]
            raw_segments = asr_out["raw_segments"]
            anchors = asr_out["anchors"]
            rms_env = asr_out["rms_env"]
            vocal_floor = asr_out["vocal_floor"]
            cache.save_anchors(
                sha1,
                req_lang,
                {
                    "model": model_id,
                    "vocal_isolated": vocal_isolated,
                    "language": primary_lang,
                    "language_confidence": lang_conf,
                    "detected_languages": detected_languages,
                    "anchors": anchors,
                },
                model_id=model_id,
                vocal_isolated=vocal_isolated,
            )

        if should_stop():
            raise Cancelled()

        progress(0.88, "aligning_timestamps")
        if mode == "align":
            log("Bước 4/5 · Đang căn chỉnh lời bài hát của bạn khớp với trục thời gian giọng hát...")
            parsed_lines = parse_user_lyrics(user_lyrics_text)
            if not parsed_lines:
                raise ValueError("Nội dung lời bài hát trống. Vui lòng nhập lời bài hát để căn chỉnh.")
            if req_lang in ("", "auto"):
                sample_txt = " ".join(x["text"] or "" for x in parsed_lines[:8])
                primary_lang = infer_segment_script_lang(sample_txt, primary_lang or "en")
            segments, alignment_method = align_user_lyrics_to_anchors(
                user_lines=parsed_lines,
                anchors=anchors,
                rms_env=rms_env,
                vocal_floor=vocal_floor,
                duration=duration,
                primary_lang=primary_lang,
            )
            segments = annotate_conservative_structure(segments, duration)
        else:
            log("Bước 4/5 · Đang hoàn thiện dấu thời gian từng từ/câu và cấu trúc bài hát...")
            segments = [
                {
                    "id": idx + 1,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "lang": seg["lang"],
                    "section": None,
                    "confidence": seg["confidence"],
                    "low_confidence": seg["low_confidence"],
                    "words": seg["words"],
                }
                for idx, seg in enumerate(raw_segments)
            ]
            segments = annotate_conservative_structure(segments, duration)
            has_words = any(bool(s.get("words")) for s in segments)
            alignment_method = "word" if has_words else "line"

        segments = attach_chords(segments, chords)

        seg_langs: dict[str, int] = {}
        for seg in segments:
            lg = str(seg.get("lang") or primary_lang)
            seg_langs[lg] = seg_langs.get(lg, 0) + 1
        if segments and seg_langs:
            existing = {d["code"] for d in detected_languages}
            for code, cnt in sorted(seg_langs.items(), key=lambda kv: kv[1], reverse=True):
                ratio = cnt / len(segments)
                if code not in existing and ratio >= 0.08:
                    detected_languages.append(
                        {"code": code, "name": language_label(code), "probability": round(ratio, 4)}
                    )

        progress(0.96, "saving_transcript")
        log(f"Bước 5/5 · Đang lưu bản lời ({len(segments)} câu · {language_label(primary_lang)})...")

        return {
            "mode": mode,
            "model": model_id,
            "audio_sha1": sha1,
            "language": primary_lang,
            "language_name": language_label(primary_lang),
            "language_confidence": round(float(lang_conf), 4),
            "detected_languages": detected_languages,
            "is_multilingual": len({s.get("lang") for s in segments if s.get("lang")}) > 1,
            "vocal_isolated": vocal_isolated,
            "vocal_cached": vocal_cached,
            "alignment_method": alignment_method,
            "duration": duration,
            "segments": segments,
            "full_text": build_full_text(segments, include_sections=False),
            "structured_text": build_full_text(segments, include_sections=True),
        }


def sanitize_edited_segments(raw_segments: list[dict], fallback_lang: str = "en") -> list[dict]:
    """Validate and clean user-edited lyric segments from the frontend Lyrics Editor."""
    cleaned: list[dict] = []
    for seg in raw_segments:
        txt = str(seg.get("text") or "").strip()
        if not txt:
            continue
        try:
            s_time = max(0.0, round(float(seg.get("start") or 0.0), 2))
        except (TypeError, ValueError):
            s_time = 0.0
        try:
            e_time = max(round(s_time + 0.05, 2), round(float(seg.get("end") or (s_time + 2.0)), 2))
        except (TypeError, ValueError):
            e_time = round(s_time + 2.0, 2)

        sec_raw = seg.get("section")
        section = str(sec_raw).strip() if sec_raw else None
        lang = str(seg.get("lang") or infer_segment_script_lang(txt, fallback_lang))

        orig_words = seg.get("words") if isinstance(seg.get("words"), list) else []
        valid_words: list[dict] = []
        if orig_words:
            joined_words = "".join(str(w.get("word") or "").strip() for w in orig_words)
            joined_txt = re.sub(r"\s+", "", txt)
            if joined_words == joined_txt:
                valid_words = _validate_words(orig_words, s_time - 0.05, e_time + 0.25)

        review_status_raw = seg.get("review_status")
        review_status = (
            review_status_raw
            if review_status_raw in ("needs_review", "verified", "edited", "skipped")
            else None
        )
        review_reason = str(seg.get("review_reason") or "").strip() or None

        cleaned.append(
            {
                "id": len(cleaned) + 1,
                "start": s_time,
                "end": e_time,
                "text": txt,
                "lang": lang,
                "section": section,
                "chord": seg.get("chord") or None,
                "confidence": seg.get("confidence"),
                "low_confidence": bool(seg.get("low_confidence", False)),
                "review_status": review_status,
                "review_reason": review_reason,
                "words": valid_words,
            }
        )
    return cleaned


def _norm_text_for_cmp(text: str) -> str:
    s = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return re.sub(r"\s+", " ", s)


def _decode_single_clip_cuda(
    w_model: Any,
    clip_audio: np.ndarray,
    clip_start: float,
    seg_start: float,
    seg_end: float,
    decode_lang: str | None,
    initial_prompt: str | None,
) -> dict[str, Any] | None:
    """High-precision Whisper decoding pass on a short single-line audio slice (CUDA FP16-Storage / FP32-Compute)."""
    kwargs: dict[str, Any] = {
        "language": decode_lang,
        "task": "transcribe",
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "temperature": (0.0, 0.2),
        "beam_size": 8,
        "best_of": 8,
        "patience": 1.4,
        "compression_ratio_threshold": 2.3,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.60,
        "fp16": False,
        "verbose": False,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    res = w_model.transcribe(clip_audio, **kwargs)
    raw_segs = res.get("segments") or []
    if not raw_segs:
        return None

    all_words: list[dict] = []
    avg_lps: list[float] = []
    no_speeches: list[float] = []
    raw_texts: list[str] = []

    for seg in raw_segs:
        t = str(seg.get("text") or "").strip()
        if not t or NOISE_ONLY_PATTERN.match(t) or HALLUCINATION_PATTERNS.match(t):
            continue
        raw_texts.append(t)
        avg_lps.append(float(seg.get("avg_logprob") or -0.5))
        no_speeches.append(float(seg.get("no_speech_prob") or 0.0))
        for w in seg.get("words") or []:
            all_words.append(
                {
                    "word": str(w.get("word") or ""),
                    "start": clip_start + float(w.get("start") or 0.0),
                    "end": clip_start + float(w.get("end") or 0.2),
                    "probability": float(w.get("probability") or 0.0),
                }
            )

    if not raw_texts:
        return None

    # Keep words whose midpoint falls within the target segment window (with small tolerance)
    kept_words = [
        w
        for w in all_words
        if (seg_start - 0.24) <= ((w["start"] + w["end"]) * 0.5) <= (seg_end + 0.24)
    ]
    if not kept_words:
        kept_words = all_words

    if kept_words:
        cand_text = "".join(w["word"] for w in kept_words).strip()
        cand_text = re.sub(r"\s+", " ", cand_text).strip()
    else:
        cand_text = " ".join(raw_texts).strip()

    if not cand_text or NOISE_ONLY_PATTERN.match(cand_text) or HALLUCINATION_PATTERNS.match(cand_text):
        return None

    valid_words = _validate_words(kept_words, seg_start - 0.18, seg_end + 0.22)
    avg_lp = float(np.mean(avg_lps)) if avg_lps else -0.5
    no_speech = float(np.mean(no_speeches)) if no_speeches else 0.0

    if valid_words:
        probs_w = [w["probability"] for w in valid_words if w.get("probability") is not None]
        w_conf = float(np.mean(probs_w)) if probs_w else math.exp(max(-3.0, min(0.0, avg_lp)))
    else:
        w_conf = math.exp(max(-3.0, min(0.0, avg_lp)))

    conf = round(max(0.05, min(0.99, 0.72 * w_conf + 0.28 * (1.0 - min(1.0, no_speech)))), 3)
    uncertain_words = sum(1 for w in valid_words if w.get("uncertain"))
    uncertain_ratio = (uncertain_words / len(valid_words)) if valid_words else 0.0
    low_conf = bool(
        conf < 0.56
        or avg_lp < -0.82
        or (uncertain_words >= 2 and uncertain_ratio >= 0.28)
    )
    score = conf - 0.12 * uncertain_ratio + (0.04 if not low_conf else 0.0)
    return {
        "text": cand_text,
        "words": valid_words,
        "confidence": conf,
        "low_confidence": low_conf,
        "uncertain_words": uncertain_words,
        "score": score,
    }


def _decode_single_clip_cpu(
    fw_model: Any,
    clip_audio: np.ndarray,
    clip_start: float,
    seg_start: float,
    seg_end: float,
    decode_lang: str | None,
    initial_prompt: str | None,
) -> dict[str, Any] | None:
    """High-precision Faster-Whisper INT8 decoding pass on CPU for a short single-line audio slice."""
    segs_iter, _ = fw_model.transcribe(
        clip_audio,
        language=decode_lang,
        task="transcribe",
        beam_size=8,
        best_of=8,
        patience=1.4,
        temperature=(0.0, 0.2),
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
        vad_filter=False,
        no_speech_threshold=0.60,
    )
    all_words: list[dict] = []
    avg_lps: list[float] = []
    no_speeches: list[float] = []
    raw_texts: list[str] = []

    for seg in segs_iter:
        t = str(seg.text or "").strip()
        if not t or NOISE_ONLY_PATTERN.match(t) or HALLUCINATION_PATTERNS.match(t):
            continue
        raw_texts.append(t)
        avg_lps.append(float(getattr(seg, "avg_logprob", -0.5) or -0.5))
        no_speeches.append(float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
        for w in getattr(seg, "words", None) or []:
            all_words.append(
                {
                    "word": str(getattr(w, "word", "") or ""),
                    "start": clip_start + float(getattr(w, "start", 0.0) or 0.0),
                    "end": clip_start + float(getattr(w, "end", 0.2) or 0.2),
                    "probability": float(getattr(w, "probability", 0.0) or 0.0),
                }
            )

    if not raw_texts:
        return None

    kept_words = [
        w
        for w in all_words
        if (seg_start - 0.24) <= ((w["start"] + w["end"]) * 0.5) <= (seg_end + 0.24)
    ]
    if not kept_words:
        kept_words = all_words

    if kept_words:
        cand_text = "".join(w["word"] for w in kept_words).strip()
        cand_text = re.sub(r"\s+", " ", cand_text).strip()
    else:
        cand_text = " ".join(raw_texts).strip()

    if not cand_text or NOISE_ONLY_PATTERN.match(cand_text) or HALLUCINATION_PATTERNS.match(cand_text):
        return None

    valid_words = _validate_words(kept_words, seg_start - 0.18, seg_end + 0.22)
    avg_lp = float(np.mean(avg_lps)) if avg_lps else -0.5
    no_speech = float(np.mean(no_speeches)) if no_speeches else 0.0

    if valid_words:
        probs_w = [w["probability"] for w in valid_words if w.get("probability") is not None]
        w_conf = float(np.mean(probs_w)) if probs_w else math.exp(max(-3.0, min(0.0, avg_lp)))
    else:
        w_conf = math.exp(max(-3.0, min(0.0, avg_lp)))

    conf = round(max(0.05, min(0.99, 0.72 * w_conf + 0.28 * (1.0 - min(1.0, no_speech)))), 3)
    uncertain_words = sum(1 for w in valid_words if w.get("uncertain"))
    uncertain_ratio = (uncertain_words / len(valid_words)) if valid_words else 0.0
    low_conf = bool(
        conf < 0.56
        or avg_lp < -0.82
        or (uncertain_words >= 2 and uncertain_ratio >= 0.28)
    )
    score = conf - 0.12 * uncertain_ratio + (0.04 if not low_conf else 0.0)
    return {
        "text": cand_text,
        "words": valid_words,
        "confidence": conf,
        "low_confidence": low_conf,
        "uncertain_words": uncertain_words,
        "score": score,
    }


def retranscribe_flagged_segments(
    *,
    cache: LyricsCache,
    source_audio: Path,
    audio_sha1: str | None,
    segments: list[dict],
    target_ids: list[int] | None = None,
    language: str | None = None,
    model_id: str = "turbo",
    isolate_vocals: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Precision multi-pass re-transcription for flagged ('Cần kiểm tra') or user-selected lines.

    Combines 3 focused high-beam decoding passes per short line:
      1. Pure isolated vocal stem (100% BS-RoFormer) + neighboring confirmed-line context prompt
      2. Crisp vocal-enhanced blend (82% vocal + 18% original) + context prompt
      3. Pure isolated vocal stem without prompt (unbiased fallback)
    and selects the highest-confidence, hallucination-filtered candidate.

    Returns (updated_segments, line_results).
    """
    ensure_windows_runtime()

    if not segments:
        return [], []

    # Determine which segment indices to re-transcribe
    target_id_set = set(target_ids) if target_ids else None
    target_indices: list[int] = []
    for idx, seg in enumerate(segments):
        seg_id = int(seg.get("id") or (idx + 1))
        if target_id_set is not None:
            if seg_id in target_id_set:
                target_indices.append(idx)
        else:
            rs = seg.get("review_status")
            is_flagged = (
                rs == "needs_review"
                or (rs not in ("verified", "edited", "skipped") and bool(seg.get("low_confidence")))
            )
            if is_flagged:
                target_indices.append(idx)

    if not target_indices:
        return list(segments), []

    sha1 = audio_sha1 or file_sha1(source_audio)
    decode_lang = language if (language and language != "auto") else None

    with tempfile.TemporaryDirectory(prefix="lyrics-refine-") as tmp:
        tmp_dir = Path(tmp)
        orig_wav = tmp_dir / "orig_16k.wav"
        convert_audio_ffmpeg(source_audio, orig_wav, sample_rate=16000)
        orig_16k = load_wav_16k(orig_wav)
        duration = float(orig_16k.shape[0]) / 16000.0

        pure_vocal = _level_vocal_dynamics(orig_16k)
        crisp_blend = orig_16k
        has_vocal_stem = False

        if isolate_vocals and cache.has_vocal(sha1):
            voc_wav = tmp_dir / "voc_16k.wav"
            convert_audio_ffmpeg(cache.vocal_path(sha1), voc_wav, sample_rate=16000)
            voc_16k = load_wav_16k(voc_wav)
            n = min(voc_16k.shape[0], orig_16k.shape[0])
            if n > 0:
                voc_16k = voc_16k[:n]
                orig_16k = orig_16k[:n]
                duration = float(n) / 16000.0
                pure_vocal = _level_vocal_dynamics(voc_16k)
                crisp_blend = _level_vocal_dynamics(
                    np.clip(0.82 * voc_16k + 0.18 * orig_16k, -1.0, 1.0).astype(np.float32)
                )
                has_vocal_stem = True

        # Load Whisper model ONCE for all target lines
        import torch

        use_cuda = torch.cuda.is_available()
        w_model = None
        fw_model = None
        if use_cuda:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / (1024**3)
            total_gb = total_b / (1024**3)
            eff_model = model_id or "turbo"
            if eff_model == "large-v3" and (total_gb <= 4.5 or free_gb < 3.4):
                # Keep exact large-v3 model via Faster-Whisper CUDA int8_float32 (~1.6 GB VRAM)
                from faster_whisper import WhisperModel

                fw_model = WhisperModel(eff_model, device="cuda", compute_type="int8_float32")
            else:
                import whisper

                _patch_openai_whisper_for_turing_fp32()
                w_model = whisper.load_model(eff_model, device="cpu")
                for m in w_model.modules():
                    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                        m.half()
                w_model = w_model.to("cuda")
        else:
            from faster_whisper import WhisperModel

            eff_model = model_id or "turbo"
            fw_model = WhisperModel(eff_model, device="cpu", compute_type="int8")

        updated_segments = [dict(s) for s in segments]
        results: list[dict] = []

        try:
            for idx in target_indices:
                seg = updated_segments[idx]
                seg_id = int(seg.get("id") or (idx + 1))
                seg_start = max(0.0, float(seg.get("start") or 0.0))
                seg_end = max(seg_start + 0.15, float(seg.get("end") or (seg_start + 2.0)))
                old_text = str(seg.get("text") or "").strip()
                old_conf = float(seg.get("confidence") if seg.get("confidence") is not None else 0.5)

                gap_before = (seg_start - float(updated_segments[idx - 1]["end"])) if idx > 0 else seg_start
                gap_after = (
                    (float(updated_segments[idx + 1]["start"]) - seg_end)
                    if idx + 1 < len(updated_segments)
                    else (duration - seg_end)
                )
                pad_l = min(0.45, max(0.10, gap_before * 0.55))
                pad_r = min(0.45, max(0.10, gap_after * 0.55))
                clip_start = max(0.0, seg_start - pad_l)
                clip_end = min(duration, seg_end + pad_r)

                s_idx = max(0, int(clip_start * 16000))
                e_idx = min(pure_vocal.shape[0], int(clip_end * 16000))
                if e_idx - s_idx < 3200:
                    results.append(
                        {
                            "id": seg_id,
                            "old_text": old_text,
                            "new_text": old_text,
                            "old_confidence": round(old_conf, 3),
                            "new_confidence": round(old_conf, 3),
                            "improved": False,
                            "changed": False,
                            "auto_resolved": False,
                            "variant": "Đoạn âm thanh quá ngắn",
                        }
                    )
                    continue

                # Gather up to 2 preceding non-flagged lines for lyrical & grammatical context
                prev_texts: list[str] = []
                for p_i in range(max(0, idx - 2), idx):
                    p_txt = str(updated_segments[p_i].get("text") or "").strip()
                    if p_txt and not updated_segments[p_i].get("low_confidence"):
                        prev_texts.append(p_txt)
                context_prompt = " ".join(prev_texts).strip() or None
                prev_norm = _norm_text_for_cmp(prev_texts[-1]) if prev_texts else ""
                old_norm = _norm_text_for_cmp(old_text)

                passes: list[tuple[str, np.ndarray, str | None]] = []
                if context_prompt:
                    passes.append(
                        (
                            "Giọng tách thuần + ngữ cảnh" if has_vocal_stem else "Âm thanh cân bằng + ngữ cảnh",
                            pure_vocal[s_idx:e_idx],
                            context_prompt,
                        )
                    )
                    passes.append(
                        (
                            "Hòa trộn rõ phụ âm + ngữ cảnh" if has_vocal_stem else "Âm thanh gốc + ngữ cảnh",
                            crisp_blend[s_idx:e_idx],
                            context_prompt,
                        )
                    )
                passes.append(
                    (
                        "Giọng tách thuần (độc lập)" if has_vocal_stem else "Âm thanh cân bằng",
                        pure_vocal[s_idx:e_idx],
                        None,
                    )
                )
                if not context_prompt and has_vocal_stem:
                    passes.append(("Hòa trộn rõ phụ âm", crisp_blend[s_idx:e_idx], None))

                best_cand: dict[str, Any] | None = None
                best_variant = ""

                for variant_label, clip_arr, prompt_txt in passes:
                    try:
                        if use_cuda and w_model is not None:
                            cand = _decode_single_clip_cuda(
                                w_model,
                                clip_arr,
                                clip_start,
                                seg_start,
                                seg_end,
                                decode_lang,
                                prompt_txt,
                            )
                        else:
                            cand = _decode_single_clip_cpu(
                                fw_model,
                                clip_arr,
                                clip_start,
                                seg_start,
                                seg_end,
                                decode_lang,
                                prompt_txt,
                            )
                    except Exception:
                        cand = None

                    if not cand:
                        continue
                    cand_norm = _norm_text_for_cmp(cand["text"])
                    # Reject accidental copy of previous line from initial_prompt
                    if prev_norm and cand_norm == prev_norm and old_norm != prev_norm:
                        continue
                    if best_cand is None or cand["score"] > best_cand["score"]:
                        best_cand = cand
                        best_variant = variant_label

                if best_cand is not None and (
                    best_cand["confidence"] >= old_conf - 0.03
                    or (not best_cand["low_confidence"] and bool(seg.get("low_confidence")))
                ):
                    new_text = best_cand["text"]
                    new_conf = best_cand["confidence"]
                    new_low = best_cand["low_confidence"]
                    auto_resolved = bool((not new_low) and new_conf >= 0.68)
                    changed = _norm_text_for_cmp(new_text) != old_norm
                    improved = bool(new_conf > old_conf + 0.015 or auto_resolved or (changed and new_conf >= old_conf))

                    seg["text"] = new_text
                    seg["words"] = best_cand["words"]
                    seg["confidence"] = new_conf
                    seg["low_confidence"] = new_low
                    seg["lang"] = infer_segment_script_lang(new_text, decode_lang or str(seg.get("lang") or "en"))
                    if auto_resolved:
                        seg["review_status"] = "verified"
                        seg["review_reason"] = None
                    else:
                        seg["review_status"] = "needs_review"

                    results.append(
                        {
                            "id": seg_id,
                            "old_text": old_text,
                            "new_text": new_text,
                            "old_confidence": round(old_conf, 3),
                            "new_confidence": round(new_conf, 3),
                            "improved": improved,
                            "changed": changed,
                            "auto_resolved": auto_resolved,
                            "variant": best_variant,
                        }
                    )
                else:
                    results.append(
                        {
                            "id": seg_id,
                            "old_text": old_text,
                            "new_text": old_text,
                            "old_confidence": round(old_conf, 3),
                            "new_confidence": round(old_conf, 3),
                            "improved": False,
                            "changed": False,
                            "auto_resolved": False,
                            "variant": "Giữ nguyên bản cũ (đã tối ưu)",
                        }
                    )
        finally:
            if w_model is not None:
                del w_model
            if fw_model is not None:
                del fw_model
            gc.collect()
            if use_cuda:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return updated_segments, results
