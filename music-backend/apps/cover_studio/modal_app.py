"""Harmony Auralis — Unified Modal.com Serverless GPU (NVIDIA L4 24GB) Workers.

Deploy with:
    modal deploy music-backend/apps/cover_studio/modal_app.py

Hosts ALL AI models for Harmony Auralis on pay-per-second NVIDIA L4 GPUs:
  1. LyricsGPUWorker:
     - BS-RoFormer vocal stem separation (FP16, batch_size=4)
     - Qwen3-ASR 1.7B singing recognition and Whisper Large v3 alignment
     - Precision line re-transcription (`retranscribe_flagged`)
  2. CoverGPUWorker:
     - Tsumugi stem-separated audio-to-MIDI transcription (`instrument_agnostic_amt`)
     - SheetSage2-Refined beat & melody annotation + audio frame feature extraction
     - Piano Cover generation (`piano_cover` V1/V2)
     - Piano Performance-to-Score MusicXML (`piano_perf2score` + `piano_score`)
"""

from __future__ import annotations

import gc
import io
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "harmony-auralis-gpu"
VOLUME_NAME = "harmony-auralis-models"
CACHE_MOUNT = "/vol/models"
GPU_TYPE = "L4"
LYRICS_GPU_TYPES = ["L4", "A10G"]

BACKEND_APPS_DIR = Path(__file__).resolve().parent.parent  # music-backend/apps

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

COMMON_ENV = {
    "PYTHONPATH": "/app/apps:/opt/tsumugi",
    "PYTHONUNBUFFERED": "1",
}

# ------------------------------------------------------------------ Image 1: Songscription (BS-RoFormer + Whisper + Qwen3-ASR)
lyrics_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "sox")
    .pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "torchcodec==0.5",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy<2.3",
        "scipy",
        "soundfile",
        "transformers==4.57.6",
        "accelerate==1.12.0",
        "safetensors",
        "sentencepiece",
        "stem-splitter==0.0.8",
        "huggingface_hub",
        "qwen-asr==0.0.6",
    )
    .env(COMMON_ENV)
    .add_local_dir(
        str(BACKEND_APPS_DIR),
        remote_path="/app/apps",
        ignore=["**/__pycache__", "**/*.pyc", "accounts/**", "roles_permissions/**"],
    )
)

# ------------------------------------------------------------------ Image 2: Piano Cover + Tsumugi + SheetSage + Score
cover_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2.3",
        "scipy",
        "soundfile",
        "symusic",
        "safetensors",
        "huggingface_hub",
        "dlchordx",
        "chord-romanizer",
        "einops",
        "mido",
        "pretty-midi",
        "pyyaml",
        "librosa",
        "stem-splitter",
        "tqdm",
        "pillow",
        "git+https://github.com/anime-song/SheetSage2-Refined",
    )
    .run_commands("git clone --depth 1 https://github.com/anime-song/tsumugi /opt/tsumugi")
    .env(COMMON_ENV)
    .add_local_dir(
        str(BACKEND_APPS_DIR),
        remote_path="/app/apps",
        ignore=["**/__pycache__", "**/*.pyc", "accounts/**", "roles_permissions/**"],
    )
)


def _setup_cache_dirs() -> None:
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_MOUNT}/hf/hub"
    os.environ["TORCH_HOME"] = f"{CACHE_MOUNT}/torch"
    os.environ["XDG_CACHE_HOME"] = f"{CACHE_MOUNT}/xdg"
    for sub in ("hf", "hf/hub", "torch", "xdg", "audio_separator", "tsumugi_checkpoints", "tsumugi_beat_chord"):
        Path(CACHE_MOUNT, sub).mkdir(parents=True, exist_ok=True)
    # Symlink Tsumugi checkpoint folders to persistent Modal Volume
    tsumugi_root = Path("/opt/tsumugi")
    if tsumugi_root.is_dir():
        for local_name, vol_sub in (
            ("checkpoints", "tsumugi_checkpoints"),
            ("beat_chord_checkpoints", "tsumugi_beat_chord"),
        ):
            target = Path(CACHE_MOUNT, vol_sub)
            link = tsumugi_root / local_name
            if link.is_symlink():
                continue
            if link.is_dir():
                for child in link.iterdir():
                    dst = target / child.name
                    if not dst.exists():
                        shutil.move(str(child), str(dst))
                shutil.rmtree(link, ignore_errors=True)
            link.symlink_to(target)


def _run_bs_roformer_l4(
    audio_path: Path,
    out_16k_wav: Path,
    logs: list[str],
    out_instrumental_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Separate vocals & instrumental using BS-RoFormer on NVIDIA L4 FP16 (~5-8s for a 4-min song)."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from stem_splitter.inference import SeparationConfig, _separate_one_file, load_mss_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="modal-vocals-") as work_str:
        work_dir = Path(work_str)
        # Convert input audio to 44.1kHz stereo WAV for BS-RoFormer
        from cover_studio.lyrics_service import convert_audio_ffmpeg, encode_wav_to_mp3, mix_non_vocal_stems

        in_44k = work_dir / "in_44k.wav"
        convert_audio_ffmpeg(audio_path, in_44k, sample_rate=44100, channels=2)

        cfg = SeparationConfig(
            target_sample_rate=44100,
            device_preference=str(device),
            use_half_precision=(device.type == "cuda"),
            amp_dtype="float16",
            torch_compile=False,
            chunk_size=441_000,
            batch_size=4 if device.type == "cuda" else 1,
        )
        logs.append("[Modal L4] Đang tách giọng hát & nhạc nền BS-RoFormer trên GPU L4 (FP16, batch=4)...")
        model = load_mss_model(cfg, device)
        try:
            dtype = torch.float16 if (cfg.use_half_precision and not cfg.enable_autocast) else torch.float32
            saved = _separate_one_file(in_44k, work_dir, cfg, model, device, dtype)
            vocals_path = saved.get("vocals")
            if vocals_path is None or not vocals_path.is_file():
                raise RuntimeError("BS-RoFormer không tạo được kênh giọng hát.")
            vocals_np, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
            mono = vocals_np.mean(axis=1)
            if sr != 16000:
                mono = AF.resample(torch.from_numpy(mono).unsqueeze(0), orig_freq=sr, new_freq=16000).squeeze(0).numpy()
            mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
            sf.write(str(out_16k_wav), mono, 16000, subtype="PCM_16")

            saved_inst: Path | None = None
            if out_instrumental_path is not None:
                inst_wav = mix_non_vocal_stems(saved, work_dir / "instrumental_44k.wav")
                if inst_wav is not None:
                    if out_instrumental_path.suffix.lower() == ".mp3" and encode_wav_to_mp3(inst_wav, out_instrumental_path, bitrate="192k"):
                        saved_inst = out_instrumental_path
                    else:
                        wav_fallback = out_instrumental_path.with_suffix(".wav")
                        shutil.copy2(inst_wav, wav_fallback)
                        saved_inst = wav_fallback
            return out_16k_wav, saved_inst
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ================================================================== Worker 1: Songscription / Lyrics AI
@app.cls(
    image=lyrics_image,
    gpu=LYRICS_GPU_TYPES,
    cpu=4.0,
    memory=16384,
    volumes={CACHE_MOUNT: model_volume},
    scaledown_window=45,
    timeout=600,
)
class LyricsGPUWorker:
    @modal.enter()
    def startup(self) -> None:
        print("[LyricsGPUWorker] startup begin", flush=True)
        _setup_cache_dirs()
        if "/app/apps" not in sys.path:
            sys.path.insert(0, "/app/apps")
        print("[LyricsGPUWorker] startup complete", flush=True)

    @modal.method()
    def ping(self) -> dict[str, Any]:
        from importlib.metadata import version

        import torch
        import transformers
        from qwen_asr import Qwen3ASRModel
        from stem_splitter.inference import SeparationConfig  # noqa: F401

        info = {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
            "qwen_asr": version("qwen-asr"),
            "qwen_repo": "Qwen/Qwen3-ASR-1.7B",
            "qwen_raw_inference": callable(getattr(Qwen3ASRModel, "_infer_asr", None)),
            "stem_splitter": version("stem-splitter"),
            "torchcodec": version("torchcodec"),
            "whisper_repo": "openai/whisper-large-v3",
            "cache_exists": Path(CACHE_MOUNT).is_dir(),
        }
        print("[LyricsGPUWorker] ping:", info, flush=True)
        return info

    @modal.method()
    def separate_instrumental(
        self,
        audio_bytes: bytes,
        ext: str = ".mp3",
    ) -> dict[str, Any]:
        """Run BS-RoFormer 6-stem on NVIDIA L4 to produce a studio-grade 44.1kHz stereo instrumental
        (`bass + drums + other + guitar + piano`, completely free of lead & stereo backing vocals)."""
        logs: list[str] = []
        with tempfile.TemporaryDirectory(prefix="modal-inst-") as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / f"source{ext or '.mp3'}"
            src.write_bytes(audio_bytes)
            voc_wav = tmp_dir / "vocals_16k.wav"
            inst_mp3 = tmp_dir / "instrumental.mp3"
            _, saved_inst = _run_bs_roformer_l4(src, voc_wav, logs, out_instrumental_path=inst_mp3)
            model_volume.commit()
            return {
                "instrumental_audio_bytes": saved_inst.read_bytes() if (saved_inst and saved_inst.is_file()) else None,
                "instrumental_ext": saved_inst.suffix.lower() if saved_inst else ".mp3",
                "vocal_wav_bytes": voc_wav.read_bytes() if voc_wav.is_file() else None,
                "logs": logs,
            }

    @modal.method()
    def extract_lyrics(
        self,
        audio_bytes: bytes,
        ext: str = ".mp3",
        vocal_wav_bytes: bytes | None = None,
        mode: str = "extract",
        user_lyrics_text: str = "",
        requested_language: str = "auto",
        model_id: str = "qwen3-asr-1.7b",
        isolate_vocals: bool = True,
        chords: list[list[Any]] | None = None,
        audio_sha1: str | None = None,
        track_title: str = "",
        **_ignored_kwargs: Any,
    ) -> dict[str, Any]:
        import numpy as np

        from cover_studio.lyrics_schema import (
            DEFAULT_STREAM_ID,
            SCHEMA_VERSION,
            assign_overlap_groups,
            compute_pipeline_config_hash,
            get_multi_singer_separation_status,
            migrate_segment,
            stream_label,
        )
        from cover_studio.lyrics_service import (
            _blend_vocal_with_mix,
            _normalize_vocal_rms,
            align_user_lyrics_to_anchors,
            annotate_conservative_structure,
            attach_chords,
            build_full_text,
            compute_vocal_envelope,
            convert_audio_ffmpeg,
            language_label,
            load_wav_16k,
            parse_user_lyrics,
        )
        from cover_studio.music_asr_engine import normalize_model_id, run_music_asr_pipeline

        logs: list[str] = []
        log = logs.append
        model_id = normalize_model_id(model_id)
        req_lang = (requested_language or "auto").strip().lower()

        with tempfile.TemporaryDirectory(prefix="modal-lyrics-") as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / f"source{ext or '.mp3'}"
            src.write_bytes(audio_bytes)

            mix_wav = tmp_dir / "mix_16k.wav"
            convert_audio_ffmpeg(src, mix_wav, sample_rate=16000, channels=1, vocal_bandpass=False)
            mix_16k = load_wav_16k(mix_wav)
            duration = round(float(mix_16k.shape[0]) / 16000.0, 2)

            vocal_isolated = False
            vocal_cached = False
            out_vocal_bytes: bytes | None = None
            out_inst_bytes: bytes | None = None
            out_inst_ext = ".mp3"

            if isolate_vocals:
                voc_wav = tmp_dir / "vocals_16k.wav"
                if vocal_wav_bytes:
                    voc_wav.write_bytes(vocal_wav_bytes)
                    vocal_isolated = True
                    vocal_cached = True
                    log("[Modal L4] Tái sử dụng kênh giọng hát BS-RoFormer đã cache từ local.")
                else:
                    inst_target = tmp_dir / "instrumental.mp3"
                    _, inst_saved = _run_bs_roformer_l4(src, voc_wav, logs, out_instrumental_path=inst_target)
                    vocal_isolated = True
                    out_vocal_bytes = voc_wav.read_bytes()
                    if inst_saved is not None and inst_saved.is_file():
                        out_inst_bytes = inst_saved.read_bytes()
                        out_inst_ext = inst_saved.suffix.lower() or ".mp3"
                full_vocal_16k = load_wav_16k(voc_wav)
            else:
                full_vocal_16k = mix_16k

            n_min = min(full_vocal_16k.shape[0], mix_16k.shape[0])
            full_vocal_16k = full_vocal_16k[:n_min]
            mix_16k = mix_16k[:n_min]

            pure_rms_env, pure_vocal_floor = compute_vocal_envelope(full_vocal_16k)
            pure_vocal_norm = _normalize_vocal_rms(full_vocal_16k)
            vocal_rms = float(np.sqrt(np.mean(np.square(full_vocal_16k, dtype=np.float64)) + 1e-12))
            mix_rms = float(np.sqrt(np.mean(np.square(mix_16k, dtype=np.float64)) + 1e-12))
            if vocal_isolated and vocal_rms >= 0.0018:
                # Feed the pure BS-RoFormer vocal stem directly into speech-to-text for maximum lyric accuracy
                asr_audio = pure_vocal_norm
            elif mode == "align" and vocal_isolated and vocal_rms < 0.0018 and mix_rms > 0.008:
                asr_audio = _blend_vocal_with_mix(full_vocal_16k, mix_16k, mix_weight=0.25)
            else:
                asr_audio = pure_vocal_norm

            log(f"[Modal L4] Đang chạy Songscription ({model_id}) trên {'Kênh vocal thuần' if vocal_isolated else 'âm thanh gốc'}...")
            asr_out = run_music_asr_pipeline(
                asr_audio=asr_audio,
                pure_vocal_audio=pure_vocal_norm if vocal_isolated else None,
                original_audio=mix_16k,
                duration=duration,
                vocal_isolated=vocal_isolated,
                rms_env=pure_rms_env,
                vocal_floor=pure_vocal_floor,
                requested_lang=req_lang,
                model_id=model_id,
                progress=lambda _p, _m: None,
                should_stop=lambda: False,
                log=log,
                audio_sha1=audio_sha1 or "",
                track_title=track_title or "",
                mode=mode,
                user_lyrics_text=user_lyrics_text or "",
            )

            primary_lang = asr_out["language"]
            lang_conf = asr_out["language_confidence"]
            detected_languages = asr_out["detected_languages"]
            raw_segments = asr_out["raw_segments"]
            raw_asr_segments = list(asr_out.get("raw_asr_segments") or [])
            diagnostics = dict(asr_out.get("diagnostics") or {})
            diagnostics["vocal_separation"] = {
                "isolated": vocal_isolated,
                "cached": vocal_cached,
                "provider": "bs-roformer-modal-l4" if vocal_isolated else "original-mix-16k",
                "vocal_floor": round(float(pure_vocal_floor), 5),
                "duration_sec": duration,
            }
            anchors = asr_out["anchors"]
            rms_env = asr_out["rms_env"]
            vocal_floor = asr_out["vocal_floor"]

            if mode == "align" and not diagnostics.get("reference_lyrics"):
                user_lines = parse_user_lyrics(user_lyrics_text)
                if not user_lines:
                    raise ValueError("Chưa có lời bài hát để căn chỉnh.")
                segments, alignment_method = align_user_lyrics_to_anchors(
                    user_lines=user_lines,
                    anchors=anchors,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    duration=duration,
                    primary_lang=primary_lang,
                )
            else:
                segments = [
                    migrate_segment(
                        {
                            **seg,
                            "id": idx + 1,
                            "section": seg.get("section"),
                        },
                        fallback_lang=primary_lang,
                    )
                    for idx, seg in enumerate(raw_segments)
                ]
                segments = annotate_conservative_structure(segments, duration)
                segments = assign_overlap_groups(segments)
                alignment_method = "word" if any(bool(s.get("words")) for s in segments) else "line"

            chord_tuples = [(float(c[0]), str(c[1])) for c in chords] if chords else None
            segments = attach_chords(segments, chord_tuples)

            seg_langs: dict[str, int] = {}
            for seg in segments:
                lg = str(seg.get("language") or seg.get("lang") or "").strip()
                if lg:
                    seg_langs[lg] = seg_langs.get(lg, 0) + 1
            if segments and seg_langs:
                existing = {d["code"] for d in detected_languages}
                for code, cnt in sorted(seg_langs.items(), key=lambda kv: kv[1], reverse=True):
                    ratio = cnt / len(segments)
                    if code not in existing and ratio >= 0.08:
                        detected_languages.append(
                            {"code": code, "name": language_label(code), "probability": round(ratio, 4)}
                        )

            cfg_hash = compute_pipeline_config_hash(
                model_id=model_id,
                requested_lang=req_lang,
                vocal_isolated=vocal_isolated,
            )
            streams_present = sorted({str(s.get("stream_id") or DEFAULT_STREAM_ID) for s in segments}) or [DEFAULT_STREAM_ID]
            transcript = {
                "schema_version": SCHEMA_VERSION,
                "mode": mode,
                "model": model_id,
                "config_hash": cfg_hash,
                "audio_sha1": audio_sha1 or "",
                "language": primary_lang,
                "language_name": language_label(primary_lang),
                "language_confidence": round(float(lang_conf), 4),
                "detected_languages": detected_languages,
                "is_multilingual": len({s.get("language") for s in segments if s.get("language")}) > 1,
                "vocal_isolated": vocal_isolated,
                "vocal_cached": vocal_cached,
                "alignment_method": alignment_method,
                "duration": duration,
                "streams": [
                    {"id": sid, "label": stream_label(sid), "has_isolated_audio": False}
                    for sid in streams_present
                ],
                "multi_singer_separation": get_multi_singer_separation_status(),
                "segments": segments,
                "raw_asr_segments": raw_asr_segments,
                "diagnostics": diagnostics,
                "edit_history": [],
                "full_text": build_full_text(segments, include_sections=False),
                "structured_text": build_full_text(segments, include_sections=True),
            }
            model_volume.commit()
            return {
                "transcript": transcript,
                "vocal_wav_bytes": out_vocal_bytes,
                "instrumental_audio_bytes": out_inst_bytes,
                "instrumental_ext": out_inst_ext,
                "stream_wav_bytes": {},
                "anchors": anchors,
                "logs": logs,
            }

    @modal.method()
    def retranscribe_flagged(
        self,
        audio_bytes: bytes,
        ext: str = ".mp3",
        vocal_wav_bytes: bytes | None = None,
        segments: list[dict] | None = None,
        target_ids: list[int] | None = None,
        language: str | None = None,
        model_id: str = "qwen3-asr-1.7b",
        isolate_vocals: bool = True,
        **_ignored_kwargs: Any,
    ) -> dict[str, Any]:
        from cover_studio.lyrics_service import (
            _blend_vocal_with_mix,
            _normalize_vocal_rms,
            convert_audio_ffmpeg,
            load_wav_16k,
        )
        from cover_studio.music_asr_engine import normalize_model_id, refine_segments_with_music_asr

        segs = list(segments or [])
        if not segs:
            return {"updated_segments": [], "results": []}

        target_id_set = set(target_ids) if target_ids else None
        target_indices = [
            idx
            for idx, seg in enumerate(segs)
            if (
                (int(seg.get("id") or (idx + 1)) in target_id_set)
                if target_id_set is not None
                else (
                    seg.get("review_status") == "needs_review"
                    or (seg.get("review_status") not in ("verified", "edited", "skipped") and bool(seg.get("low_confidence")))
                )
            )
        ]
        if not target_indices:
            return {"updated_segments": segs, "results": []}

        decode_lang = language if (language and language != "auto") else None

        with tempfile.TemporaryDirectory(prefix="modal-refine-") as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / f"source{ext or '.mp3'}"
            src.write_bytes(audio_bytes)
            orig_wav = tmp_dir / "orig_16k.wav"
            convert_audio_ffmpeg(src, orig_wav, sample_rate=16000)
            orig_16k = load_wav_16k(orig_wav)
            duration = float(orig_16k.shape[0]) / 16000.0

            pure_vocal = _normalize_vocal_rms(orig_16k)
            crisp_blend = _normalize_vocal_rms(orig_16k)
            has_vocal_stem = False
            out_vocal_bytes: bytes | None = None
            out_inst_bytes: bytes | None = None
            out_inst_ext = ".mp3"

            if isolate_vocals:
                voc_wav = tmp_dir / "voc_16k.wav"
                if vocal_wav_bytes:
                    voc_wav.write_bytes(vocal_wav_bytes)
                else:
                    sep_logs: list[str] = []
                    inst_target = tmp_dir / "instrumental.mp3"
                    _, inst_saved = _run_bs_roformer_l4(src, voc_wav, sep_logs, out_instrumental_path=inst_target)
                    if voc_wav.is_file():
                        out_vocal_bytes = voc_wav.read_bytes()
                    if inst_saved is not None and inst_saved.is_file():
                        out_inst_bytes = inst_saved.read_bytes()
                        out_inst_ext = inst_saved.suffix.lower() or ".mp3"
                if voc_wav.is_file():
                    voc_16k = load_wav_16k(voc_wav)
                    n = min(voc_16k.shape[0], orig_16k.shape[0])
                    if n > 0:
                        voc_16k = voc_16k[:n]
                        orig_16k = orig_16k[:n]
                        duration = float(n) / 16000.0
                        pure_vocal = _normalize_vocal_rms(voc_16k)
                        crisp_blend = _blend_vocal_with_mix(voc_16k, orig_16k, mix_weight=0.08)
                        has_vocal_stem = True

            updated_segments, results = refine_segments_with_music_asr(
                segments=segs,
                target_indices=target_indices,
                pure_vocal=pure_vocal,
                crisp_blend=crisp_blend,
                duration=duration,
                has_vocal_stem=has_vocal_stem,
                decode_lang=decode_lang,
                model_id=normalize_model_id(model_id),
                label_prefix="Modal L4",
            )
            model_volume.commit()
            return {
                "updated_segments": updated_segments,
                "results": results,
                "vocal_wav_bytes": out_vocal_bytes,
                "instrumental_audio_bytes": out_inst_bytes,
                "instrumental_ext": out_inst_ext,
            }


# ================================================================== Worker 2: Tsumugi + SheetSage + Piano Cover + Score
@app.cls(
    image=cover_image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16384,
    volumes={CACHE_MOUNT: model_volume},
    scaledown_window=45,
    timeout=900,
)
class CoverGPUWorker:
    @modal.enter()
    def startup(self) -> None:
        _setup_cache_dirs()
        for p in ("/app/apps", "/opt/tsumugi"):
            if p not in sys.path:
                sys.path.insert(0, p)
        self._cover_model = None
        self._cover_spec: str | None = None
        self._score_tuple = None
        self._score_spec: str | None = None

    def _get_cover_model(self, model_spec: str):
        import torch
        from piano_cover.hub import load_cover
        from piano_cover.style import style_slot

        if self._cover_spec == model_spec and self._cover_model is not None:
            return self._cover_model
        self._score_tuple = None
        self._score_spec = None
        self._cover_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_cover(model_spec, device="cuda")
        style_slot(model)
        self._cover_model = model
        self._cover_spec = model_spec
        return model

    def _get_score_model(self, score_spec: str):
        import torch
        from piano_perf2score.hub import load_perf2score

        if self._score_spec == score_spec and self._score_tuple is not None:
            return self._score_tuple
        self._cover_model = None
        self._cover_spec = None
        self._score_tuple = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._score_tuple = load_perf2score(score_spec, device="cuda")
        self._score_spec = score_spec
        return self._score_tuple

    @modal.method()
    def transcribe_audio(
        self,
        audio_bytes: bytes,
        ext: str = ".mp3",
        sheetsage_model: str | None = None,
    ) -> dict[str, Any]:
        """Run Tsumugi stem-separated AMT + SheetSage2-Refined beat/melody annotation on NVIDIA L4."""
        import torch

        logs: list[str] = []
        # Release cached cover/score models to give full 24GB VRAM to Tsumugi
        self._cover_model = None
        self._cover_spec = None
        self._score_tuple = None
        self._score_spec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="modal-tsumugi-") as tmp:
            tmp_dir = Path(tmp)
            audio_path = tmp_dir / f"source{ext or '.mp3'}"
            audio_path.write_bytes(audio_bytes)
            work_dir = tmp_dir / "work"
            try:
                os.chdir("/opt/tsumugi")
                from instrument_agnostic_amt.cli import infer_stem

                logs.append("[Modal L4] Đang tách nhạc cụ & chép nốt MIDI bằng Tsumugi trên GPU L4...")
                result = infer_stem.run_stem_separated_transcription(
                    audio_path,
                    output_root=work_dir,
                    refine_instruments=True,
                    predict_velocity=True,
                    predict_beat_chord=True,
                    amp=True,
                    window_batch_size=8,
                    merge_onset_ms=50.0,
                    semi_crf_backend="auto",
                    compile_model=False,
                    compile_velocity=False,
                    device="cuda",
                )
                merged = Path(result["merged_midi_path"])
                midi_bytes = merged.read_bytes()
            finally:
                os.chdir(orig_cwd)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Run SheetSage2-Refined immediately on the same L4 GPU
            from piano_ar.sheetsage import SheetSageAnnotator

            logs.append("[Modal L4] Đang phân tích nhịp phách & giai điệu bằng SheetSage2-Refined...")
            annotator = SheetSageAnnotator(sheetsage_model, device="cuda")
            try:
                sheetsage = annotator.annotate(audio_path)
            finally:
                del annotator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_volume.commit()
            return {
                "midi_bytes": midi_bytes,
                "sheetsage": sheetsage,
                "logs": logs,
            }

    @modal.method()
    def sheetsage_annotate(
        self,
        audio_bytes: bytes,
        ext: str = ".mp3",
        sheetsage_model: str | None = None,
    ) -> dict[str, Any]:
        import torch
        from piano_ar.sheetsage import SheetSageAnnotator

        with tempfile.TemporaryDirectory(prefix="modal-ss-") as tmp:
            audio_path = Path(tmp) / f"source{ext or '.mp3'}"
            audio_path.write_bytes(audio_bytes)
            annotator = SheetSageAnnotator(sheetsage_model, device="cuda")
            try:
                res = annotator.annotate(audio_path)
            finally:
                del annotator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            model_volume.commit()
            return res

    @modal.method()
    def generate_covers(
        self,
        model_spec: str,
        source_midi_bytes: bytes,
        audio_bytes: bytes | None,
        audio_ext: str,
        sheetsage_json: dict | None,
        frames_npz_bytes: bytes | None,
        params_dict: dict,
        num_samples: int,
        seed: int,
        prompt_patches: list | None = None,
        style_vector: list[float] | None = None,
        sheetsage_model: str | None = None,
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from piano_ar.beat_tokenizer import BeatTokenizer
        from piano_ar.evaluation import events_end_frame
        from piano_ar.sheetsage import SheetSageAnnotator
        from piano_ar.tokenizer import KIND, KIND_NOTE
        from piano_cover.audio_frames import FrameExtractor, FrameProjector
        from piano_cover.generate import CoverParams, generate_covers, prepare_source
        from piano_cover.metadata import apply_sheetsage_beats, score_for_cover
        from piano_cover.style import set_style
        from symusic import Score

        with tempfile.TemporaryDirectory(prefix="modal-cover-") as tmp:
            tmp_dir = Path(tmp)
            src_mid = tmp_dir / "source.mid"
            src_mid.write_bytes(source_midi_bytes)
            audio_path: Path | None = None
            if audio_bytes:
                audio_path = tmp_dir / f"source{audio_ext or '.mp3'}"
                audio_path.write_bytes(audio_bytes)

            model = self._get_cover_model(model_spec)
            params = CoverParams.from_dict(params_dict)
            if style_vector is not None:
                params.channel = set_style(model, style_vector)

            c = model.config
            need_ss = c.sheetsage_beats or c.sheetsage_melody or isinstance(model.tokenizer, BeatTokenizer)
            ss_data = sheetsage_json
            if need_ss and ss_data is None:
                if audio_path is None:
                    raise FileNotFoundError("Cần tệp âm thanh gốc để dự đoán nhịp SheetSage.")
                annotator = SheetSageAnnotator(sheetsage_model, device="cuda")
                try:
                    ss_data = annotator.annotate(audio_path)
                finally:
                    del annotator
                    gc.collect()
                    torch.cuda.empty_cache()

            ss_path: Path | None = None
            if ss_data is not None:
                ss_path = tmp_dir / "sheetsage.json"
                ss_path.write_text(json.dumps(ss_data, ensure_ascii=False), encoding="utf-8")
                apply_sheetsage_beats(src_mid, ss_data)

            frames_arr = None
            out_frames_bytes: bytes | None = None
            if c.frame_dims:
                if frames_npz_bytes:
                    frames_arr = np.load(io.BytesIO(frames_npz_bytes))["frames"]
                else:
                    if audio_path is None:
                        raise FileNotFoundError("Cần tệp âm thanh gốc để trích xuất đặc trưng âm thanh.")
                    pca_path = getattr(model, "frames_pca_path", None)
                    if pca_path is None or not Path(pca_path).is_file():
                        raise FileNotFoundError(f"Không tìm thấy PCA frames cho model {model_spec}.")
                    extractor = FrameExtractor(sheetsage_model, device="cuda")
                    try:
                        frames_arr = FrameProjector(extractor, Path(pca_path))(audio_path)
                    finally:
                        del extractor
                        gc.collect()
                        torch.cuda.empty_cache()
                    buf = io.BytesIO()
                    np.savez(buf, frames=frames_arr)
                    out_frames_bytes = buf.getvalue()

            source = prepare_source(src_mid, model.tokenizer, c, ss_path, frames_arr)
            tokenizer = model.tokenizer
            frame_rate = tokenizer.config.frame_rate

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            covers = generate_covers(
                model, source, params, num_samples=num_samples, prompt=prompt_patches
            )

            source_score = None
            try:
                source_score = Score(str(src_mid)).to("second")
            except Exception:
                pass

            out_covers: list[dict[str, Any]] = []
            for idx, cover in enumerate(covers):
                mid_out = tmp_dir / f"cover_{idx}.mid"
                try:
                    score_for_cover(tokenizer, cover.events, source_score).dump_midi(str(mid_out))
                    metadata_copied = source_score is not None
                except Exception:
                    tokenizer.events_to_midi(cover.events, mid_out)
                    metadata_copied = False
                notes = int((cover.events[:, KIND] == KIND_NOTE).sum()) if len(cover.events) else 0
                dur = float(events_end_frame(cover.events) / frame_rate) if len(cover.events) else 0.0
                out_covers.append(
                    {
                        "cover_midi_bytes": mid_out.read_bytes(),
                        "patches": cover.patches,
                        "duration": dur,
                        "notes": notes,
                        "metadata_copied": metadata_copied,
                    }
                )

            model_volume.commit()
            return {
                "covers": out_covers,
                "sheetsage": ss_data,
                "frames_npz_bytes": out_frames_bytes,
                "num_channels": int(model.decoder.config.num_channels),
                "updated_source_midi_bytes": src_mid.read_bytes(),
            }

    @modal.method()
    def generate_score(
        self,
        score_model_spec: str,
        cover_midi_bytes: bytes,
        source_midi_bytes: bytes | None,
        sheetsage_json: dict,
        title: str,
    ) -> dict[str, Any]:
        from piano_perf2score.generate import score_from_cover
        from piano_score.musicxml import write_musicxml

        with tempfile.TemporaryDirectory(prefix="modal-score-") as tmp:
            tmp_dir = Path(tmp)
            cov_mid = tmp_dir / "cover.mid"
            cov_mid.write_bytes(cover_midi_bytes)
            src_mid: Path | None = None
            if source_midi_bytes:
                src_mid = tmp_dir / "source.mid"
                src_mid.write_bytes(source_midi_bytes)

            model, tokenizer, banned = self._get_score_model(score_model_spec)
            measures = score_from_cover(model, tokenizer, cov_mid, sheetsage_json, src_mid, banned=banned)
            xml_path = tmp_dir / "score.musicxml"
            write_musicxml(measures, xml_path, title=title)
            model_volume.commit()
            return {
                "musicxml": xml_path.read_text(encoding="utf-8"),
                "measures": len(measures),
            }
