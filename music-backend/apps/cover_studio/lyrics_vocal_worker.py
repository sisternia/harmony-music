"""Subprocess worker for vocal stem isolation using BS-RoFormer in the .tsumugi environment.

Runs isolated inside `<tsumugi>/.venv/Scripts/python.exe` so stem_splitter and torchcodec
dependencies do not collide with the main Cover Studio `.venv`.
Handles Windows FFmpeg DLL loading (`os.add_dll_directory`), GTX 1650 4 GB VRAM chunk sizing,
live batch progress (`@@progress`), NaN validation, and automatic FP32 / CPU fallback.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def emit(kind: str, data: dict) -> None:
    print(f"@@{kind} {json.dumps(data, separators=(',', ':'))}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate vocal stem using BS-RoFormer")
    parser.add_argument("--tsumugi-dir", required=True)
    parser.add_argument("--audio", required=True, help="Input 44.1kHz stereo WAV")
    parser.add_argument("--out", required=True, help="Output 16kHz mono PCM16 vocals WAV")
    parser.add_argument("--ffmpeg-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    audio_in = Path(args.audio).resolve()
    audio_out = Path(args.out).resolve()
    tsumugi_dir = Path(args.tsumugi_dir).resolve()

    if args.ffmpeg_dir:
        ff_dir = Path(args.ffmpeg_dir).resolve()
        if ff_dir.is_dir():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(ff_dir))
            os.environ["PATH"] = f"{ff_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    os.chdir(tsumugi_dir)
    sys.path.insert(0, str(tsumugi_dir))

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from stem_splitter.inference import SeparationConfig, _separate_one_file, load_mss_model, resolve_device

    requested_device = None if args.device == "auto" else args.device
    device = resolve_device(requested_device)

    vram_gb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        vram_gb = total_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        if free_gb < 1.1:
            print(f"[Vocals] Free VRAM low ({free_gb:.2f} GB), falling back to CPU.", flush=True)
            device = torch.device("cpu")
        else:
            frac = min(0.92, (free_bytes * 0.88) / max(total_bytes, 1))
            torch.cuda.set_per_process_memory_fraction(frac)

    # Memory-aware chunking: GTX 1650 (4 GB VRAM) uses smaller chunks & batch_size=1
    is_low_vram = device.type == "cuda" and vram_gb <= 5.0
    chunk_size = 264_600 if is_low_vram else 441_000  # ~6s on 4GB VRAM, ~10s otherwise

    attempts: list[tuple[torch.device, bool, int]] = []
    if device.type == "cuda":
        attempts.append((device, True, chunk_size))
        attempts.append((device, False, 220_500))
    attempts.append((torch.device("cpu"), False, 352_800))

    last_err: Exception | None = None
    for run_device, use_half, run_chunk in attempts:
        with tempfile.TemporaryDirectory(prefix="harmony-vocals-") as work_str:
            work_dir = Path(work_str)
            model = None
            try:
                cfg = SeparationConfig(
                    target_sample_rate=44100,
                    device_preference=str(run_device),
                    use_half_precision=use_half and run_device.type == "cuda",
                    amp_dtype="float16",
                    torch_compile=False,
                    chunk_size=run_chunk,
                    batch_size=1,
                )
                print(
                    f"[Vocals] Running BS-RoFormer on {run_device} "
                    f"(fp16={cfg.use_half_precision}, chunk={run_chunk})...",
                    flush=True,
                )
                model = load_mss_model(cfg, run_device)
                dtype = torch.float16 if (cfg.use_half_precision and not cfg.enable_autocast) else torch.float32

                def on_batch(done: int, total: int) -> None:
                    emit("progress", {"stage": "separating_vocals", "done": int(done), "total": int(total)})

                saved = _separate_one_file(audio_in, work_dir, cfg, model, run_device, dtype, on_batch=on_batch)
                vocals_path = saved.get("vocals")
                if vocals_path is None or not vocals_path.is_file():
                    raise RuntimeError("BS-RoFormer did not produce a vocals stem.")

                vocals_np, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
                if not np.isfinite(vocals_np).all():
                    raise ValueError("Non-finite values (NaN/Inf) detected in separated vocals; retrying safely.")

                # Downmix stereo vocals to mono and resample cleanly to 16,000 Hz for ASR & caching
                mono = vocals_np.mean(axis=1)
                if sr != 16000:
                    mono_t = torch.from_numpy(mono).unsqueeze(0)
                    mono = AF.resample(mono_t, orig_freq=sr, new_freq=16000).squeeze(0).numpy()

                mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
                audio_out.parent.mkdir(parents=True, exist_ok=True)
                tmp_out = audio_out.with_suffix(".tmp.wav")
                sf.write(str(tmp_out), mono, 16000, subtype="PCM_16")
                os.replace(tmp_out, audio_out)
                emit("done", {"out": str(audio_out), "device": str(run_device)})
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[Vocals] Attempt on {run_device} (fp16={use_half}) failed: {exc}", flush=True)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raise SystemExit(f"Vocal separation failed across all fallbacks: {last_err}")


if __name__ == "__main__":
    main()

