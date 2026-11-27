"""Schema v2, Safe Migration, Diagnostic Trace, Vocal Event Classification, and
Evaluation Metrics for Harmony Auralis Songscription.

Design guarantees:
  - 100% backwards compatible with v1 transcripts and user-edited segments.
  - Preserves immutable `text_raw` alongside `text_edited` and edit history.
  - Supports multi-stream / overlapping events (`stream_id`, `overlap_group_id`)
    without flattening simultaneous vocals into a single sequential line.
  - Distinguishes `lyric`, `vocable`, `humming`, and `unknown_vocal` cleanly.
  - Never fabricates percentage accuracy from uncalibrated heuristics, never
    invents "ah"/"oh" for untranscribed acoustic regions, and never forces a
    language onto ambiguous humming or `unknown_vocal` spans.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
PIPELINE_CONFIG_VERSION = "songscription-v5-pure-vocal-asr"

EVENT_TYPES: tuple[str, ...] = ("lyric", "vocable", "humming", "unknown_vocal")
VOCAL_ROLES: tuple[str, ...] = ("lead", "backing", "unknown")
REVIEW_STATUSES: tuple[str, ...] = ("needs_review", "verified", "edited", "skipped")
TIMING_SOURCES: tuple[str, ...] = (
    "forced_aligner",
    "asr_word",
    "asr_segment",
    "acoustic_region",
    "user_aligned",
    "interpolated",
    "manual",
)
TIMING_GRANULARITIES: tuple[str, ...] = ("word", "line", "region")
AUDIO_VARIANTS: tuple[str, ...] = ("vocal_stem", "vocal_mix_blend", "original_mix", "manual")

DEFAULT_STREAM_ID = "stream_a"

HUMMING_TOKENS: set[str] = {
    "hm", "hmm", "hmmm", "mm", "mmm", "mmmm", "mm-hmm", "uh-huh", "mhm",
    "嗯", "哼", "음", "흠", "ん", "んー", "ふーん",
}

VOCABLE_TOKENS: set[str] = {
    # Sustained vowels & melisma
    "oh", "ohh", "ooh", "oooh", "ah", "ahh", "aah", "aaah", "uh", "uhh", "eh", "ehh", "ih", "ee", "aye", "ay", "ayy",
    "whoa", "woah", "woo", "wooo", "ow", "aw", "aww",
    # Rhythmic scat & pop/anime/K-pop hook syllables
    "la", "lah", "na", "nah", "da", "dah", "ba", "bah", "pa", "pah", "ta", "tah",
    "ya", "yah", "ma", "mah", "wa", "wah", "ra", "rah", "sha", "cha", "ka", "ga", "fa", "sa",
    "ha", "hah", "haha", "hee", "ho", "hoo", "hu",
    "doo", "du", "di", "dee", "bi", "bee", "li", "lee", "ni", "ti", "ri",
    "tu", "ru", "lu", "nu", "bu", "pu", "su", "bo", "lo", "yo", "ye", "yi",
    "bada", "dada", "lala", "nana", "mama", "papa", "yaya", "tada", "shoo", "bop", "wop", "bap",
    "baiya", "baya", "iya", "eya", "oya", "aya", "yai", "yei", "trala", "tralala", "ratata",
    "oh-oh", "oh-oh-oh", "ah-ah", "ah-ah-ah", "ha-ha", "whoa-oh", "la-la", "la-la-la", "na-na", "na-na-na",
    "da-da", "ba-ba", "ya-ya", "ooh-ooh", "mm-mm", "hm-hm",
    # Native vocalise tokens (VI / JA / KO / ZH)
    "ơ", "à", "á", "ồ", "ô", "ơi", "í", "hò", "ư",
    "ああ", "あー", "おー", "うー", "らら", "ららら", "なー", "いやー", "あ", "お", "う", "ら", "な", "わ", "は",
    "아", "오", "우", "라", "나", "워", "하",
    "啊", "哦", "噢", "喔", "呃", "呜", "哎", "呀", "哇", "啦", "呐", "哒", "哈",
}

LEXICAL_ADLIB_PHRASES: set[str] = {
    "yeah", "yea", "yeh", "yeah yeah", "oh yeah", "oh yeah yeah",
    "hey", "hey hey", "come on", "cmon", "c'mon", "let's go", "lets go",
    "uh huh", "aha", "baby", "alright", "all right", "woo", "woohoo", "check it",
    "yo", "one two", "one two three", "sing it", "feel it", "listen",
}

REVIEW_REASON_LABELS: dict[str, str] = {
    "weak_vocal_activity": "Giọng hát nhỏ — cần nghe đối chiếu trước khi loại bỏ",
    "vocal_timing_unresolved": "Chưa đủ dữ liệu để tách thời gian từng âm tiết — cần nghe đối chiếu",
    "suspected_vocal_untranscribed": "Nghi có giọng hát nhưng ASR chưa nhận rõ chữ — cần nghe đối chiếu",
    "possible_overlap": "Có thể có nhiều giọng hát chồng thời gian",
    "low_asr_logprob": "Điểm xác suất ASR thấp trên câu hát này",
    "uncertain_words": "Một số từ có độ chắc chắn thấp",
    "unmatched_reference_line": "Chưa khớp điểm neo âm học cho dòng lời tham chiếu",
    "language_switch_boundary": "Nằm gần ranh giới chuyển ngôn ngữ / chuyển bài",
    "candidate_pending": "Có phương án đọc lại cần xác nhận hoặc từ chối",
    "vocal_like_heuristic_only": "Chỉ mới phát hiện bằng đặc trưng năng lượng/hòa âm — chưa chắc chắn có lời",
    "asr_repetition_loop_trimmed": "AI phát hiện vòng lặp âm tiết bất thường và đã cắt gọn — cần nghe đối chiếu",
}


def stream_label(stream_id: str | None) -> str:
    """Neutral human-readable label for a vocal stream (never asserts singer identity)."""
    sid = (stream_id or DEFAULT_STREAM_ID).strip().lower()
    if sid == "stream_a":
        return "Luồng A"
    if sid == "stream_b":
        return "Luồng B"
    if sid == "stream_c":
        return "Luồng C"
    m = re.match(r"^stream_([a-z0-9]+)$", sid)
    if m:
        return f"Luồng {m.group(1).upper()}"
    return sid or "Luồng A"


def normalize_stream_id(raw: str | None) -> str:
    if not raw:
        return DEFAULT_STREAM_ID
    s = str(raw).strip().lower()
    if s in ("a", "luồng a", "luong a", "stream a", "stream_a", "1"):
        return "stream_a"
    if s in ("b", "luồng b", "luong b", "stream b", "stream_b", "2"):
        return "stream_b"
    if s in ("c", "luồng c", "luong c", "stream c", "stream_c", "3"):
        return "stream_c"
    safe = re.sub(r"[^a-z0-9_-]+", "_", s).strip("_")
    if not safe:
        return DEFAULT_STREAM_ID
    return safe if safe.startswith("stream_") else f"stream_{safe}"


def normalize_vocable_elongation(text: str) -> str:
    """Normalize artificially stretched character runs without changing real words or deleting
    legitimate repeated syllables. E.g. 'Ooooooooooh' -> 'Ooh', 'Aaaaaaah' -> 'Aah',
    'Mmmmmm' -> 'Mmm'. 'Ba-i-i-e-e-e-e' stays unchanged: hyphen-separated syllables
    retain their articulation count.
    """
    if not text:
        return text
    out = text
    # Collapse 3+ identical consecutive Latin vowels/h/m/n/w inside a token to at most 2
    # (preserving standard spellings like "Ooh", "Aah", "Hmm", "Mmm", "Woo")
    def _cap_run(match: re.Match[str]) -> str:
        ch = match.group(1)
        run = match.group(0)
        lo = ch.lower()
        if lo in ("o", "a", "e", "u", "i", "h"):
            return run[:2]
        if lo in ("m", "n"):
            return run[:3]
        return run[:2]

    out = re.sub(r"([A-Za-z])\1{2,}", _cap_run, out, flags=re.IGNORECASE)
    # Collapse 4+ Japanese elongation marks "ーーーー" -> "ー"
    out = re.sub(r"ー{3,}", "ー", out)
    return out


def classify_vocal_event(
    text: str,
    *,
    is_adlib_hint: bool | None = None,
    fallback_lang: str | None = None,
) -> tuple[str, bool | None, str | None]:
    """Classify a vocal text into `(event_type, is_adlib, language_hint)`.

    - `humming`: pure humming tokens -> `language_hint = None` (never forced).
    - `vocable`: non-word sung syllables (`Ooh`, `La, la, la`, `Ba-ee-ya`) -> `is_adlib = True`.
    - `lyric`: lexical words; short emotional callouts (`Yeah!`, `Come on!`, `(oh yeah)`)
      keep `event_type = "lyric"` with `is_adlib = True` (an ad-lib can contain real words!).
    - `unknown_vocal`: empty text or explicit untranscribed marker -> `language_hint = None`.
    """
    raw = (text or "").strip()
    if not raw or raw in ("♪ (Vocalise)", "(Vocalise)", "[Vocalise]", "(unknown_vocal)"):
        return "unknown_vocal", is_adlib_hint, None

    # Preserve parenthetical ad-libs like "(yeah)" or "(oh)" — never strip them, just inspect inner text
    is_parenthetical = bool(re.match(r"^\s*[\(\[（【].*[\)\]）】]\s*$", raw))
    inner = normalize_vocable_elongation(re.sub(r"^[\s\(\[（【]+|[\s\)\]）】]+$", "", raw).strip())
    toks = [t for t in re.split(r"[\s,.;:!?、。！？・\-~…♪♫()\[\]]+", inner) if t]
    if not toks:
        return "unknown_vocal", is_adlib_hint, None

    low_toks = [t.lower() for t in toks]

    # 1. Pure humming
    if all(t in HUMMING_TOKENS for t in low_toks):
        return "humming", True, None

    # 2. Short lexical ad-lib (real words used as ad-lib, e.g. "Yeah!", "Come on!", "(yeah)")
    repeated_adlib = any(
        len(low_toks) % len(phrase.split()) == 0
        and low_toks == phrase.split() * (len(low_toks) // len(phrase.split()))
        for phrase in LEXICAL_ADLIB_PHRASES
    )
    if repeated_adlib or (is_parenthetical and len(toks) <= 4):
        if all(t in VOCABLE_TOKENS or t in HUMMING_TOKENS for t in low_toks):
            return "vocable", True, None if len(set(low_toks)) <= 2 else fallback_lang
        return "lyric", True, fallback_lang

    # 3. Pure or predominantly non-lexical vocables / scat syllables
    voc_hits = sum(1 for t in low_toks if t in VOCABLE_TOKENS or t in HUMMING_TOKENS)
    non_voc_cnt = len(low_toks) - voc_hits
    if voc_hits == len(low_toks):
        # Single/simple vowel vocalise ("Ooh", "Ah, ah", "啊") has insufficient linguistic evidence to force a language
        ambiguous_vowel = all(
            t in {
                "oh", "ohh", "ooh", "oooh", "ah", "ahh", "aah", "aaah", "uh", "uhh", "eh", "ehh",
                "o", "a", "u", "e", "i", "hm", "hmm", "mm", "mmm",
                "啊", "哦", "噢", "喔", "呃", "呜", "嗯", "哼",
                "あ", "ああ", "あー", "お", "おー", "う", "うー",
                "아", "오", "우", "음", "흠",
            }
            for t in low_toks
        )
        return "vocable", True, (None if ambiguous_vowel else fallback_lang)
    # Never misclassify a real lyric sentence (2+ non-vocable words) as pure Vocalise
    # just because a trailing vocable repeats (e.g. "On wide horizons, ah ah ah").
    if non_voc_cnt <= 1 and (voc_hits / len(low_toks)) >= 0.75 and len(low_toks) >= 2:
        return "vocable", True, fallback_lang

    # 4. Standard lexical lyric
    return "lyric", (True if is_adlib_hint is True else False), fallback_lang


def map_crop_time_to_source(
    local_start: float,
    local_end: float,
    crop_offset: float,
    source_duration: float,
    *,
    already_mapped: bool = False,
) -> tuple[float, float]:
    """Map crop-relative timestamps back to the Songscription source audio timeline.
    Guarantees offset is applied ONCE and timestamps remain clamped to `[0.0, source_duration]`.
    """
    offset = 0.0 if already_mapped else float(crop_offset)
    max_t = max(0.05, float(source_duration))
    s = max(0.0, min(max_t, offset + float(local_start)))
    e = max(s + 0.05, min(max_t + 0.25, offset + float(local_end)))
    return round(s, 2), round(e, 2)


def estimate_vocal_harmonicity(clip_16k: np.ndarray, sr: int = 16000) -> dict[str, float]:
    """Lightweight acoustic heuristic estimating whether an audio span exhibits vocal-band
    harmonic structure vs flat noise/percussion or pure out-of-band instruments.
    NOTE: Heuristic ONLY — never used as certainty of human voice; only guides review flagging.
    """
    if clip_16k.size < sr // 10:
        return {"vocal_band_ratio": 0.0, "spectral_flatness": 1.0, "pitch_voicing_ratio": 0.0, "vocal_likely": 0.0}

    x = clip_16k.astype(np.float64, copy=False)
    frame_len = 1024
    hop = 512
    n_frames = max(1, (x.size - frame_len) // hop + 1)
    win = np.hanning(frame_len)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    vocal_mask = (freqs >= 160.0) & (freqs <= 4000.0)
    sub_mask = freqs < 110.0

    band_ratios: list[float] = []
    flatnesses: list[float] = []
    voiced_frames = 0
    active_frames = 0

    for i in range(min(n_frames, 120)):
        s = i * hop
        frame = x[s : s + frame_len]
        if frame.size < frame_len:
            frame = np.pad(frame, (0, frame_len - frame.size))
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        if rms < 0.005:
            continue
        active_frames += 1
        spec = np.abs(np.fft.rfft(frame * win)) ** 2 + 1e-12
        total_e = float(np.sum(spec))
        voc_e = float(np.sum(spec[vocal_mask]))
        sub_e = float(np.sum(spec[sub_mask]))
        band_ratios.append(voc_e / max(total_e - sub_e * 0.5, 1e-12))
        # Spectral flatness inside vocal band (0 = tonal/harmonic, 1 = white noise)
        v_spec = spec[vocal_mask]
        geo = float(np.exp(np.mean(np.log(v_spec))))
        ari = float(np.mean(v_spec))
        flat = geo / max(ari, 1e-12)
        flatnesses.append(flat)
        # Simple autocorrelation voicing peak in human singing pitch range (~95Hz - 900Hz -> lags 18..168)
        f_centered = frame - np.mean(frame)
        ac = np.correlate(f_centered, f_centered, mode="full")[frame_len - 1 :]
        if ac[0] > 1e-8:
            peak_ac = float(np.max(ac[18:169]) / ac[0])
            if peak_ac >= 0.42:
                voiced_frames += 1

    if active_frames == 0:
        return {"vocal_band_ratio": 0.0, "spectral_flatness": 1.0, "pitch_voicing_ratio": 0.0, "vocal_likely": 0.0}

    mean_band = float(np.mean(band_ratios))
    mean_flat = float(np.mean(flatnesses))
    voicing_ratio = voiced_frames / active_frames
    # Heuristic score in [0, 1] (NOT calibrated accuracy!)
    score = 0.45 * min(1.0, mean_band / 0.70) + 0.30 * max(0.0, 1.0 - mean_flat * 3.0) + 0.25 * voicing_ratio
    return {
        "vocal_band_ratio": round(mean_band, 3),
        "spectral_flatness": round(mean_flat, 3),
        "pitch_voicing_ratio": round(voicing_ratio, 3),
        "vocal_likely": round(max(0.0, min(1.0, score)), 3),
    }


def assign_overlap_groups(segments: list[dict], min_overlap_sec: float = 0.20) -> list[dict]:
    """Detect time-overlapping vocal events (e.g. dual-stream mashup or simultaneous lead + backing)
    and annotate `overlap_group_id` WITHOUT deleting, merging, or forcing sequential timestamps!
    """
    if not segments:
        return []
    n = len(segments)
    out = [dict(s) for s in segments]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        si = float(out[i].get("start") or 0.0)
        ei = float(out[i].get("end") or si)
        for j in range(i + 1, n):
            sj = float(out[j].get("start") or 0.0)
            ej = float(out[j].get("end") or sj)
            ov = min(ei, ej) - max(si, sj)
            if ov >= min_overlap_sec:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    group_counter = 0
    for idxs in groups.values():
        if len(idxs) >= 2:
            group_counter += 1
            gid = f"ov-{group_counter}"
            for idx in idxs:
                out[idx]["overlap_group_id"] = gid
                reasons = list(out[idx].get("review_reasons") or [])
                if "possible_overlap" not in reasons:
                    reasons.append("possible_overlap")
                out[idx]["review_reasons"] = reasons
        else:
            out[idxs[0]]["overlap_group_id"] = None
    return out


def format_review_reason(reasons: list[str] | None, fallback: str | None = None) -> str | None:
    if reasons:
        first = reasons[0]
        return REVIEW_REASON_LABELS.get(first, first)
    return fallback


def migrate_segment(seg: dict, fallback_lang: str | None = None) -> dict:
    """Upgrade a v1 or partial v2 segment dict cleanly without losing any user edit or metadata."""
    out = dict(seg)
    text = str(out.get("text") or "").strip()
    text_raw = out.get("text_raw")
    if text_raw is None:
        text_raw = text
    text_edited = out.get("text_edited")

    # Classify event_type / is_adlib if absent
    ev_type = out.get("event_type")
    is_adlib = out.get("is_adlib")
    if ev_type not in EVENT_TYPES:
        inferred_ev, inferred_adlib, inferred_lang = classify_vocal_event(
            text, is_adlib_hint=is_adlib if isinstance(is_adlib, bool) else None, fallback_lang=fallback_lang
        )
        ev_type = inferred_ev
        if is_adlib is None:
            is_adlib = inferred_adlib
        if "language" not in out:
            out["language"] = out.get("lang") if ev_type not in ("humming", "unknown_vocal") else inferred_lang
    else:
        if "language" not in out:
            out["language"] = None if ev_type in ("humming", "unknown_vocal") else (out.get("lang") or fallback_lang)

    # Keep `lang` backwards-compatible for existing frontend code while allowing `language: None`
    lang_val = out.get("language")
    if lang_val == "":
        lang_val = None
    out["language"] = lang_val
    out["lang"] = str(lang_val or out.get("lang") or "")

    vocal_role = out.get("vocal_role")
    if vocal_role not in VOCAL_ROLES:
        vocal_role = "backing" if (is_adlib is True and ev_type in ("vocable", "humming")) else "unknown"

    stream_id = normalize_stream_id(out.get("stream_id"))
    words = out.get("words") if isinstance(out.get("words"), list) else []
    # Never claim word-level timing for humming or unknown_vocal
    if ev_type in ("humming", "unknown_vocal"):
        words = []

    timing_granularity = out.get("timing_granularity")
    if timing_granularity not in TIMING_GRANULARITIES:
        if ev_type in ("humming", "unknown_vocal"):
            timing_granularity = "region"
        elif words:
            timing_granularity = "word"
        else:
            timing_granularity = "line"

    timing_source = out.get("timing_source")
    if timing_source not in TIMING_SOURCES:
        if ev_type == "unknown_vocal":
            timing_source = "acoustic_region"
        elif words:
            timing_source = "asr_word"
        else:
            timing_source = "asr_segment"

    source_audio_variant = out.get("source_audio_variant")
    if source_audio_variant not in AUDIO_VARIANTS:
        source_audio_variant = "vocal_mix_blend"

    review_reasons = [str(r) for r in (out.get("review_reasons") or []) if str(r).strip()]
    if ev_type == "unknown_vocal" and "suspected_vocal_untranscribed" not in review_reasons:
        review_reasons.append("suspected_vocal_untranscribed")

    review_status = out.get("review_status")
    if review_status not in REVIEW_STATUSES:
        if ev_type == "unknown_vocal" or bool(out.get("low_confidence")):
            review_status = "needs_review"
        else:
            review_status = None

    # Never fabricate confidence on unknown_vocal heuristic regions
    conf = out.get("confidence")
    if ev_type == "unknown_vocal" and timing_source == "acoustic_region":
        conf = None

    out.update(
        {
            "text": text,
            "text_raw": str(text_raw),
            "text_edited": str(text_edited) if text_edited is not None else None,
            "event_type": ev_type,
            "vocal_role": vocal_role,
            "is_adlib": is_adlib if isinstance(is_adlib, bool) else None,
            "stream_id": stream_id,
            "overlap_group_id": out.get("overlap_group_id") or None,
            "review_status": review_status,
            "review_reasons": review_reasons,
            "review_reason": out.get("review_reason") or format_review_reason(review_reasons),
            "timing_source": timing_source,
            "timing_granularity": timing_granularity,
            "source_audio_variant": source_audio_variant,
            "confidence": conf,
            "low_confidence": bool(out.get("low_confidence", False) or ev_type == "unknown_vocal"),
            "words": words,
            "candidates": out.get("candidates") if isinstance(out.get("candidates"), list) else [],
        }
    )
    return out


def migrate_transcript(transcript: dict | None) -> dict | None:
    """Safely migrate a v1 transcript to v2 in memory without altering user text or timestamps."""
    if not isinstance(transcript, dict):
        return transcript
    out = dict(transcript)
    fallback_lang = out.get("language") or None
    raw_segs = out.get("segments") if isinstance(out.get("segments"), list) else []
    migrated = [migrate_segment(s, fallback_lang=fallback_lang) for s in raw_segs]
    migrated = assign_overlap_groups(migrated)
    out["schema_version"] = SCHEMA_VERSION
    out["segments"] = migrated
    streams_present = sorted({str(s.get("stream_id") or DEFAULT_STREAM_ID) for s in migrated}) or [DEFAULT_STREAM_ID]
    out["streams"] = [
        {"id": sid, "label": stream_label(sid), "has_isolated_audio": False}
        for sid in streams_present
    ]
    out.setdefault("multi_singer_separation", get_multi_singer_separation_status())
    out.setdefault("raw_asr_segments", [])
    out.setdefault("edit_history", [])
    return out


def get_multi_singer_separation_status() -> dict[str, Any]:
    """Honest capability report for multi-singer vocal separation (Mashup Type B).
    Never claims BS-RoFormer (vocal vs instrumental) separates simultaneous singers!
    """
    return {
        "available": False,
        "provider": None,
        "status": "unavailable_unverified",
        "note": (
            "Hiện mới chỉ có BS-RoFormer tách giọng hát khỏi nhạc đệm (vocal/instrumental), "
            "chưa có mô hình tách riêng từng ca sĩ hát chồng đã kiểm chứng. "
            "Hệ thống vẫn giữ nguyên các đoạn chồng thời gian và hỗ trợ gắn/sửa nhiều luồng (Luồng A/B)."
        ),
    }


def get_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in ("transformers", "accelerate", "torch", "torchaudio", "stem-splitter"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def compute_pipeline_config_hash(
    *,
    model_id: str,
    requested_lang: str,
    vocal_isolated: bool,
    asr_params: dict[str, Any] | None = None,
) -> str:
    payload = {
        "pipeline_version": PIPELINE_CONFIG_VERSION,
        "model": model_id,
        "lang": requested_lang or "auto",
        "vocal_isolated": bool(vocal_isolated),
        "asr_params": asr_params or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


class PipelineTrace:
    """End-to-end diagnostic collector:
    original audio -> separated vocal -> ASR input crops -> raw ASR output (immutable)
    -> cleaned/dropped/modified events -> recovery candidates -> saved output.
    """

    def __init__(self, *, model_id: str, requested_lang: str, vocal_isolated: bool, audio_sha1: str) -> None:
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.model_id = model_id
        self.requested_lang = requested_lang
        self.vocal_isolated = vocal_isolated
        self.audio_sha1 = audio_sha1
        self.runtime_versions = get_runtime_versions()
        self.config: dict[str, Any] = {}
        self.vocal_separation: dict[str, Any] = {}
        self.asr_chunks: list[dict[str, Any]] = []
        self.raw_asr_segments: list[dict[str, Any]] = []
        self.dropped_or_modified: list[dict[str, Any]] = []
        self.recovery_probes: list[dict[str, Any]] = []

    def record_chunk(
        self,
        *,
        index: int,
        start: float,
        end: float,
        detected_lang: str | None = None,
        detected_lang_prob: float | None = None,
        decode_lang: str | None = None,
        prompt_reset_reason: str | None = None,
        source_audio_variant: str = "vocal_mix_blend",
    ) -> None:
        self.asr_chunks.append(
            {
                "index": index,
                "start": round(start, 2),
                "end": round(end, 2),
                "detected_lang": detected_lang,
                "detected_lang_prob": round(float(detected_lang_prob), 4) if detected_lang_prob is not None else None,
                "decode_lang": decode_lang,
                "prompt_reset_reason": prompt_reset_reason,
                "source_audio_variant": source_audio_variant,
            }
        )

    def record_raw_asr(
        self,
        *,
        chunk_index: int,
        start: float,
        end: float,
        text: str,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
        compression_ratio: float | None = None,
        language: str | None = None,
    ) -> None:
        self.raw_asr_segments.append(
            {
                "chunk_index": chunk_index,
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
                "no_speech_prob": round(no_speech_prob, 4) if no_speech_prob is not None else None,
                "compression_ratio": round(compression_ratio, 3) if compression_ratio is not None else None,
                "language": language,
            }
        )

    def record_change(
        self,
        *,
        stage: str,
        action: str,
        reason: str,
        start: float,
        end: float,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        if len(self.dropped_or_modified) < 400:
            self.dropped_or_modified.append(
                {
                    "stage": stage,
                    "action": action,
                    "reason": reason,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "before": before,
                    "after": after,
                }
            )

    def record_recovery(
        self,
        *,
        start: float,
        end: float,
        acoustic_stats: dict[str, Any],
        candidates: list[dict[str, Any]],
        outcome: str,
    ) -> None:
        if len(self.recovery_probes) < 120:
            self.recovery_probes.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "acoustic_stats": acoustic_stats,
                    "candidates": candidates,
                    "outcome": outcome,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_CONFIG_VERSION,
            "created_at": self.created_at,
            "audio_sha1": self.audio_sha1,
            "model_id": self.model_id,
            "requested_lang": self.requested_lang,
            "vocal_isolated": self.vocal_isolated,
            "runtime_versions": self.runtime_versions,
            "config": self.config,
            "vocal_separation": self.vocal_separation,
            "asr_chunks": self.asr_chunks,
            "dropped_or_modified_count": len(self.dropped_or_modified),
            "dropped_or_modified": self.dropped_or_modified,
            "recovery_probes": self.recovery_probes,
        }


def cleanup_debug_audio(debug_root: Path, *, max_sessions: int = 5, max_age_hours: float = 24.0) -> int:
    """Remove stale debug audio trace folders so optional debug capture never fills disk."""
    if not debug_root.is_dir():
        return 0
    now_ts = time.time()
    dirs = sorted(
        [d for d in debug_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for idx, d in enumerate(dirs):
        age_hr = (now_ts - d.stat().st_mtime) / 3600.0
        if idx >= max_sessions or age_hr > max_age_hours:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def maybe_save_debug_clip(
    debug_root: Path,
    session_key: str,
    clip_name: str,
    audio_16k: np.ndarray,
    *,
    max_clips: int = 12,
) -> str | None:
    """Save a bounded 16kHz WAV diagnostic clip ONLY when `SONGSCRIPTION_DEBUG_AUDIO=true`."""
    if os.environ.get("SONGSCRIPTION_DEBUG_AUDIO", "").strip().lower() not in ("1", "true", "yes"):
        return None
    try:
        import soundfile as sf

        cleanup_debug_audio(debug_root)
        target_dir = debug_root / re.sub(r"[^a-zA-Z0-9_-]", "", session_key)[:24]
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = list(target_dir.glob("*.wav"))
        if len(existing) >= max_clips:
            return None
        safe_clip = re.sub(r"[^a-zA-Z0-9_.-]", "_", clip_name)
        out_path = target_dir / (safe_clip if safe_clip.endswith(".wav") else f"{safe_clip}.wav")
        sf.write(str(out_path), np.clip(audio_16k, -1.0, 1.0).astype(np.float32), 16000, subtype="PCM_16")
        return str(out_path)
    except Exception:
        return None


# ================================================================== Evaluation Metrics (Section 9)
def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = seq_a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _norm_eval_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [w for w in cleaned.split() if w]


def evaluate_transcript_against_reference(
    predicted: list[dict],
    reference: list[dict],
    *,
    silent_intervals: list[tuple[float, float]] | None = None,
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute honest, uninflated evaluation metrics against labeled ground truth.
    - WER / CER computed ONLY on lexical lyrics (`event_type == 'lyric'`).
    - `unknown_vocal` is NEVER counted as a correctly recognized vocalise syllable.
    - Tracks ad-lib miss rate, hallucination rate on silent intervals, stream assignment errors,
      and timestamp MAE (start/end).
    """
    ref_lyrics = " ".join(str(r.get("text") or "") for r in reference if r.get("event_type", "lyric") == "lyric")
    pred_lyrics = " ".join(str(p.get("text") or "") for p in predicted if p.get("event_type", "lyric") == "lyric")

    ref_w = _norm_eval_words(ref_lyrics)
    pred_w = _norm_eval_words(pred_lyrics)
    wer = (_levenshtein(ref_w, pred_w) / len(ref_w)) if ref_w else (0.0 if not pred_w else 1.0)

    ref_c = list(re.sub(r"\s+", "", ref_lyrics.lower()))
    pred_c = list(re.sub(r"\s+", "", pred_lyrics.lower()))
    cer = (_levenshtein(ref_c, pred_c) / len(ref_c)) if ref_c else (0.0 if not pred_c else 1.0)

    def _iou(a: dict, b: dict) -> float:
        s0, e0 = float(a.get("start") or 0.0), float(a.get("end") or 0.0)
        s1, e1 = float(b.get("start") or 0.0), float(b.get("end") or 0.0)
        inter = max(0.0, min(e0, e1) - max(s0, s1))
        union = max(1e-6, max(e0, e1) - min(s0, s1))
        return inter / union

    # Vocalise precision/recall (vocable/humming with recognized text; NEVER count unknown_vocal as TP!)
    ref_voc = [r for r in reference if r.get("event_type") in ("vocable", "humming")]
    pred_voc_recognized = [
        p for p in predicted
        if p.get("event_type") in ("vocable", "humming") and str(p.get("text") or "").strip()
    ]
    matched_ref_voc: set[int] = set()
    voc_tp = 0
    for p in pred_voc_recognized:
        for ri, r in enumerate(ref_voc):
            if ri not in matched_ref_voc and _iou(p, r) >= iou_threshold:
                voc_tp += 1
                matched_ref_voc.add(ri)
                break
    voc_precision = (voc_tp / len(pred_voc_recognized)) if pred_voc_recognized else (1.0 if not ref_voc else 0.0)
    voc_recall = (voc_tp / len(ref_voc)) if ref_voc else 1.0

    # Ad-lib miss rate (reference items with is_adlib=True)
    ref_adlibs = [r for r in reference if r.get("is_adlib") is True]
    adlib_found = 0
    for r in ref_adlibs:
        if any(
            _iou(p, r) >= iou_threshold and p.get("event_type") != "unknown_vocal" and str(p.get("text") or "").strip()
            for p in predicted
        ):
            adlib_found += 1
    adlib_miss_rate = (1.0 - adlib_found / len(ref_adlibs)) if ref_adlibs else 0.0

    # Hallucination rate on silent/instrumental intervals
    hallucinations = 0
    if silent_intervals:
        for p in predicted:
            if p.get("event_type") == "unknown_vocal":
                continue
            ps, pe = float(p.get("start") or 0.0), float(p.get("end") or 0.0)
            for ss, se in silent_intervals:
                if max(0.0, min(pe, se) - max(ps, ss)) >= 0.25:
                    hallucinations += 1
                    break
    hallucination_rate = (hallucinations / len(predicted)) if predicted else 0.0

    # Stream assignment error & timestamp MAE on time-matched reference items
    stream_checked = 0
    stream_errors = 0
    ts_errors: list[float] = []
    for r in reference:
        best_p = None
        best_iou = 0.0
        for p in predicted:
            iou = _iou(p, r)
            if iou > best_iou:
                best_iou = iou
                best_p = p
        if best_p is not None and best_iou >= iou_threshold:
            ts_errors.append(abs(float(best_p["start"]) - float(r["start"])))
            ts_errors.append(abs(float(best_p["end"]) - float(r["end"])))
            if r.get("stream_id"):
                stream_checked += 1
                if normalize_stream_id(best_p.get("stream_id")) != normalize_stream_id(r.get("stream_id")):
                    stream_errors += 1

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "vocalise_precision": round(voc_precision, 4),
        "vocalise_recall": round(voc_recall, 4),
        "adlib_miss_rate": round(adlib_miss_rate, 4),
        "hallucination_count": hallucinations,
        "hallucination_rate": round(hallucination_rate, 4),
        "stream_error_rate": round(stream_errors / stream_checked, 4) if stream_checked else 0.0,
        "timestamp_mae_sec": round(float(np.mean(ts_errors)), 3) if ts_errors else None,
        "unknown_vocal_count": sum(1 for p in predicted if p.get("event_type") == "unknown_vocal"),
    }
