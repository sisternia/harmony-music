"""Songscription recognition backends and shared lyric segmentation/alignment.

Qwen3-ASR recognizes singing by default, then uses Whisper to align that text.
Whisper remains available as an alternative recognition backend.
Neither backend guarantees recognition of every vocal syllable.
"""

from __future__ import annotations

import gc
import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .engine import Cancelled
from .lyrics_schema import (
    DEFAULT_STREAM_ID,
    PipelineTrace,
    assign_overlap_groups,
    classify_vocal_event,
    estimate_vocal_harmonicity,
    format_review_reason,
    map_crop_time_to_source,
    normalize_vocable_elongation,
)

WHISPER_LARGE_V3_REPO = "openai/whisper-large-v3"
QWEN3_ASR_ID = "qwen3-asr-1.7b"
QWEN3_ASR_REPO = "Qwen/Qwen3-ASR-1.7B"

MODEL_REPO_BY_ID: dict[str, str] = {
    "whisper-large-v3": WHISPER_LARGE_V3_REPO,
    QWEN3_ASR_ID: QWEN3_ASR_REPO,
}

SONGSCRIPTION_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": QWEN3_ASR_ID,
        "label": "Qwen3-ASR 1.7B + Whisper Align (Khuyên dùng · Nhận dạng giọng hát trên GPU)",
    },
    {
        "id": "whisper-large-v3",
        "label": "Whisper Large v3 (OpenAI · Chạy thuần GPU không cần API)",
    },
)

DEFAULT_MODEL_ID = QWEN3_ASR_ID

ISO_TO_WHISPER_LANG: dict[str, str] = {
    "en": "english",
    "zh": "chinese",
    "yue": "cantonese",
    "ja": "japanese",
    "ko": "korean",
    "vi": "vietnamese",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "th": "thai",
    "id": "indonesian",
}

WHISPER_LANG_TO_ISO: dict[str, str] = {v: k for k, v in ISO_TO_WHISPER_LANG.items()}
WHISPER_LANG_TO_ISO["mandarin"] = "zh"

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\.,!?;:'\"\-~…♪♫()\[\]【】「」『』、。！？・]+$")
_TRAIL_PUNCT_CHARS = ".,!?;:…、。！？・"
_VOCAL_TOKEN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]|[^\W_]+", re.UNICODE)


def _vocalise_tokens(text: str) -> list[str]:
    """Return vocalise/scat tokens with their occurrence count intact."""
    return [normalize_vocable_elongation(m.group()).casefold() for m in _VOCAL_TOKEN_RE.finditer(text or "")]


def _is_vocalise_text(text: str) -> bool:
    event_type, is_adlib, _ = classify_vocal_event(text)
    return event_type in ("vocable", "humming") or is_adlib is True


def _vocalise_loses_occurrences(source: str, candidate: str) -> bool:
    """Detect a candidate that dropped one or more sung vocalise occurrences.

    Forced alignment and short re-transcription can return only the first ``Ah``
    from ``Ah ah``.  Never let that shorter candidate replace the recognized
    source.  This compares a multiset rather than a set so repeated syllables
    remain meaningful.
    """
    if not _is_vocalise_text(source):
        return False
    source_tokens = _vocalise_tokens(source)
    candidate_tokens = _vocalise_tokens(candidate)
    if not source_tokens:
        return False
    return len(candidate_tokens) < len(source_tokens)


def _vocalise_adds_occurrences(source: str, candidate: str) -> bool:
    """Return true only when candidate adds occurrences of the same syllables."""
    if not (_is_vocalise_text(source) and _is_vocalise_text(candidate)):
        return False
    source_tokens, candidate_tokens = _vocalise_tokens(source), _vocalise_tokens(candidate)
    if len(candidate_tokens) <= len(source_tokens):
        return False
    source_counts, candidate_counts = Counter(source_tokens), Counter(candidate_tokens)
    return all(candidate_counts[token] >= count for token, count in source_counts.items())


def _alignment_covers_text(words: list[dict[str, Any]], text: str) -> bool:
    """Alignment may omit tokens; compare ordered characters including repeats."""
    normalize = lambda value: re.sub(r"[\W_]+", "", value.casefold())
    return normalize("".join(str(w.get("word") or "") for w in words)) == normalize(text)


# ================================================================== Registry & Language Helpers
def normalize_model_id(raw: str | None) -> str:
    """Keep supported model IDs; map unknown or legacy IDs to the existing default."""
    val = str(raw or "").strip().lower()
    if val in MODEL_REPO_BY_ID:
        return val
    return DEFAULT_MODEL_ID


def iso_to_whisper_lang(iso_code: str | None) -> str | None:
    if not iso_code or iso_code in ("auto", ""):
        return None
    low = iso_code.strip().lower()
    if low in ISO_TO_WHISPER_LANG:
        return ISO_TO_WHISPER_LANG[low]
    if low in WHISPER_LANG_TO_ISO:
        return low
    return low if len(low) >= 2 else None


def whisper_lang_to_iso(whisper_lang: str | None, fallback: str = "en") -> str:
    if not whisper_lang:
        return fallback
    low = str(whisper_lang).strip().lower().strip("<|>")
    if low in ISO_TO_WHISPER_LANG:
        return low
    if low in WHISPER_LANG_TO_ISO:
        return WHISPER_LANG_TO_ISO[low]
    return low if len(low) == 2 and low.isalpha() else fallback


# ================================================================== Word Formatting, DTW Alignment & Line Grouping
def _format_whisper_chunks(
    raw_chunks: Any,
    *,
    offset_sec: float,
    max_duration: float,
    chunk_duration: float,
    timing_source: str = "asr_word",
) -> list[dict[str, Any]]:
    """Convert Hugging Face Whisper `chunks` (`[{"text": str, "timestamp": (s, e)}]`) to Harmony Auralis word tokens."""
    if not raw_chunks or not isinstance(raw_chunks, (list, tuple)):
        return []

    words: list[dict[str, Any]] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        raw_txt = str(item.get("text") or "")
        txt = raw_txt.strip()
        if not txt:
            continue
        ts = item.get("timestamp")
        if not isinstance(ts, (list, tuple)) or len(ts) < 2:
            continue
        s_raw, e_raw = ts[0], ts[1]
        if s_raw is None:
            continue
        if e_raw is None:
            e_raw = min(chunk_duration, float(s_raw) + 0.35)
        try:
            ws_f, we_f = map_crop_time_to_source(float(s_raw), float(e_raw), offset_sec, max_duration)
        except (TypeError, ValueError):
            continue
        display = raw_txt if (words and raw_txt.startswith(" ")) else txt
        if words and not display.startswith(" ") and not _CJK_RE.search(txt) and not _CJK_RE.search(words[-1]["word"]):
            if not re.match(r"^(?:[,.;:!?)\]}]|['’](?:s|t|re|ve|ll|m|d)\b|n['’]t\b)", txt, re.IGNORECASE):
                display = " " + txt
        words.append(
            {
                "word": display,
                "start": round(ws_f, 2),
                "end": round(max(ws_f + 0.02, we_f), 2),
                "probability": None,
                "uncertain": False,
                "timing_source": timing_source,
            }
        )
    return words


def _dtw_path(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard monotonic DTW backtrace on (N_tokens, N_frames) negative similarity matrix."""
    n, m = cost_matrix.shape
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0

    for j in range(1, m + 1):
        cost[0, j] = 0.0  # allow leading silence before token 0

    for i in range(1, n + 1):
        row = cost_matrix[i - 1]
        for j in range(1, m + 1):
            c_diag = cost[i - 1, j - 1]
            c_up = cost[i - 1, j]
            c_left = cost[i, j - 1]
            if c_diag <= c_up and c_diag <= c_left:
                best, tr = c_diag, 0
            elif c_up <= c_left:
                best, tr = c_up, 1
            else:
                best, tr = c_left, 2
            cost[i, j] = float(row[j - 1]) + best
            trace[i, j] = tr

    # Pick optimal ending frame (allow trailing silence after last token)
    j = int(np.argmin(cost[n, 1:])) + 1 if m > 0 else 0
    i = n
    text_indices: list[int] = []
    time_indices: list[int] = []
    while i > 0 and j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        tr = int(trace[i, j])
        if tr == 0:
            i -= 1
            j -= 1
        elif tr == 1:
            i -= 1
        else:
            j -= 1

    text_indices.reverse()
    time_indices.reverse()
    return np.asarray(text_indices, dtype=np.int32), np.asarray(time_indices, dtype=np.int32)


def _split_tokens_to_words(
    tokenizer: Any,
    token_ids: list[int],
    token_starts: list[float],
    token_ends: list[float],
) -> list[dict[str, Any]]:
    """Group Whisper BPE subword tokens into display words while preserving exact punctuation."""
    if not token_ids:
        return []

    words: list[dict[str, Any]] = []
    cur_ids: list[int] = []
    cur_s: float = token_starts[0]
    cur_e: float = token_ends[0]

    def _flush() -> None:
        if not cur_ids:
            return
        piece = tokenizer.decode(cur_ids, skip_special_tokens=True)
        if not piece or not piece.strip():
            return
        words.append(
            {
                "word": piece,
                "start": round(cur_s, 2),
                "end": round(max(cur_s + 0.04, cur_e), 2),
                "probability": None,
                "uncertain": False,
                "timing_source": "forced_aligner",
            }
        )

    for idx, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=True)
        if not tok_str:
            continue
        is_cjk_tok = bool(_CJK_RE.search(tok_str))
        starts_new_word = bool(
            cur_ids
            and (
                tok_str.startswith((" ", "\u2581"))
                or (is_cjk_tok and tok_str.strip() and not _PUNCT_ONLY_RE.match(tok_str))
            )
        )
        if starts_new_word:
            _flush()
            cur_ids = [tid]
            cur_s = token_starts[idx]
            cur_e = token_ends[idx]
        else:
            if not cur_ids:
                cur_s = token_starts[idx]
            cur_ids.append(tid)
            cur_e = max(cur_e, token_ends[idx])
    _flush()

    if words:
        words[0]["word"] = str(words[0]["word"]).lstrip()
    return words


def _word_syllable_weight(word_text: str) -> float:
    clean = re.sub(r"[\s.,!?;:'\"\-~…♪♫()\[\]【】「」『』、。！？・]+", "", word_text)
    if not clean:
        return 1.0
    if _CJK_RE.search(clean):
        return float(max(1, len(clean)))
    return max(1.0, len(clean) / 3.0)


def _find_vocal_offset_after(
    rms_env: np.ndarray | None,
    vocal_floor: float,
    start_sec: float,
    max_sec: float,
) -> float:
    if rms_env is None or rms_env.size == 0 or max_sec <= start_sec + 0.05:
        return max_sec
    thresh = max(vocal_floor * 2.0, 0.015)
    n_f = rms_env.size
    i0 = max(0, min(n_f - 1, int(math.floor(start_sec * 20.0))))
    i1 = max(i0 + 1, min(n_f, int(math.ceil(max_sec * 20.0))))
    last_active, silent_run = i0, 0
    for k in range(i0, i1):
        if float(rms_env[k]) >= thresh:
            last_active, silent_run = k, 0
        else:
            silent_run += 1
            if silent_run >= 4 and last_active > i0:
                break
    return min(max_sec, round((last_active + 1) * 0.05, 2))


def _repair_aligned_words(
    words: list[dict[str, Any]],
    rms_env: np.ndarray | None,
    vocal_floor: float,
    chunk_end: float,
) -> list[dict[str, Any]]:
    """Repair collapsed <=65ms / duplicate-start / stalled word tokens using real 50ms vocal RMS
    energy + character-length weighting while leaving clean word timestamps intact."""
    if not words:
        return []
    repaired = [dict(w) for w in words]
    n = len(repaired)

    def _is_collapsed(idx: int) -> bool:
        s, e = float(repaired[idx]["start"]), float(repaired[idx]["end"])
        if (e - s) <= 0.065:
            return True
        if idx > 0 and s <= float(repaired[idx - 1]["start"]) + 0.025:
            return True
        if idx + 1 < n and float(repaired[idx + 1]["start"]) <= s + 0.025:
            return True
        return False

    i = 0
    while i < n:
        if not _is_collapsed(i):
            i += 1
            continue
        j = i
        while j + 1 < n and _is_collapsed(j + 1):
            j += 1

        span_a, span_b = i, j
        span_start = float(repaired[span_a]["start"])
        span_end = max(float(repaired[k]["end"]) for k in range(span_a, span_b + 1))

        if span_a > 0:
            prev_w = repaired[span_a - 1]
            prev_s, prev_e = float(prev_w["start"]), float(prev_w["end"])
            prev_expected = max(0.55, 0.34 * _word_syllable_weight(str(prev_w["word"])))
            gap_from_prev = span_start - prev_e
            if (prev_e - prev_s) > prev_expected * 1.45 and gap_from_prev <= 0.16:
                span_a -= 1
                span_start = prev_s
            elif gap_from_prev >= 0.32 and (span_b - span_a >= 1 or (span_end - span_start) <= 0.08):
                if rms_env is not None and rms_env.size > 0:
                    f0 = max(0, min(rms_env.size - 1, int(math.floor((prev_e + 0.05) * 20.0))))
                    f1 = max(f0 + 1, min(rms_env.size, int(math.ceil(span_start * 20.0))))
                    seg_rms = rms_env[f0:f1]
                    thresh = max(vocal_floor * 2.0, 0.015)
                    if seg_rms.size > 0 and float(np.mean(seg_rms >= thresh)) >= 0.40:
                        active_pos = np.where(seg_rms >= thresh)[0]
                        if active_pos.size > 0:
                            span_start = max(prev_e + 0.04, round((f0 + int(active_pos[0])) * 0.05, 2))
                else:
                    span_start = round(prev_e + 0.05, 2)

        next_limit = float(repaired[span_b + 1]["start"]) - 0.02 if (span_b + 1 < n) else chunk_end
        count_in_span = span_b - span_a + 1
        min_natural = sum(max(0.14, 0.16 * _word_syllable_weight(str(repaired[k]["word"]))) for k in range(span_a, span_b + 1))
        if (span_end - span_start) < min_natural and next_limit > span_end + 0.08:
            cap_end = min(next_limit, span_start + max(min_natural * 1.8, 0.75 * count_in_span))
            span_end = max(span_end, _find_vocal_offset_after(rms_env, vocal_floor, span_end, cap_end))

        while span_a > 0 and (span_end - span_start) < 0.13 * (span_b - span_a + 1):
            if span_start - float(repaired[span_a - 1]["end"]) > 0.25:
                break
            span_a -= 1
            span_start = float(repaired[span_a]["start"])
            count_in_span = span_b - span_a + 1

        total_dur = max(0.06 * count_in_span, span_end - span_start)
        weights = [_word_syllable_weight(str(repaired[k]["word"])) for k in range(span_a, span_b + 1)]
        w_sum = max(1e-6, sum(weights))
        cur = span_start
        for offset_k, k in enumerate(range(span_a, span_b + 1)):
            dur_k = total_dur * (weights[offset_k] / w_sum)
            nxt_t = span_start + total_dur if offset_k == count_in_span - 1 else (cur + dur_k)
            repaired[k]["start"] = round(cur, 2)
            repaired[k]["end"] = round(max(cur + 0.04, nxt_t), 2)
            cur = nxt_t
        i = span_b + 1

    for k in range(1, n):
        prev_s, prev_e = float(repaired[k - 1]["start"]), float(repaired[k - 1]["end"])
        cur_s, cur_e = float(repaired[k]["start"]), float(repaired[k]["end"])
        if cur_s < prev_e:
            if cur_e > prev_e + 0.04:
                cur_s = prev_e
            else:
                mid = round(max(prev_s + 0.04, (prev_s + max(prev_e, cur_e)) * 0.5), 2)
                repaired[k - 1]["end"] = mid
                cur_s = mid
            repaired[k]["start"] = round(cur_s, 2)
            repaired[k]["end"] = round(max(cur_s + 0.04, cur_e), 2)
    return repaired


def _has_collapsed_words(words: list[dict[str, Any]]) -> bool:
    collapsed = 0
    for idx, w in enumerate(words):
        s, e = float(w["start"]), float(w["end"])
        if (e - s) <= 0.065 or (idx > 0 and s <= float(words[idx - 1]["start"]) + 0.025):
            collapsed += 1
            if collapsed >= 2:
                return True
    return False


def _enforce_non_overlapping_same_stream(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair conflicting main-lyric timings without erasing explicit backing/ad-lib events."""
    if len(segments) <= 1:
        return list(segments)

    adlibs = [
        dict(seg) for seg in segments
        if seg.get("is_adlib") is True or seg.get("event_type") in ("vocable", "humming")
        or seg.get("vocal_role") == "backing"
    ]
    if adlibs:
        main = [
            seg for seg in segments
            if not (seg.get("is_adlib") is True or seg.get("event_type") in ("vocable", "humming")
                    or seg.get("vocal_role") == "backing")
        ]
        preserved = _enforce_non_overlapping_same_stream(main)
        for seg in adlibs:
            # Keep every emitted occurrence, even when coarse timings are identical.
            # Separate overlapping event lanes so playback can highlight every row.
            used = {
                str(other.get("stream_id") or DEFAULT_STREAM_ID)
                for other in preserved
                if min(float(seg["end"]), float(other["end"])) > max(float(seg["start"]), float(other["start"]))
            }
            stream_id = str(seg.get("stream_id") or DEFAULT_STREAM_ID)
            lane = 2
            while stream_id in used:
                stream_id = f"stream_{chr(96 + lane)}" if lane <= 26 else f"stream_{lane}"
                lane += 1
            seg["stream_id"] = stream_id
            preserved.append(seg)
        return assign_overlap_groups(sorted(preserved, key=lambda s: (float(s["start"]), float(s["end"]))))

    by_stream: dict[str, list[dict[str, Any]]] = {}
    for seg in segments:
        sid = str(seg.get("stream_id") or DEFAULT_STREAM_ID)
        by_stream.setdefault(sid, []).append(dict(seg))

    merged_out: list[dict[str, Any]] = []
    for items in by_stream.values():
        items.sort(key=lambda s: (float(s["start"]), -float(s["end"])))
        clean_stream: list[dict[str, Any]] = []
        for cur in items:
            cs, ce = float(cur["start"]), float(cur["end"])
            if not clean_stream:
                clean_stream.append(cur)
                continue
            prev = clean_stream[-1]
            ps, pe = float(prev["start"]), float(prev["end"])
            if cs >= ps - 0.05 and ce <= pe + 0.05:
                continue
            if cs < pe:
                if cs <= ps + 0.12:
                    if (ce - cs) > (pe - ps) and len(str(cur.get("text") or "")) >= len(str(prev.get("text") or "")):
                        clean_stream[-1] = cur
                    continue
                new_pe = round(cs, 2)
                prev["end"] = new_pe
                pw = prev.get("words")
                if isinstance(pw, list) and pw and float(pw[-1]["end"]) > new_pe:
                    first_ws = min(ps, float(pw[0]["start"]))
                    scale = max(0.05, new_pe - first_ws) / max(0.05, float(pw[-1]["end"]) - first_ws)
                    scaled_w: list[dict[str, Any]] = []
                    for w in pw:
                        wc = dict(w)
                        ws_n = min(new_pe, round(first_ws + (float(w["start"]) - first_ws) * scale, 2))
                        we_n = min(new_pe, max(ws_n + 0.02, round(first_ws + (float(w["end"]) - first_ws) * scale, 2)))
                        wc["start"], wc["end"] = ws_n, we_n
                        scaled_w.append(wc)
                    scaled_w[-1]["end"] = new_pe
                    prev["words"] = scaled_w
            clean_stream.append(cur)
        merged_out.extend(clean_stream)

    merged_out.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return merged_out


def _group_words_into_lyric_lines(
    words: list[dict[str, Any]],
    fallback_text: str,
    chunk_start: float,
    chunk_end: float,
    *,
    is_vocalise_chunk: bool = False,
) -> list[tuple[str, float, float, list[dict[str, Any]]]]:
    """Segment word-timed tokens into crisp 1-sentence lyric lines using universal signals
    (punctuation, newlines, breath pauses) while keeping separate ad-lib/vocalise phrases apart."""
    if not words or not _alignment_covers_text(words, fallback_text):
        clean = fallback_text.strip()
        return [(clean, round(chunk_start, 2), round(chunk_end, 2), [])] if clean else []

    split_idxs: list[int] = []
    last_split = 0
    n = len(words)
    is_cjk = bool(_CJK_RE.search(fallback_text))
    max_words_cap = 18 if is_cjk else 13
    max_dur_cap = 5.0

    def _pick_best_interior_split(lo_idx: int, hi_idx: int) -> int:
        best_pos, best_score = hi_idx, -1e9
        for k in range(lo_idx, hi_idx + 1):
            w_l, w_r = words[k], words[k + 1]
            lt = str(w_l["word"]).strip()
            score = max(0.0, float(w_r["start"]) - float(w_l["end"])) * 4.5
            if lt.endswith((",", ";", "、")):
                score += 3.0
            frac = (k + 1 - last_split) / max(1, (hi_idx + 2 - last_split))
            score -= abs(frac - 0.55) * 0.8
            if score > best_score:
                best_score, best_pos = score, k + 1
        return best_pos

    i = 0
    while i < n - 1:
        w_cur, w_nxt = words[i], words[i + 1]
        raw_cur = str(w_cur["word"])
        cur_txt = raw_cur.strip()
        gap = max(0.0, float(w_nxt["start"]) - float(w_cur["end"]))
        line_dur = float(w_cur["end"]) - float(words[last_split]["start"])
        left_cnt, right_cnt = (i + 1) - last_split, n - (i + 1)
        hard_punct = cur_txt.endswith((".", "!", "?", "。", "！", "？", "…"))
        newline_after = bool(w_cur.get("_newline_after"))

        if is_vocalise_chunk:
            if (
                (newline_after and left_cnt >= 1 and right_cnt >= 1)
                or (hard_punct and left_cnt >= 1 and right_cnt >= 1 and gap >= 0.15)
                or (gap >= 0.42 and line_dur >= 0.45)
                or (line_dur >= 3.6 and gap >= 0.28 and left_cnt >= 2)
            ):
                split_idxs.append(i + 1)
                last_split = i + 1
            i += 1
            continue

        soft_punct = cur_txt.endswith((",", ";", "、"))
        cjk_space = is_cjk and raw_cur.endswith((" ", "　"))

        should_split = False
        if (newline_after or hard_punct) and left_cnt >= 2 and right_cnt >= 1:
            should_split = True
        elif (newline_after or hard_punct) and left_cnt == 1 and right_cnt >= 2 and (gap >= 0.05 or line_dur >= 0.35):
            should_split = True
        elif gap >= (0.30 if is_cjk else 0.34) and left_cnt >= 2 and line_dur >= 0.85:
            should_split = True
        elif cjk_space and left_cnt >= 3 and right_cnt >= 2 and (gap >= 0.08 or line_dur >= 1.8):
            should_split = True
        elif soft_punct and left_cnt >= (4 if is_cjk else 5) and right_cnt >= 3 and (gap >= 0.12 or line_dur >= 2.0):
            should_split = True
        elif right_cnt >= 2 and (left_cnt >= max_words_cap or (line_dur >= max_dur_cap and left_cnt >= (6 if is_cjk else 7))):
            best_pos = _pick_best_interior_split(min(i, last_split + 2), i)
            if best_pos > last_split:
                split_idxs.append(best_pos)
                last_split = best_pos
                i = best_pos
                continue

        if should_split:
            split_idxs.append(i + 1)
            last_split = i + 1
        i += 1

    bounds = [0] + split_idxs + [n]
    lines: list[tuple[str, float, float, list[dict[str, Any]]]] = []
    for b in range(len(bounds) - 1):
        sub_w = [{k: v for k, v in w.items() if k != "_newline_after"} for w in words[bounds[b] : bounds[b + 1]]]
        if not sub_w:
            continue
        sub_w[0]["word"] = str(sub_w[0]["word"]).lstrip()
        sub_w[-1]["word"] = re.sub(r"[,;、]\s*$", "", str(sub_w[-1]["word"]))
        if b > 0 and len(sub_w[0]["word"]) >= 1 and sub_w[0]["word"][0].islower() and sub_w[0]["word"][0].isalpha():
            sub_w[0]["word"] = sub_w[0]["word"][0].upper() + sub_w[0]["word"][1:]
        txt = re.sub(r"\s+", " ", "".join(str(w["word"]) for w in sub_w)).strip()
        txt = re.sub(r"[,;、]\s*$", "", txt).strip()
        if not txt or _PUNCT_ONLY_RE.match(txt):
            continue
        ls = round(float(sub_w[0]["start"]), 2)
        le = round(max(ls + 0.10, float(sub_w[-1]["end"])), 2)
        lines.append((txt, ls, le, sub_w))

    if lines:
        return lines
    clean_words = [{k: v for k, v in w.items() if k != "_newline_after"} for w in words]
    return [(fallback_text.strip(), round(chunk_start, 2), round(chunk_end, 2), clean_words)]


def _split_plain_text_into_lines(
    text: str,
    s_time: float,
    e_time: float,
) -> list[tuple[str, float, float, list[dict[str, Any]]]]:
    raw = text.strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？\n])\s+", raw) if p.strip()]
    if len(parts) <= 1 or (e_time - s_time) < 3.0:
        return [(raw, round(s_time, 2), round(e_time, 2), [])]
    total_chars = sum(max(1, len(p)) for p in parts)
    out: list[tuple[str, float, float, list[dict[str, Any]]]] = []
    cur, dur = s_time, max(0.2, e_time - s_time)
    for idx, part in enumerate(parts):
        nxt = e_time if idx == len(parts) - 1 else min(e_time, cur + dur * (max(1, len(part)) / total_chars))
        out.append((part, round(cur, 2), round(max(cur + 0.15, nxt), 2), []))
        cur = nxt
    return out


# ================================================================== Official OpenAI Whisper Large v3 Engine
class WhisperLargeV3Engine:
    """Official Hugging Face Transformers wrapper for `openai/whisper-large-v3`
    (https://huggingface.co/openai/whisper-large-v3) with native word-level timestamps
    and cross-attention DTW forced alignment."""

    def __init__(
        self,
        *,
        model_id: str = "whisper-large-v3",
        log: Callable[[str], None] = lambda _m: None,
    ) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.model_id = normalize_model_id(model_id)
        if self.model_id == QWEN3_ASR_ID:
            raise ValueError("Use Qwen3ASREngine for Qwen recognition, not the Whisper wrapper.")
        self.repo_id = MODEL_REPO_BY_ID[self.model_id]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        log(f"Đang nạp {self.repo_id} ({self.device}, {str(self.dtype).split('.')[-1]})...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.dtype,
            device=self.device,
        )

    def detect_language(self, audio_16k: np.ndarray) -> str:
        """Detect song language directly using `openai/whisper-large-v3`'s language head."""
        if self.model is None or self.processor is None or audio_16k.size < 1600:
            return "en"
        try:
            import torch

            sample = audio_16k[: 30 * 16000].astype(np.float32, copy=False)
            inputs = self.processor(sample, sampling_rate=16000, return_tensors="pt")
            feats = inputs.input_features.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                lang_ids = self.model.detect_language(feats)
            lang_tok = self.processor.tokenizer.decode([int(lang_ids[0].item())], skip_special_tokens=False)
            return whisper_lang_to_iso(lang_tok, fallback="en")
        except Exception:
            return "en"

    def transcribe_chunk(
        self,
        chunk_16k: np.ndarray,
        *,
        offset_sec: float,
        total_duration: float,
        language_iso: str | None = None,
        want_timestamps: bool = True,
    ) -> dict[str, Any]:
        if self.pipe is None or chunk_16k.size < 1600:
            return {"text": "", "language": language_iso or "en", "words": [], "timing_source": "asr_segment"}

        arr = chunk_16k.astype(np.float32, copy=False)
        chunk_dur = float(arr.shape[0]) / 16000.0
        det_iso = language_iso or self.detect_language(arr)
        w_lang = iso_to_whisper_lang(det_iso)

        gen_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "temperature": (0.0, 0.2, 0.4),
        }
        if w_lang:
            gen_kwargs["language"] = w_lang

        audio_input = {"array": arr, "sampling_rate": 16000}
        try:
            res = self.pipe(
                audio_input,
                return_timestamps="word" if want_timestamps else True,
                generate_kwargs=gen_kwargs,
            )
        except Exception:
            # Fallback to segment-level timestamps if cross-attention DTW fails on an unusual clip
            res = self.pipe(
                audio_input,
                return_timestamps=True,
                generate_kwargs={k: v for k, v in gen_kwargs.items() if k != "temperature"},
            )

        raw_text = str((res or {}).get("text") or "").strip()
        raw_chunks = (res or {}).get("chunks") or []
        words = (
            _format_whisper_chunks(
                raw_chunks,
                offset_sec=offset_sec,
                max_duration=total_duration,
                chunk_duration=chunk_dur,
                timing_source="asr_word",
            )
            if want_timestamps
            else []
        )
        return {
            "text": raw_text,
            "language": det_iso,
            "language_raw": w_lang or det_iso,
            "words": words,
            "timing_source": "asr_word" if words else "asr_segment",
        }

    def force_align_chunk(
        self,
        chunk_16k: np.ndarray,
        text: str,
        *,
        offset_sec: float,
        total_duration: float,
        language_iso: str | None,
    ) -> list[dict[str, Any]]:
        """Word-level forced alignment directly via `openai/whisper-large-v3` cross-attention DTW
        (supports all 99 Whisper languages including English, Japanese, Chinese, Korean, Vietnamese)."""
        clean_text = text.strip()
        if self.model is None or self.processor is None or not clean_text or chunk_16k.size < 1600:
            return []

        try:
            import torch
            from scipy.ndimage import median_filter

            arr = chunk_16k[: 30 * 16000].astype(np.float32, copy=False)
            chunk_dur = float(arr.shape[0]) / 16000.0
            n_frames = max(2, min(1500, int(math.ceil(chunk_dur * 50.0))))

            tokenizer = self.processor.tokenizer
            w_lang = iso_to_whisper_lang(language_iso) or "english"
            prefix_ids = [
                int(tid)
                for _pos, tid in tokenizer.get_decoder_prompt_ids(language=w_lang, task="transcribe", no_timestamps=True)
            ]
            sot_id = int(tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
            prefix = [sot_id] + prefix_ids
            text_ids = tokenizer.encode(" " + clean_text, add_special_tokens=False)
            if not text_ids:
                return []
            eot_id = int(tokenizer.eos_token_id)
            dec_ids = torch.tensor([prefix + text_ids + [eot_id]], dtype=torch.long, device=self.device)

            inputs = self.processor(arr, sampling_rate=16000, return_tensors="pt")
            feats = inputs.input_features.to(self.device, dtype=self.dtype)

            with torch.no_grad():
                outputs = self.model(
                    input_features=feats,
                    decoder_input_ids=dec_ids,
                    output_attentions=True,
                    return_dict=True,
                )

            cross_attns = outputs.cross_attentions
            if not cross_attns:
                return []

            align_heads = getattr(self.model.generation_config, "alignment_heads", None)
            if not align_heads:
                # Fallback to last 6 decoder layers if alignment_heads is absent
                n_layers = len(cross_attns)
                n_heads = int(cross_attns[0].shape[1])
                align_heads = [[l_i, h_i] for l_i in range(max(0, n_layers - 6), n_layers) for h_i in range(n_heads)]

            # Collect selected cross-attention heads: shape -> (H_sel, T_text, n_frames)
            t_start = len(prefix)
            t_end = t_start + len(text_ids)
            selected = [
                cross_attns[int(layer)][0, int(head), t_start:t_end, :n_frames].float()
                for layer, head in align_heads
                if int(layer) < len(cross_attns)
            ]
            if not selected:
                return []

            weights = torch.stack(selected, dim=0)  # (H, N_tok, N_frames)
            std, mean = torch.std_mean(weights, dim=-1, keepdim=True, unbiased=False)
            weights = (weights - mean) / (std + 1e-6)
            weights_np = weights.cpu().numpy()
            if n_frames >= 7:
                weights_np = median_filter(weights_np, size=(1, 1, 7))
            matrix = weights_np.mean(axis=0)  # (N_tok, N_frames)

            text_idx, time_idx = _dtw_path(-matrix)
            if text_idx.size == 0:
                return []

            # Map each text token index to its start/end frame (20ms per Whisper encoder frame)
            n_tok = len(text_ids)
            tok_starts: list[float] = []
            tok_ends: list[float] = []
            last_t = 0.0
            for ti in range(n_tok):
                hits = time_idx[text_idx == ti]
                if hits.size > 0:
                    s_local = float(hits[0]) * 0.02
                    e_local = float(hits[-1] + 1) * 0.02
                    last_t = e_local
                else:
                    s_local = last_t
                    e_local = min(chunk_dur, last_t + 0.06)
                ws_f, we_f = map_crop_time_to_source(s_local, e_local, offset_sec, total_duration)
                tok_starts.append(ws_f)
                tok_ends.append(we_f)

            return _split_tokens_to_words(tokenizer, text_ids, tok_starts, tok_ends)
        except Exception:
            return []

    def close(self) -> None:
        self.pipe = None
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _parse_qwen_raw_output(raw: str, forced_language: str | None = None) -> tuple[str, str]:
    """Remove Qwen metadata without the SDK's destructive repetition cleanup.

    qwen-asr 0.0.6 ``parse_asr_output`` collapses runs of 20 repeated
    patterns. Those can be real sung syllables, so preserve the decoded text.
    """
    text = raw.strip()
    if "<asr_text>" not in text:
        return forced_language or "", text
    metadata, text = text.split("<asr_text>", 1)
    match = re.search(r"^\s*language\s+(.+?)\s*$", metadata, re.MULTILINE | re.IGNORECASE)
    language = forced_language or (match.group(1).strip() if match else "")
    return ("" if language.casefold() == "none" else language), text.strip()


class Qwen3ASREngine:
    """Singing ASR; Whisper aligns Qwen's text without rewriting it.

    Qwen's own forced aligner does not support Vietnamese. Reusing the existing
    Whisper aligner avoids silently inventing equally spaced word timestamps.
    Install ``qwen-asr==0.0.6`` in the lyrics environment to enable this backend.
    """

    _LANGUAGES = {
        **{code: name.title() for code, name in ISO_TO_WHISPER_LANG.items()},
        "ar": "Arabic", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
        "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
        "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
        "mk": "Macedonian", "ro": "Romanian", "tr": "Turkish", "hi": "Hindi",
    }

    def __init__(self, *, log: Callable[[str], None] = lambda _m: None) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR chưa được cài trong môi trường Songscription. "
                "Cài qwen-asr==0.0.6 hoặc triển khai lại worker Modal; "
                "chọn Whisper Large v3 nếu muốn dùng môi trường hiện tại."
            ) from exc
        import torch

        self.model_id = QWEN3_ASR_ID
        self.repo_id = QWEN3_ASR_REPO
        self._log = log
        self._aligner: WhisperLargeV3Engine | None = None
        self._alignment_unavailable = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log(f"Đang nạp {self.repo_id} ({self.device})...")
        self.model = Qwen3ASRModel.from_pretrained(
            self.repo_id, dtype=dtype, device_map=self.device,
            max_inference_batch_size=1, max_new_tokens=320,
        )

    @classmethod
    def _language_name(cls, language_iso: str | None) -> str | None:
        if not language_iso or language_iso == "auto":
            return None
        low = language_iso.strip().lower()
        if low in cls._LANGUAGES:
            return cls._LANGUAGES[low]
        return next((name for name in cls._LANGUAGES.values() if name.lower() == low), None)

    @classmethod
    def _language_iso(cls, detected: str, fallback: str = "en") -> str:
        # A mixed-language result may be "Japanese,English". Segment-level schema
        # needs a primary language; retain the complete value in language_raw.
        first = detected.split(",", 1)[0].strip().lower()
        return next((code for code, name in cls._LANGUAGES.items() if name.lower() == first), fallback)

    def detect_language(self, audio_16k: np.ndarray) -> str:
        return self.transcribe_chunk(
            audio_16k, offset_sec=0.0, total_duration=audio_16k.size / 16000.0,
            want_timestamps=False,
        )["language"]

    def transcribe_chunk(
        self, chunk_16k: np.ndarray, *, offset_sec: float, total_duration: float,
        language_iso: str | None = None, want_timestamps: bool = True,
    ) -> dict[str, Any]:
        if self.model is None or chunk_16k.size < 1600:
            return {"text": "", "language": language_iso or "en", "words": [], "timing_source": "asr_segment"}
        # Pinned 0.0.6 adapter: use the SDK's raw inference to avoid its
        # parse_asr_output -> detect_and_fix_repetitions text deletion. No SDK
        # globals are changed; only these <=28s Songscription crops use it.
        infer = getattr(self.model, "_infer_asr", None)
        if not callable(infer):
            raise RuntimeError("Qwen runtime không tương thích: cần qwen-asr==0.0.6 với raw inference.")
        import torch

        language_name = self._language_name(language_iso)
        with torch.inference_mode():
            result = infer(
                # Context is vocabulary bias, not an instruction prompt.
                contexts=[""], wavs=[chunk_16k.astype(np.float32, copy=False)],
                languages=[language_name],
            )
        if not isinstance(result, list) or not result or not isinstance(result[0], str):
            raise RuntimeError("Qwen runtime trả dữ liệu raw không hợp lệ; kiểm tra qwen-asr==0.0.6.")
        from .lyrics_service import _is_runaway_token_loop

        raw_lang, text = _parse_qwen_raw_output(result[0], language_name)
        lang = language_iso or self._language_iso(raw_lang)
        chunk_dur = chunk_16k.size / 16000.0
        skip_align_loop = bool(text and _is_runaway_token_loop(text, duration_sec=chunk_dur))
        words = self.force_align_chunk(
            chunk_16k, text, offset_sec=offset_sec, total_duration=total_duration,
            language_iso=lang,
        ) if (text and want_timestamps and not skip_align_loop) else []
        return {
            "text": text, "language": lang, "language_raw": raw_lang, "words": words,
            "timing_source": "forced_aligner" if words else "asr_segment",
        }

    def force_align_chunk(
        self, chunk_16k: np.ndarray, text: str, *, offset_sec: float,
        total_duration: float, language_iso: str | None,
    ) -> list[dict[str, Any]]:
        if self._alignment_unavailable or not text.strip():
            return []
        try:
            if self._aligner is None:
                self._aligner = WhisperLargeV3Engine(model_id="whisper-large-v3", log=self._log)
            return self._aligner.force_align_chunk(
                chunk_16k, text, offset_sec=offset_sec, total_duration=total_duration,
                language_iso=language_iso,
            )
        except Exception as exc:
            self._alignment_unavailable = True
            self._log(f"Không nạp được bộ căn thời gian Whisper ({type(exc).__name__}); giữ lời Qwen và mốc theo đoạn.")
            return []

    def close(self) -> None:
        self.model = None
        if self._aligner is not None:
            self._aligner.close()
            self._aligner = None
        gc.collect()


def create_music_asr_engine(
    model_id: str, *, log: Callable[[str], None] = lambda _m: None,
) -> WhisperLargeV3Engine | Qwen3ASREngine:
    """Select the requested recognizer without silently substituting another model."""
    eff_model = normalize_model_id(model_id)
    if eff_model == QWEN3_ASR_ID:
        return Qwen3ASREngine(log=log)
    return WhisperLargeV3Engine(model_id=eff_model, log=log)


# ================================================================== Clean Songscription Pipeline
def _build_segment_record(
    *,
    start: float,
    end: float,
    text: str,
    text_raw: str,
    stream_id: str,
    primary_lang: str,
    chunk_lang: str | None,
    source_variant: str,
    words: list[dict[str, Any]],
    timing_source: str,
    is_adlib_hint: bool | None = None,
) -> dict[str, Any]:
    from .lyrics_service import infer_segment_script_lang

    ev_type, ev_adlib, lang_hint = classify_vocal_event(
        text, is_adlib_hint=is_adlib_hint, fallback_lang=chunk_lang or primary_lang
    )
    is_adlib = bool(ev_adlib is True or is_adlib_hint)
    if ev_type in ("humming", "unknown_vocal"):
        words = []
    seg_lang = (
        None
        if ev_type in ("humming", "unknown_vocal") or (ev_type == "vocable" and lang_hint is None)
        else (lang_hint or infer_segment_script_lang(text, chunk_lang or primary_lang))
    )
    return {
        "start": round(start, 2),
        "end": round(max(start + 0.12, end), 2),
        "text": text,
        "text_raw": text_raw,
        "text_edited": None,
        "event_type": ev_type,
        "vocal_role": "lead" if (stream_id == DEFAULT_STREAM_ID and not is_adlib) else "backing",
        "is_adlib": ev_adlib if isinstance(ev_adlib, bool) else is_adlib,
        "stream_id": stream_id,
        "language": seg_lang,
        "lang": str(seg_lang or ""),
        "confidence": None,
        "low_confidence": False,
        "review_status": None,
        "review_reasons": [],
        "review_reason": None,
        "timing_source": timing_source if words else "asr_segment",
        "timing_granularity": "word" if words else ("region" if ev_type in ("humming", "unknown_vocal") else "line"),
        "source_audio_variant": source_variant,
        "words": words,
        "candidates": [],
    }


def _flag_untranscribed_vocal_chunk(
    *,
    chunk_audio: np.ndarray,
    start: float,
    end: float,
    rms_env: np.ndarray,
    vocal_floor: float,
    stream_id: str,
    trace: PipelineTrace,
) -> dict[str, Any] | None:
    """When Whisper Large v3 returns no text on a clearly harmonic singing span, emit an honest
    `unknown_vocal` review flag instead of fabricating 'Ah'/'Oh' or fake confidence %."""
    from .lyrics_service import segment_vocal_activity

    if (end - start) < 1.6 or chunk_audio.size < 6400:
        return None
    mr, ar = segment_vocal_activity(rms_env, vocal_floor, start, end)
    harm = estimate_vocal_harmonicity(chunk_audio)
    if ar < 0.55 or mr < max(vocal_floor * 3.5, 0.030) or harm["vocal_likely"] < 0.40:
        return None
    reasons = ["suspected_vocal_untranscribed", "vocal_like_heuristic_only"]
    trace.record_recovery(
        start=start,
        end=end,
        acoustic_stats={"mean_rms": round(mr, 4), "active_ratio": round(ar, 3), **harm},
        candidates=[],
        outcome="flagged_unknown_vocal",
    )
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "text": "",
        "text_raw": "",
        "text_edited": None,
        "event_type": "unknown_vocal",
        "vocal_role": "backing" if stream_id != DEFAULT_STREAM_ID else "unknown",
        "is_adlib": None,
        "stream_id": stream_id,
        "language": None,
        "lang": "",
        "confidence": None,
        "low_confidence": True,
        "review_status": "needs_review",
        "review_reasons": reasons,
        "review_reason": format_review_reason(reasons),
        "timing_source": "acoustic_region",
        "timing_granularity": "region",
        "source_audio_variant": "vocal_stem",
        "words": [],
        "candidates": [],
    }


def _refine_line_with_second_pass_aligner(
    *,
    asr: WhisperLargeV3Engine | Qwen3ASREngine,
    asr_audio: np.ndarray,
    line_text: str,
    ls: float,
    le: float,
    stage1_words: list[dict[str, Any]],
    raw_words_had_collapse: bool,
    reg_s: float,
    reg_e: float,
    duration: float,
    lang_iso: str | None,
    rms_env: np.ndarray,
    vocal_floor: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Run fast Stage-2 single-sentence cross-attention DTW alignment only when Stage-1 had >= 2 collapsed tokens."""
    if not stage1_words or not raw_words_had_collapse or len(stage1_words) < 3:
        return ls, le, stage1_words

    a_s = max(0, int(max(reg_s, ls - 0.12) * 16000))
    a_e = min(asr_audio.shape[0], int(min(reg_e, le + 0.12) * 16000))
    if a_e - a_s < 3200:
        return ls, le, stage1_words

    fine_raw = asr.force_align_chunk(
        asr_audio[a_s:a_e],
        line_text,
        offset_sec=a_s / 16000.0,
        total_duration=duration,
        language_iso=lang_iso,
    )
    if not fine_raw or not _alignment_covers_text(fine_raw, line_text):
        return ls, le, stage1_words

    fine_rep = _repair_aligned_words(fine_raw, rms_env, vocal_floor, min(reg_e, le + 0.06))
    fine_rep[0]["word"] = str(fine_rep[0]["word"]).lstrip()
    new_s = round(max(reg_s, float(fine_rep[0]["start"])), 2)
    new_e = round(min(reg_e, max(new_s + 0.10, float(fine_rep[-1]["end"]))), 2)
    return new_s, new_e, fine_rep


def align_reference_lines_with_whisper(
    *,
    asr: WhisperLargeV3Engine | Qwen3ASREngine,
    asr_audio: np.ndarray,
    duration: float,
    lines: list[dict[str, Any]],
    primary_lang: str,
    rms_env: np.ndarray,
    vocal_floor: float,
    source_variant: str,
) -> list[dict[str, Any]]:
    """Force-align verified reference or user-provided lyric lines directly to vocal audio
    using `openai/whisper-large-v3` cross-attention DTW, preserving 100% exact line text & breaks."""
    from .lyrics_service import infer_segment_script_lang

    if not lines:
        return []

    n_lines = len(lines)
    aligned_segments: list[dict[str, Any]] = []
    prev_end_bound = 0.0

    for idx, item in enumerate(lines):
        txt = str(item.get("text") or "").strip()
        if not txt or _PUNCT_ONLY_RE.match(txt):
            continue

        est_s = max(0.0, float(item.get("start") or 0.0))
        next_est_s = (
            max(est_s + 0.5, float(lines[idx + 1].get("start") or (est_s + 4.0)))
            if idx + 1 < n_lines
            else duration
        )
        raw_end = item.get("end")
        if raw_end is not None:
            est_e = max(est_s + 0.4, min(duration, float(raw_end)))
        else:
            est_e = max(est_s + 0.5, min(duration, next_est_s, est_s + 12.0))

        win_s = max(0.0, max(prev_end_bound - 0.15, est_s - 0.55))
        win_e = min(duration, min(next_est_s + 0.35, est_e + 0.55))
        if win_e - win_s < 0.35:
            win_e = min(duration, win_s + 1.5)

        line_lang = infer_segment_script_lang(txt, primary_lang or "en")
        s_i = max(0, int(win_s * 16000))
        e_i = min(asr_audio.shape[0], int(win_e * 16000))

        words: list[dict[str, Any]] = []
        if e_i - s_i >= 2400:
            raw_w = asr.force_align_chunk(
                asr_audio[s_i:e_i],
                txt,
                offset_sec=s_i / 16000.0,
                total_duration=duration,
                language_iso=line_lang,
            )
            if raw_w and _alignment_covers_text(raw_w, txt):
                words = _repair_aligned_words(raw_w, rms_env, vocal_floor, win_e)
                if words:
                    words[0]["word"] = str(words[0]["word"]).lstrip()

        if words:
            ls = round(max(0.0, float(words[0]["start"])), 2)
            le = round(min(duration, max(ls + 0.15, float(words[-1]["end"]))), 2)
            timing_source = "forced_aligner"
        else:
            ls = round(est_s, 2)
            le = round(min(duration, max(ls + 0.25, est_e)), 2)
            timing_source = "reference_synced"

        rec = _build_segment_record(
            start=ls,
            end=le,
            text=txt,
            text_raw=txt,
            stream_id=str(item.get("stream_id") or DEFAULT_STREAM_ID),
            primary_lang=primary_lang,
            chunk_lang=line_lang,
            source_variant=source_variant,
            words=words,
            timing_source=timing_source,
        )
        if item.get("section"):
            rec["section"] = item["section"]
        if not words:
            rec.update(low_confidence=True, review_status="needs_review",
                       review_reasons=["vocal_timing_unresolved"],
                       review_reason=format_review_reason(["vocal_timing_unresolved"]))
        aligned_segments.append(rec)
        prev_end_bound = max(prev_end_bound, le)

    return assign_overlap_groups(_enforce_non_overlapping_same_stream(aligned_segments))


def run_music_asr_pipeline(
    *,
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
    audio_sha1: str = "",
    track_title: str = "",
    mode: str = "extract",
    user_lyrics_text: str = "",
    pure_vocal_audio: np.ndarray | None = None,
    original_audio: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recognize singing with the selected backend and preserve alignment provenance."""
    from .lyrics_service import (
        HALLUCINATION_PATTERNS,
        NOISE_ONLY_PATTERN,
        _align_vocalise_to_envelope,
        _clean_vocalise_text,
        _extract_vocal_regions,
        _is_mostly_vocalise,
        _is_runaway_token_loop,
        _norm_text_for_cmp,
        _split_long_regions_at_pauses,
        language_label,
        sanitize_runaway_repetition,
        segment_vocal_activity,
    )

    eff_model = normalize_model_id(model_id)
    req_lang = (requested_lang or "auto").strip().lower()
    forced_iso = req_lang if (req_lang and req_lang != "auto") else None
    source_variant = "vocal_stem" if vocal_isolated else "original_mix"
    progress(0.40, "recognizing_lyrics")

    asr = create_music_asr_engine(eff_model, log=log)
    try:
        trace = PipelineTrace(
            model_id=eff_model,
            requested_lang=req_lang,
            vocal_isolated=vocal_isolated,
            audio_sha1=audio_sha1,
        )
        trace.config = {
            "mode": eff_model,
            "asr_repo": asr.repo_id,
            "alignment_repo": WHISPER_LARGE_V3_REPO,
        }

        regions = (
            _extract_vocal_regions(rms_env, vocal_floor, duration)
            if vocal_isolated
            else _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)
        ) or _split_long_regions_at_pauses([(0.0, duration)], rms_env, duration)

        valid_chunks: list[tuple[int, float, float, np.ndarray]] = []
        for idx, (reg_s, reg_e) in enumerate(regions):
            s_i, e_i = max(0, int(reg_s * 16000)), min(asr_audio.shape[0], int(reg_e * 16000))
            if e_i - s_i >= 3200:
                valid_chunks.append((idx, reg_s, reg_e, asr_audio[s_i:e_i]))

        # Detect song-level language once on the strongest singing chunk when auto
        song_iso = forced_iso
        if song_iso is None and valid_chunks:
            best_chunk = max(valid_chunks[:4], key=lambda item: float(np.sqrt(np.mean(np.square(item[3], dtype=np.float64)) + 1e-12)))
            song_iso = asr.detect_language(best_chunk[3])

        out_segments: list[dict[str, Any]] = []
        lang_counts: dict[str, int] = {}
        total_chunks = max(1, len(valid_chunks))
        processed = 0

        def _split_chunk_for_loop_retry(cs: float, ce: float, depth: int) -> list[tuple[float, float]]:
            max_s = 8.5 if depth == 0 else 5.0
            min_s = 4.2 if depth == 0 else 2.4
            parts = _split_long_regions_at_pauses([(cs, ce)], rms_env, duration, max_chunk_sec=max_s, min_chunk_sec=min_s)
            if len(parts) >= 2:
                return parts
            # Fallback: split near the deepest vocal energy valley in the middle 50% of the chunk
            mid_lo, mid_hi = cs + (ce - cs) * 0.28, cs + (ce - cs) * 0.72
            if rms_env.size > 0 and mid_hi > mid_lo + 0.2:
                i0 = max(0, min(rms_env.size - 1, int(round(mid_lo * 20.0))))
                i1 = max(i0 + 1, min(rms_env.size, int(round(mid_hi * 20.0))))
                cut = round((i0 + int(np.argmin(rms_env[i0:i1]))) * 0.05, 2)
                if cs + 1.5 <= cut <= ce - 1.5:
                    return [(round(cs, 2), cut), (cut, round(ce, 2))]
            mid = round((cs + ce) * 0.5, 2)
            return [(round(cs, 2), mid), (mid, round(ce, 2))]

        def _process_chunk(chunk_idx: int, reg_s: float, reg_e: float, arr: np.ndarray, *, depth: int = 0) -> None:
            if should_stop():
                raise Cancelled()
            chunk_dur = max(0.1, reg_e - reg_s)
            wres = asr.transcribe_chunk(
                arr,
                offset_sec=reg_s,
                total_duration=duration,
                language_iso=forced_iso if eff_model == QWEN3_ASR_ID else forced_iso or song_iso,
                want_timestamps=True,
            )
            chunk_lang = forced_iso or wres.get("language") or song_iso or "en"
            raw_txt = str(wres.get("text") or "").strip()
            loop_trimmed = False

            if raw_txt and _is_runaway_token_loop(raw_txt, duration_sec=chunk_dur):
                # 1. Long chunk hit autoregressive repetition loop -> split at energy valleys & re-transcribe!
                if chunk_dur > 7.5 and depth < 2:
                    sub_spans = _split_chunk_for_loop_retry(reg_s, reg_e, depth)
                    if len(sub_spans) >= 2:
                        log(
                            f"Phát hiện vòng lặp âm tiết tại {reg_s:.1f}–{reg_e:.1f}s — "
                            f"chia nhỏ thành {len(sub_spans)} đoạn để đọc lại chính xác..."
                        )
                        for sub_s, sub_e in sub_spans:
                            si, ei = max(0, int(sub_s * 16000)), min(asr_audio.shape[0], int(sub_e * 16000))
                            if ei - si >= 3200:
                                _process_chunk(chunk_idx, sub_s, sub_e, asr_audio[si:ei], depth=depth + 1)
                        return

                # 2. Short sub-chunk still looped on Qwen -> cross-check with already-loaded Whisper aligner
                whisper_fallback = getattr(asr, "_aligner", None)
                if isinstance(asr, Qwen3ASREngine) and whisper_fallback is not None and callable(getattr(whisper_fallback, "transcribe_chunk", None)):
                    try:
                        alt = whisper_fallback.transcribe_chunk(
                            arr,
                            offset_sec=reg_s,
                            total_duration=duration,
                            language_iso=forced_iso or song_iso or chunk_lang,
                            want_timestamps=True,
                        )
                        alt_txt = str(alt.get("text") or "").strip()
                        if (
                            alt_txt
                            and not NOISE_ONLY_PATTERN.match(alt_txt)
                            and not HALLUCINATION_PATTERNS.match(alt_txt)
                            and not _is_runaway_token_loop(alt_txt, duration_sec=chunk_dur)
                        ):
                            wres = alt
                            raw_txt = alt_txt
                            chunk_lang = forced_iso or alt.get("language") or chunk_lang
                    except Exception:
                        pass

                # 3. Still looped -> cleanly trim degenerate tail (preserving lexical prefix) or reject pure spam
                if raw_txt and _is_runaway_token_loop(raw_txt, duration_sec=chunk_dur):
                    trimmed, loop_trimmed = sanitize_runaway_repetition(raw_txt, duration_sec=chunk_dur)
                    trace.record_change(
                        stage="asr_loop_guard",
                        action="trimmed" if trimmed else "dropped",
                        reason="runaway_repetition_loop",
                        start=reg_s,
                        end=reg_e,
                        before=raw_txt[:160],
                        after=trimmed[:160],
                    )
                    raw_txt = trimmed
                    if raw_txt:
                        realigned = asr.force_align_chunk(
                            arr,
                            raw_txt,
                            offset_sec=reg_s,
                            total_duration=duration,
                            language_iso=chunk_lang,
                        )
                        wres = {
                            **wres,
                            "text": raw_txt,
                            "words": realigned,
                            "timing_source": "forced_aligner" if realigned else "asr_segment",
                        }
                    else:
                        wres = {**wres, "text": "", "words": []}

            trace.record_chunk(
                index=chunk_idx,
                start=reg_s,
                end=reg_e,
                detected_lang=wres.get("language"),
                detected_lang_prob=None,
                decode_lang=chunk_lang,
                source_audio_variant=source_variant,
            )
            if raw_txt:
                trace.record_raw_asr(
                    chunk_index=chunk_idx, start=reg_s, end=reg_e, text=raw_txt, language=chunk_lang
                )
            if (
                not raw_txt
                or NOISE_ONLY_PATTERN.match(raw_txt)
                or HALLUCINATION_PATTERNS.match(raw_txt)
                or _is_runaway_token_loop(raw_txt, duration_sec=chunk_dur)
            ):
                if vocal_isolated:
                    flagged = _flag_untranscribed_vocal_chunk(
                        chunk_audio=arr,
                        start=reg_s,
                        end=reg_e,
                        rms_env=rms_env,
                        vocal_floor=vocal_floor,
                        stream_id=DEFAULT_STREAM_ID,
                        trace=trace,
                    )
                    if flagged is not None:
                        out_segments.append(flagged)
                return

            raw_w = list(wres.get("words") or [])
            incomplete_alignment = bool(raw_w and not _alignment_covers_text(raw_w, raw_txt))
            words = _repair_aligned_words(raw_w, rms_env, vocal_floor, reg_e)
            line_tuples = (
                _group_words_into_lyric_lines(
                    words, raw_txt, reg_s, reg_e, is_vocalise_chunk=_is_mostly_vocalise(raw_txt)
                )
                if words
                else _split_plain_text_into_lines(raw_txt, reg_s, reg_e)
            )

            is_mixed_so_far = len(lang_counts) > 1
            dom_lang = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else chunk_lang
            for l_txt, ls, le, lw in line_tuples:
                collapsed_line = bool(lw and len(lw) >= 2 and (le - ls) <= 0.16)
                raw_in_line = [w for w in raw_w if (ls - 0.15) <= float(w["start"]) <= (le + 0.15)]
                ls, le, lw = _refine_line_with_second_pass_aligner(
                    asr=asr,
                    asr_audio=asr_audio,
                    line_text=l_txt,
                    ls=ls,
                    le=le,
                    stage1_words=lw,
                    raw_words_had_collapse=_has_collapsed_words(raw_in_line),
                    reg_s=reg_s,
                    reg_e=reg_e,
                    duration=duration,
                    lang_iso=chunk_lang,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                )
                mr, ar = segment_vocal_activity(rms_env, vocal_floor, ls, le)
                quiet_vocal = vocal_isolated and ar < 0.08 and mr < vocal_floor * 0.9
                clean_lang = dom_lang if len(re.sub(r"\W+", "", l_txt)) <= 2 else chunk_lang
                cleaned, is_adlib = _clean_vocalise_text(
                    l_txt, clean_lang, allow_rhythmic=True, is_mixed=is_mixed_so_far
                )
                if not cleaned or NOISE_ONLY_PATTERN.match(cleaned) or HALLUCINATION_PATTERNS.match(cleaned):
                    continue
                if _norm_text_for_cmp(cleaned) != _norm_text_for_cmp(l_txt):
                    lw = []
                rec = _build_segment_record(
                    start=ls,
                    end=le,
                    text=cleaned,
                    text_raw=l_txt,
                    stream_id=DEFAULT_STREAM_ID,
                    primary_lang=chunk_lang,
                    chunk_lang=chunk_lang,
                    source_variant=source_variant,
                    words=lw,
                    timing_source=(lw[0].get("timing_source") or wres.get("timing_source") or "asr_word") if lw else "asr_segment",
                    is_adlib_hint=is_adlib,
                )
                if loop_trimmed or quiet_vocal or collapsed_line or incomplete_alignment:
                    rec["low_confidence"] = True
                    rec["review_status"] = "needs_review"
                    reasons_l: list[str] = []
                    if loop_trimmed:
                        reasons_l.append("asr_repetition_loop_trimmed")
                    if quiet_vocal:
                        reasons_l.append("weak_vocal_activity")
                    elif collapsed_line or incomplete_alignment:
                        reasons_l.append("vocal_timing_unresolved")
                    rec["review_reasons"] = reasons_l
                    rec["review_reason"] = format_review_reason(reasons_l)
                if rec.get("language"):
                    lg = str(rec["language"])
                    lang_counts[lg] = lang_counts.get(lg, 0) + 1
                # If Pass-1 ASR collapsed a multi-pulse vocalise (e.g. "Ah — ah" -> "Ah."),
                # expand each articulated pulse with exact onset/offset word timestamps.
                if (
                    (rec.get("event_type") in ("vocable", "humming") or rec.get("is_adlib") is True)
                    and vocal_isolated
                    and pure_vocal_audio is not None
                    and (float(rec["end"]) - float(rec["start"])) >= 0.55
                ):
                    from .vocal_recovery import expand_collapsed_vocalise_with_asr

                    exp = expand_collapsed_vocalise_with_asr(
                        asr=asr,
                        text=str(rec["text"]),
                        full_vocal_16k=pure_vocal_audio,
                        start=float(rec["start"]),
                        end=float(rec["end"]),
                        vocal_floor=vocal_floor,
                        duration=duration,
                        language_iso=forced_iso or chunk_lang,
                        original_audio=original_audio,
                        should_stop=should_stop,
                    )
                    if exp is not None:
                        exp_txt, exp_words = exp
                        rec["text"] = exp_txt
                        rec["text_raw"] = exp_txt
                        rec["words"] = exp_words
                        if exp_words:
                            rec["start"] = round(float(exp_words[0]["start"]), 2)
                            rec["end"] = round(float(exp_words[-1]["end"]), 2)
                            rec["timing_source"] = str(exp_words[0].get("timing_source") or "acoustic_region")
                            rec["timing_granularity"] = "word"
                        else:
                            rec["timing_source"] = "asr_segment"
                            rec["timing_granularity"] = "line"
                out_segments.extend(_align_vocalise_to_envelope(rec, rms_env, vocal_floor, duration))

        for idx, reg_s, reg_e, arr in valid_chunks:
            _process_chunk(idx, reg_s, reg_e, arr, depth=0)
            processed += 1
            progress(round(0.45 + 0.29 * min(1.0, processed / total_chunks), 3), "recognizing_lyrics")

        all_segments = _enforce_non_overlapping_same_stream(
            sorted(out_segments, key=lambda s: (float(s["start"]), float(s["end"])))
        )
        all_segments = assign_overlap_groups(all_segments)

        if req_lang and req_lang != "auto":
            primary_lang, lang_conf = req_lang, 1.0
            detected_languages = [{"code": primary_lang, "name": language_label(primary_lang), "probability": 1.0}]
        elif lang_counts:
            total_c = sum(lang_counts.values())
            ranked = sorted(lang_counts.items(), key=lambda kv: kv[1], reverse=True)
            primary_lang = ranked[0][0]
            lang_conf = round(ranked[0][1] / max(1, total_c), 4)
            detected_languages = [
                {"code": c, "name": language_label(c), "probability": round(cnt / max(1, total_c), 4)}
                for c, cnt in ranked[:4]
                if cnt / max(1, total_c) >= 0.05
            ]
        else:
            primary_lang, lang_conf = song_iso or "en", 0.0
            detected_languages = []

        diagnostics = trace.to_dict()
        if mode == "extract":
            from .vocal_recovery import merge_recovered_vocal_events, recover_vocals_with_asr

            if should_stop():
                raise Cancelled()
            progress(0.78, "reviewing_vocals")
            recovered, recovery_stats = recover_vocals_with_asr(
                asr=asr, segments=all_segments,
                full_vocal_16k=pure_vocal_audio if pure_vocal_audio is not None else asr_audio,
                rms_env=rms_env, vocal_floor=vocal_floor, duration=duration,
                primary_lang=primary_lang, vocal_isolated=vocal_isolated,
                original_audio=original_audio, should_stop=should_stop, log=log,
            )
            diagnostics["adlib_recovery"] = recovery_stats
            all_segments = assign_overlap_groups(sorted(
                merge_recovered_vocal_events(all_segments, recovered),
                key=lambda seg: (float(seg["start"]), float(seg["end"])),
            ))
        anchors = [vw for seg in all_segments for vw in (seg.get("words") or [])]
        progress(0.84, "aligning_timestamps")

        # Word-level DTW forced alignment with `openai/whisper-large-v3` for user lyrics or audio-verified reference
        if mode == "align" and user_lyrics_text.strip():
            from .lyrics_service import align_user_lyrics_to_anchors, parse_user_lyrics

            user_lines = parse_user_lyrics(user_lyrics_text)
            if user_lines:
                rough_segs, _ = align_user_lyrics_to_anchors(
                    user_lines=user_lines,
                    anchors=anchors,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    duration=duration,
                    primary_lang=primary_lang,
                )
                ref_aligned = align_reference_lines_with_whisper(
                    asr=asr,
                    asr_audio=asr_audio,
                    duration=duration,
                    lines=[
                        {
                            "text": s["text"],
                            "start": s["start"],
                            "end": s["end"],
                            "section": s.get("section"),
                            "stream_id": s.get("stream_id"),
                        }
                        for s in rough_segs
                    ],
                    primary_lang=primary_lang,
                    rms_env=rms_env,
                    vocal_floor=vocal_floor,
                    source_variant=source_variant,
                )
                if ref_aligned:
                    all_segments = ref_aligned
                    anchors = [vw for seg in all_segments for vw in (seg.get("words") or [])]
                    diagnostics["reference_lyrics"] = {
                        "source": "user_aligned_whisper",
                        "line_count": len(all_segments),
                    }
        elif mode == "extract" and track_title.strip() and all_segments:
            from .lyrics_reference import find_verified_reference_lyrics

            ref_match = find_verified_reference_lyrics(
                track_title=track_title,
                duration=duration,
                asr_segments=all_segments,
                log=log,
            )
            if ref_match is not None:
                ref_lines_input: list[dict[str, Any]] = []
                if len(ref_match.get("synced_lines") or []) >= 4:
                    offset = float(ref_match.get("time_offset") or 0.0)
                    for item in ref_match["synced_lines"]:
                        st = max(0.0, min(duration - 0.2, float(item["start"]) + offset))
                        ref_lines_input.append({"start": st, "text": item["text"]})
                elif str(ref_match.get("plain_text") or "").strip():
                    from .lyrics_service import align_user_lyrics_to_anchors, parse_user_lyrics

                    parsed_plain = parse_user_lyrics(str(ref_match["plain_text"]))
                    if parsed_plain:
                        rough_segs, _ = align_user_lyrics_to_anchors(
                            user_lines=parsed_plain,
                            anchors=anchors,
                            rms_env=rms_env,
                            vocal_floor=vocal_floor,
                            duration=duration,
                            primary_lang=primary_lang,
                        )
                        ref_lines_input = [
                            {"start": s["start"], "end": s["end"], "text": s["text"], "section": s.get("section")}
                            for s in rough_segs
                        ]

                if ref_lines_input:
                    log(f"Đang căn chỉnh từng từ bằng openai/whisper-large-v3 ({len(ref_lines_input)} câu)...")
                    ref_aligned = align_reference_lines_with_whisper(
                        asr=asr,
                        asr_audio=asr_audio,
                        duration=duration,
                        lines=ref_lines_input,
                        primary_lang=primary_lang,
                        rms_env=rms_env,
                        vocal_floor=vocal_floor,
                        source_variant=source_variant,
                    )
                    if ref_aligned:
                        from .vocal_recovery import merge_recovered_vocal_events

                        vocal_extras = [
                            seg for seg in all_segments
                            if seg.get("is_adlib") is True or seg.get("event_type") in ("vocable", "humming", "unknown_vocal")
                            or seg.get("vocal_role") == "backing"
                            or "candidate_pending" in (seg.get("review_reasons") or [])
                        ]
                        all_segments = assign_overlap_groups(
                            sorted(
                                merge_recovered_vocal_events(ref_aligned, vocal_extras),
                                key=lambda seg: (float(seg["start"]), float(seg["end"])),
                            )
                        )
                        anchors = [vw for seg in all_segments for vw in (seg.get("words") or [])]
                        diagnostics["reference_lyrics"] = {
                            "source": ref_match["source"],
                            "track_name": ref_match["track_name"],
                            "artist_name": ref_match["artist_name"],
                            "match_score": ref_match["match_score"],
                            "time_offset": ref_match["time_offset"],
                            "line_count": len(all_segments),
                        }

        diagnostics["recognition_engine"] = asr.repo_id
        diagnostics["word_alignment"] = (
            "whisper-large-v3-dtw" if any(w.get("timing_source") == "forced_aligner" for w in anchors)
            else "whisper-native-word" if anchors else "segment"
        )
        return {
            "language": primary_lang,
            "language_name": language_label(primary_lang),
            "language_confidence": round(float(lang_conf), 4),
            "detected_languages": detected_languages,
            "raw_segments": all_segments,
            "raw_asr_segments": trace.raw_asr_segments,
            "diagnostics": diagnostics,
            "anchors": anchors,
            "rms_env": rms_env,
            "vocal_floor": vocal_floor,
        }
    finally:
        asr.close()


def refine_segments_with_music_asr(
    *,
    segments: list[dict[str, Any]],
    target_indices: list[int],
    pure_vocal: np.ndarray,
    crisp_blend: np.ndarray,
    duration: float,
    has_vocal_stem: bool,
    decode_lang: str | None,
    model_id: str = DEFAULT_MODEL_ID,
    label_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-transcribe flagged/selected lines with the selected recognition backend."""
    from .lyrics_service import (
        HALLUCINATION_PATTERNS,
        NOISE_ONLY_PATTERN,
        _clean_vocalise_text,
        _is_runaway_token_loop,
        _norm_text_for_cmp,
        infer_segment_script_lang,
        sanitize_runaway_repetition,
    )

    eff_model = normalize_model_id(model_id)
    pfx = f"{label_prefix} · " if label_prefix else ""
    asr = create_music_asr_engine(eff_model)
    try:
        updated = [dict(s) for s in segments]
        results: list[dict[str, Any]] = []

        for idx in target_indices:
            seg = updated[idx]
            seg_id = int(seg.get("id") or (idx + 1))
            seg_s = max(0.0, float(seg.get("start") or 0.0))
            seg_e = max(seg_s + 0.15, float(seg.get("end") or (seg_s + 2.0)))
            old_text = str(seg.get("text") or "").strip()
            old_norm = _norm_text_for_cmp(old_text)
            old_is_vocalise = (
                seg.get("event_type") in ("vocable", "humming")
                or seg.get("is_adlib") is True
                or _is_vocalise_text(old_text)
            )
            seg_lang = decode_lang or str(seg.get("language") or seg.get("lang") or "") or None

            # Exact target bounds keep untimed/incomplete readings from copying
            # adjacent lyrics out of a padded crop.
            cs, ce = seg_s, min(duration, seg_e)
            s_i, e_i = max(0, int(cs * 16000)), min(pure_vocal.shape[0], int(ce * 16000))
            if e_i - s_i < 3200:
                results.append(
                    {
                        "id": seg_id,
                        "old_text": old_text,
                        "new_text": old_text,
                        "old_confidence": 0.0,
                        "new_confidence": 0.0,
                        "improved": False,
                        "changed": False,
                        "auto_resolved": False,
                        "variant": "Đoạn âm thanh quá ngắn",
                    }
                )
                continue

            cands: list[dict[str, Any]] = []
            chosen_text, chosen_variant, chosen_lang = "", "", seg_lang
            chosen_v_key = "vocal_stem" if has_vocal_stem else "original_mix"
            chosen_words: list[dict[str, Any]] = []
            variants = (
                (
                    (f"{pfx}{asr.repo_id} (Kênh vocal)", pure_vocal[s_i:e_i], "vocal_stem"),
                    (f"{pfx}{asr.repo_id} (Vocal + Mix)", crisp_blend[s_i:e_i], "vocal_mix_blend"),
                )
                if has_vocal_stem
                else ((f"{pfx}{asr.repo_id} (Bản gốc)", pure_vocal[s_i:e_i], "original_mix"),)
            )
            for v_lbl, arr, v_key in variants:
                w_out = asr.transcribe_chunk(
                    arr, offset_sec=cs, total_duration=duration, language_iso=seg_lang, want_timestamps=True
                )
                txt = str(w_out.get("text") or "").strip()
                clip_dur = max(0.1, ce - cs)
                trimmed_ref = False
                if txt and _is_runaway_token_loop(txt, duration_sec=clip_dur):
                    txt, trimmed_ref = sanitize_runaway_repetition(txt, duration_sec=clip_dur)
                if not txt or _is_runaway_token_loop(txt, duration_sec=clip_dur) or NOISE_ONLY_PATTERN.match(txt) or HALLUCINATION_PATTERNS.match(txt):
                    continue
                raw_words = [] if trimmed_ref else list(w_out.get("words") or [])
                words_in = [
                    w
                    for w in _repair_aligned_words(raw_words, None, 0.005, ce)
                    if (seg_s - 0.25) <= ((float(w["start"]) + float(w["end"])) * 0.5) <= (seg_e + 0.25)
                ]
                if words_in and _alignment_covers_text(words_in, txt):
                    words_in[0] = {**words_in[0], "word": str(words_in[0]["word"]).lstrip()}
                else:
                    words_in = []
                cleaned, _ = _clean_vocalise_text(txt, w_out.get("language") or seg_lang or "en", allow_rhythmic=True)
                if not cleaned:
                    continue
                ev_c, _, lg_c = classify_vocal_event(cleaned, fallback_lang=w_out.get("language") or seg_lang)
                if (
                    has_vocal_stem
                    and not trimmed_ref
                    and (old_is_vocalise or ev_c in ("vocable", "humming"))
                    and clip_dur >= 0.55
                ):
                    from .vocal_recovery import expand_collapsed_vocalise_with_asr

                    exp_ref = expand_collapsed_vocalise_with_asr(
                        asr=asr,
                        text=cleaned,
                        full_vocal_16k=pure_vocal,
                        start=cs,
                        end=ce,
                        vocal_floor=0.005,
                        duration=duration,
                        language_iso=seg_lang,
                        original_audio=crisp_blend,
                    )
                    if exp_ref is not None:
                        cleaned, words_in = exp_ref
                        txt = cleaned
                        ev_c, _, lg_c = classify_vocal_event(cleaned, fallback_lang=w_out.get("language") or seg_lang)
                cands.append(
                    {
                        "candidate_id": f"rt-{eff_model}-{seg_id}-{len(cands) + 1}",
                        "text": cleaned,
                        "event_type": ev_c,
                        "language": lg_c or w_out.get("language"),
                        "source_audio_variant": v_key if has_vocal_stem else "original_mix",
                        "config": {"engine": asr.repo_id, "variant": v_lbl},
                        "status": "pending",
                    }
                )
                if old_is_vocalise and _vocalise_loses_occurrences(old_text, cleaned):
                    # A retry with fewer syllables is an alternative reading,
                    # never a reason to overwrite an already recognized repeat.
                    continue
                if not chosen_text:
                    chosen_text, chosen_variant, chosen_v_key = cleaned, v_lbl, v_key
                    chosen_lang = lg_c or w_out.get("language") or seg_lang
                    chosen_words = words_in if _norm_text_for_cmp(cleaned) == _norm_text_for_cmp(txt) else []
                elif old_is_vocalise and _vocalise_adds_occurrences(chosen_text, cleaned):
                    # Prefer the retry that acoustically recognizes additional
                    # repetitions of the same syllables (e.g. Ah -> Ah ah).
                    chosen_text, chosen_variant, chosen_v_key = cleaned, v_lbl, v_key
                    chosen_lang = lg_c or w_out.get("language") or seg_lang
                    chosen_words = words_in if _norm_text_for_cmp(cleaned) == _norm_text_for_cmp(txt) else []

            if cands:
                for c in cands:
                    if c["text"] == chosen_text:
                        c["status"] = "accepted"
                        break
                seg["candidates"] = cands

            if chosen_text:
                changed = _norm_text_for_cmp(chosen_text) != old_norm
                ev_new, adlib_new, lang_new = classify_vocal_event(chosen_text, fallback_lang=chosen_lang or "en")
                seg.setdefault("text_raw", old_text)
                seg["text"] = chosen_text
                seg["event_type"] = ev_new
                if adlib_new is not None:
                    seg["is_adlib"] = adlib_new
                seg["words"] = [] if ev_new in ("humming", "unknown_vocal") else chosen_words
                res_lang = (
                    None
                    if ev_new in ("humming", "unknown_vocal") or (ev_new == "vocable" and lang_new is None)
                    else (lang_new or infer_segment_script_lang(chosen_text, chosen_lang or "en"))
                )
                seg["language"], seg["lang"] = res_lang, str(res_lang or "")
                seg["timing_source"] = (seg["words"][0].get("timing_source") or "asr_word") if seg["words"] else "asr_segment"
                seg["timing_granularity"] = "word" if seg["words"] else ("region" if ev_new in ("humming", "unknown_vocal") else "line")
                seg["source_audio_variant"] = chosen_v_key
                seg["low_confidence"] = False
                seg["review_status"] = "verified" if len(cands) >= 2 and len({c["text"].lower() for c in cands}) == 1 else "needs_review"
                results.append(
                    {
                        "id": seg_id,
                        "old_text": old_text,
                        "new_text": chosen_text,
                        "old_confidence": 0.0,
                        "new_confidence": 0.0,
                        "improved": changed or seg["review_status"] == "verified",
                        "changed": changed,
                        "auto_resolved": seg["review_status"] == "verified",
                        "variant": chosen_variant,
                        "candidates": cands,
                    }
                )
            else:
                results.append(
                    {
                        "id": seg_id,
                        "old_text": old_text,
                        "new_text": old_text,
                        "old_confidence": 0.0,
                        "new_confidence": 0.0,
                        "improved": False,
                        "changed": False,
                        "auto_resolved": False,
                        "variant": "Giữ nguyên bản cũ",
                        "candidates": cands,
                    }
                )
        return updated, results
    finally:
        asr.close()
