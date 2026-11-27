"""Targeted singing retries on the already loaded ASR engine, without a text API."""
from __future__ import annotations

import difflib
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from .engine import Cancelled
from .lyrics_schema import (
    DEFAULT_STREAM_ID,
    classify_vocal_event,
    estimate_vocal_harmonicity,
    format_review_reason,
)


def _word_interval(word: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start, end = float(word["start"]), float(word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if math.isfinite(start) and math.isfinite(end) and end > start else None


def _recognized_intervals(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = []
    for seg in segments:
        if seg.get("event_type") == "unknown_vocal" or not str(seg.get("text") or "").strip():
            continue
        # Word boundaries expose missed syllables inside an otherwise recognized line.
        words = [w for w in seg.get("words") or [] if str(w.get("word") or "").strip() and _word_interval(w)]
        if words:
            intervals.extend(_word_interval(w) for w in words)
        elif (interval := _word_interval(seg)) is not None:
            intervals.append(interval)
    return sorted(intervals)

def find_uncovered_vocal_gaps(
    *,
    segments: list[dict[str, Any]],
    full_vocal_16k: np.ndarray,
    rms_env: np.ndarray,
    vocal_floor: float,
    duration: float,
    min_gap_sec: float = 0.25,
    max_phrase_sec: float = 5.5,
    max_gaps: int | None = None,
) -> list[dict[str, Any]]:
    """Scan the BS-RoFormer vocal stem for singing spans NOT covered by existing lyric lines.

    Designed for general ad-lib / vocalise / humming recovery across any song:
      - Masks out regions already covered by recognized lyrics (+0.05s guard band).
      - Bridges only micro-dips (< 0.45s); genuine breath pauses keep distinct ad-lib phrases separate.
      - Splits long uncovered vocal spans (> 5.5s) at deepest internal energy valleys so multi-phrase
        vocalise bridges become separate lines instead of one giant merged block.
      - Filters by harmonicity to reject pure drum/percussion bleed.
    """
    n_frames = rms_env.size
    if n_frames == 0 or duration <= 0.5 or full_vocal_16k.size < 3200:
        return []

    covered = np.zeros(n_frames, dtype=bool)
    seg_intervals = _recognized_intervals(segments)
    for start, end in seg_intervals:
        s_t = max(0.0, start - 0.05)
        e_t = min(duration, end + 0.05)
        i0 = max(0, min(n_frames, int(math.floor(s_t * 20.0))))
        i1 = max(i0, min(n_frames, int(math.ceil(e_t * 20.0))))
        covered[i0:i1] = True

    thresh = max(float(vocal_floor) * 2.5, 0.016)
    uncovered_active = (~covered) & (rms_env >= thresh)
    if not np.any(uncovered_active):
        return []

    # Bridge only short micro-dips (< 0.45s = 9 frames) inside the SAME uncovered zone.
    # Never bridge across already-covered lyric segments or pauses >= 0.45s!
    bridged = uncovered_active.copy()
    idx = 0
    while idx < n_frames:
        if not bridged[idx]:
            j = idx
            while j < n_frames and not bridged[j]:
                j += 1
            if idx > 0 and j < n_frames and (j - idx) < 9 and not np.any(covered[idx:j]):
                bridged[idx:j] = True
            idx = j
        else:
            idx += 1

    raw_spans: list[tuple[int, int]] = []
    in_sp, st = False, 0
    for k in range(n_frames):
        if bridged[k] and not in_sp:
            in_sp, st = True, k
        elif not bridged[k] and in_sp:
            in_sp = False
            raw_spans.append((st, k))
    if in_sp:
        raw_spans.append((st, n_frames))

    # Split long uncovered spans (> max_phrase_sec) at deepest low-energy valley so distinct ad-lib
    # phrases separated by soft breath/reverb tails never collapse into a single line.
    split_spans: list[tuple[int, int]] = []
    max_frames = max(20, int(round(max_phrase_sec * 20.0)))
    min_sub_frames = max(4, int(round(min_gap_sec * 20.0)))

    def _split_recursive(a: int, b: int) -> None:
        if (b - a) <= max_frames:
            split_spans.append((a, b))
            return
        # Search interior for the deepest smoothed energy valley
        margin = max(min_sub_frames, int((b - a) * 0.18))
        lo, hi = a + margin, b - margin
        if hi <= lo:
            split_spans.append((a, b))
            return
        sub = rms_env[a:b]
        med = float(np.median(sub)) if sub.size else thresh
        win_sl = rms_env[lo:hi]
        smooth = np.convolve(win_sl, np.ones(5, dtype=np.float32) / 5.0, mode="same") if win_sl.size >= 5 else win_sl
        rel_min = int(np.argmin(smooth))
        valley_val = float(smooth[rel_min])
        # Split whenever there is a genuine vocal dip (< 55% of median or close to floor) OR span > 8.5s
        if valley_val <= max(thresh * 1.45, med * 0.55) or (b - a) > int(8.5 * 20):
            cut = lo + rel_min
            # Trim weak valley frames around the cut so the two phrases start/end cleanly
            l_end = cut
            while l_end > a + min_sub_frames and rms_env[l_end - 1] < thresh * 1.15:
                l_end -= 1
            r_start = cut + 1
            while r_start < b - min_sub_frames and rms_env[r_start] < thresh * 1.15:
                r_start += 1
            _split_recursive(a, l_end)
            _split_recursive(r_start, b)
        else:
            split_spans.append((a, b))

    for a, b in raw_spans:
        if (b - a) >= min_sub_frames:
            _split_recursive(a, b)

    # Compute safe neighbor bounds from existing segments so recovered ad-libs never collide

    candidates: list[dict[str, Any]] = []
    for a, b in split_spans:
        gs = round(max(0.0, a * 0.05), 2)
        ge = round(min(duration, b * 0.05), 2)
        dur = ge - gs
        if dur < min_gap_sec:
            continue
        s_i = max(0, int(gs * 16000))
        e_i = min(full_vocal_16k.shape[0], int(ge * 16000))
        if e_i - s_i < 3200:
            continue
        # The heuristic samples a short prefix; inspect distributed slices so a
        # long phrase is not judged solely on its first breath or reverb tail.
        harm = max(
            (estimate_vocal_harmonicity(full_vocal_16k[pos:min(e_i, pos + 48000)])
             for pos in range(s_i, e_i, 48000)),
            key=lambda h: h["vocal_likely"],
        )
        if harm["vocal_likely"] < 0.42 or harm["pitch_voicing_ratio"] < 0.12:
            continue

        # Hard bounds from nearest preceding / succeeding lyric lines
        bound_s = 0.0
        bound_e = duration
        for seg_s, seg_e in seg_intervals:
            if seg_e <= gs + 0.15:
                bound_s = max(bound_s, seg_e + 0.02)
            elif seg_s >= ge - 0.15:
                bound_e = min(bound_e, seg_s - 0.02)
                break
        if bound_e - bound_s < min_gap_sec * 0.7:
            continue

        mean_rms = float(np.mean(rms_env[a:b])) if b > a else 0.0
        candidates.append(
            {
                "start": round(max(bound_s, gs), 2),
                "end": round(min(bound_e, ge), 2),
                "bound_start": round(bound_s, 2),
                "bound_end": round(bound_e, 2),
                "duration": round(min(bound_e, ge) - max(bound_s, gs), 2),
                "mean_rms": round(mean_rms, 4),
                "vocal_likely": harm["vocal_likely"],
                "pitch_voicing_ratio": harm["pitch_voicing_ratio"],
            }
        )

    if max_gaps is not None and len(candidates) > max_gaps:
        # Keep the strongest vocal-energy spans, then restore chronological order
        ranked = sorted(candidates, key=lambda g: g["mean_rms"] * g["vocal_likely"] * min(4.0, g["duration"]), reverse=True)
        candidates = sorted(ranked[:max_gaps], key=lambda g: g["start"])
    return candidates


def find_adlib_review_windows(
    *, segments: list[dict[str, Any]], full_vocal_16k: np.ndarray,
    rms_env: np.ndarray, vocal_floor: float, duration: float,
) -> list[dict[str, Any]]:
    """Review acoustic gaps and explicitly suspect lines; never reread every lyric.

    Windows partition source time without overlap. Model word times already use
    that timeline, so callers must not add the crop offset a second time.
    """
    gaps = find_uncovered_vocal_gaps(
        segments=segments, full_vocal_16k=full_vocal_16k, rms_env=rms_env,
        vocal_floor=vocal_floor, duration=duration,
    )
    windows = [{**g, "occupied": False} for g in gaps]
    reserved = [(g["start"], g["end"]) for g in gaps]
    suspect_reasons = {"possible_overlap", "uncertain_words", "low_asr_logprob", "vocal_timing_unresolved"}
    for seg in sorted(segments, key=lambda s: float(s.get("start") or 0)):
        if seg.get("review_status") in {"verified", "edited", "skipped"} or not str(seg.get("text") or "").strip():
            continue
        text = str(seg.get("text") or "").strip()
        event_type, is_adlib, _ = classify_vocal_event(text)
        short_vocalise = (
            (event_type in ("vocable", "humming") or is_adlib is True)
            and len(_tokens(text)) == 1
        )
        # A single recognized ``Ah`` can mask a second short vocal burst in
        # the same span. Re-read only these compact vocal events, using the
        # isolated stem and then the mix as independent acoustic views.
        if not (
            seg.get("low_confidence")
            or suspect_reasons.intersection(seg.get("review_reasons") or [])
            or short_vocalise
        ):
            continue
        interval = _word_interval(seg)
        if interval is None:
            continue
        spans = [(max(0.0, interval[0]), min(duration, interval[1]))]
        for used_start, used_end in sorted(reserved):
            remaining = []
            for start, end in spans:
                if used_end <= start or used_start >= end:
                    remaining.append((start, end))
                else:
                    remaining.extend((a, b) for a, b in ((start, used_start), (used_end, end)) if b - a >= .25)
            spans = remaining
        for start, end in spans:
            while end - start >= .25:
                stop = min(end, start + 5.5)
                windows.append({"start": start, "end": stop, "occupied": True})
                reserved.append((start, stop))
                start = stop
    for window in windows:
        # Gaps use exact bounds: an untimed response from a padded gap could
        # otherwise copy adjacent lead words into the missing region.
        pad = .2 if window["occupied"] else 0.
        window["crop_start"] = max(0.0, window["start"] - pad)
        window["crop_end"] = min(duration, window["end"] + pad)
    return sorted(windows, key=lambda w: w["start"])


_TOKEN_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[^\W_]+")


def _tokens(text: str) -> list[str]:
    # Match CJK characters individually and Latin syllables/words with their
    # occurrence count. A set would erase real repeated ad-libs.
    return [m.group().casefold() for m in _TOKEN_PATTERN.finditer(text)]


def _new_timed_words(
    words: list[dict[str, Any]], baseline: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Subtract observations from the first pass one-to-one, preserving insertions.

    Temporal evidence disambiguates repeated words where first-pass word times
    exist. Untimed first-pass lines can only support conservative sequence matches.
    """
    old_tokens, old_times = [], []
    for seg in baseline:
        if seg.get("event_type") == "unknown_vocal":
            continue
        timed = [w for w in seg.get("words") or [] if _word_interval(w)]
        units = timed or [{"word": seg.get("text") or ""}]
        for unit in units:
            for token in _tokens(str(unit.get("word") or "")):
                old_tokens.append(token)
                old_times.append(_word_interval(unit))
    new_tokens, owners = [], []
    for index, word in enumerate(words):
        for token in _tokens(str(word.get("word") or "")):
            new_tokens.append(token)
            owners.append(index)
    matched: set[int] = set()
    # Match timed occurrences first, preferring the closest time. Sequence-only
    # matching can pair a late "oh" with an earlier "oh" and duplicate the late one.
    options = []
    for old_index, (token, old_time) in enumerate(zip(old_tokens, old_times)):
        if old_time is None:
            continue
        for position, new_token in enumerate(new_tokens):
            new_time = _word_interval(words[owners[position]])
            if token == new_token and new_time and (
                min(old_time[1], new_time[1]) > max(old_time[0], new_time[0])
                or abs(sum(old_time) - sum(new_time)) <= .5
            ):
                options.append((abs(sum(old_time) - sum(new_time)), old_index, position))
    used_old = set()
    for _, old_index, position in sorted(options):
        if old_index not in used_old and position not in matched:
            used_old.add(old_index)
            matched.add(position)
    untimed_tokens = [token for token, time in zip(old_tokens, old_times) if time is None]
    remaining = [pos for pos in range(len(new_tokens)) if pos not in matched]
    matcher = difflib.SequenceMatcher(None, untimed_tokens, [new_tokens[pos] for pos in remaining], autojunk=False)
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            matched.add(remaining[block.b + delta])
    counts = {index: [] for index in range(len(words))}
    for position, owner in enumerate(owners):
        counts[owner].append(position in matched)
    groups, group = [], []
    for index, word in enumerate(words):
        matches = counts[index]
        if matches and not all(matches):
            if any(matches):
                spans = list(_TOKEN_PATTERN.finditer(str(word.get("word") or "")))
                # Preserve unmatched syllables with the parent token's coarse
                # region. Never pretend these are precise subword timestamps.
                text = " ".join(span.group() for span, matched_token in zip(spans, matches) if not matched_token)
                group.append({**word, "word": " " + text, "_partial_word": True})
            else:
                group.append(word)
        elif group:
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    return groups


def _pulse_pitch_hz(clip: np.ndarray, sr: int = 16000) -> float:
    if clip.size < 400:
        return 0.0
    fc = clip.astype(np.float64, copy=False) - float(np.mean(clip))
    ac = np.correlate(fc, fc, mode="full")[fc.size - 1 :]
    hi = min(ac.size, 161)
    if ac[0] <= 1e-8 or hi <= 20:
        return 0.0
    sub = ac[18:hi]
    pk = int(np.argmax(sub))
    return float(sr) / float(18 + pk) if float(sub[pk] / ac[0]) >= 0.40 else 0.0


def detect_vocalise_syllable_pulses(
    vocal_16k: np.ndarray,
    start_sec: float,
    end_sec: float,
    vocal_floor: float,
    *,
    sr: int = 16000,
    min_pulse_sec: float = 0.22,
) -> list[tuple[float, float]]:
    """Segment a vocal clip into articulated acoustic pulses using smoothed RMS valley
    prominence and pitch-step confirmation (separates multi-syllable singing without
    splitting sustained vibrato notes)."""
    i0, i1 = max(0, int(round(start_sec * sr))), min(vocal_16k.size, int(round(end_sec * sr)))
    hop, win = int(sr * 0.025), int(sr * 0.060)
    if i1 - i0 < max(win, int(sr * 0.35)):
        return [(round(start_sec, 2), round(end_sec, 2))]

    clip = vocal_16k[i0:i1].astype(np.float64, copy=False)
    frames = np.lib.stride_tricks.sliding_window_view(clip, win)[::hop]
    env = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    smooth = np.convolve(env, np.array([0.1, 0.2, 0.4, 0.2, 0.1]), mode="same") if env.size >= 5 else env

    active_thresh = max(float(vocal_floor) * 2.4, 0.015)
    min_peak_amp = max(float(vocal_floor) * 3.0, 0.022)
    min_dist_f = max(4, int(round(min_pulse_sec / 0.025)))

    peaks: list[int] = []
    for k in range(1, smooth.size - 1):
        if smooth[k] >= min_peak_amp and smooth[k] >= smooth[k - 1] and smooth[k] >= smooth[k + 1]:
            if peaks and (k - peaks[-1]) < min_dist_f:
                if smooth[k] > smooth[peaks[-1]]:
                    peaks[-1] = k
            else:
                peaks.append(k)

    act = np.flatnonzero(smooth >= active_thresh)
    if len(peaks) <= 1:
        if act.size == 0:
            return [(round(start_sec, 2), round(end_sec, 2))]
        ps = round(start_sec + int(act[0]) * 0.025, 2)
        pe = round(min(end_sec, start_sec + (int(act[-1]) + 1) * 0.025 + 0.04), 2)
        return [(ps, max(ps + 0.15, pe))]

    while len(peaks) >= 2:
        merged = False
        for idx in range(len(peaks) - 1):
            p_l, p_r = peaks[idx], peaks[idx + 1]
            v_val = float(np.min(smooth[p_l : p_r + 1]))
            ratio = v_val / max(min(float(smooth[p_l]), float(smooth[p_r])), 1e-9)
            fl = _pulse_pitch_hz(clip[max(0, (p_l - 2) * hop) : min(clip.size, (p_l + 4) * hop)], sr)
            fr = _pulse_pitch_hz(clip[max(0, (p_r - 2) * hop) : min(clip.size, (p_r + 4) * hop)], sr)
            semitones = abs(12.0 * math.log2(fr / fl)) if (fl > 60.0 and fr > 60.0) else 0.0
            is_boundary = (ratio <= 0.56 or v_val <= active_thresh * 1.1) or (ratio <= 0.72 and 2.2 <= semitones <= 10.5)
            if not is_boundary:
                peaks.pop(idx + 1 if smooth[p_l] >= smooth[p_r] else idx)
                merged = True
                break
        if not merged:
            break

    boundaries = [int(act[0]) if act.size else 0]
    boundaries.extend(p_l + int(np.argmin(smooth[p_l : p_r + 1])) for p_l, p_r in zip(peaks, peaks[1:]))
    boundaries.append((int(act[-1]) + 1) if act.size else int(smooth.size))

    pulses: list[tuple[float, float]] = []
    for a, b in zip(boundaries, boundaries[1:]):
        while a < b - 2 and smooth[a] < active_thresh:
            a += 1
        while b > a + 2 and smooth[b - 1] < active_thresh:
            b -= 1
        ps = round(start_sec + a * 0.025, 2)
        pe = round(min(end_sec, start_sec + b * 0.025 + 0.02), 2)
        if pulses and ps < pulses[-1][1]:
            mid = round((pulses[-1][1] + ps) * 0.5, 2)
            pulses[-1] = (pulses[-1][0], max(pulses[-1][0] + 0.12, mid))
            ps = pulses[-1][1]
        if pe - ps >= 0.16:
            pulses.append((ps, pe))
    return pulses or [(round(start_sec, 2), round(end_sec, 2))]


def expand_collapsed_vocalise_with_asr(
    *,
    asr: Any,
    text: str,
    full_vocal_16k: np.ndarray,
    start: float,
    end: float,
    vocal_floor: float,
    duration: float,
    language_iso: str | None = None,
    original_audio: np.ndarray | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Re-recognize each acoustic pulse via ASR when a multi-syllable vocalise/ad-lib
    was collapsed to fewer tokens than distinct acoustic pulses.

    Every emitted token comes strictly from ASR output on that pulse — never defaults
    or copies text onto unrecognized pulses, and never fabricates equal-spaced word times.
    """
    from .lyrics_service import (
        HALLUCINATION_PATTERNS,
        NOISE_ONLY_PATTERN,
        _is_runaway_token_loop,
        _normalize_vocal_rms,
    )

    clean_in = (text or "").strip()
    if not clean_in or full_vocal_16k.size < 3200 or (end - start) < 0.55:
        return None
    ev_type, is_adlib, _ = classify_vocal_event(clean_in)
    if ev_type not in ("vocable", "humming") and is_adlib is not True:
        return None

    in_toks = _tokens(clean_in)
    if not in_toks:
        return None

    pulses = detect_vocalise_syllable_pulses(full_vocal_16k, start, end, vocal_floor)
    if len(pulses) < 2 or len(in_toks) >= len(pulses):
        return None

    recognized: list[tuple[float, float, str, list[dict[str, Any]]]] = []
    for ps, pe in pulses:
        if should_stop():
            raise Cancelled
        li = int(round(max(start, ps - 0.03) * 16000))
        ri = min(full_vocal_16k.size, int(round(min(end, pe + 0.03) * 16000)))
        if ri - li < 1600:
            continue
        p_dur = (ri - li) / 16000.0
        sources_p = [full_vocal_16k] + ([original_audio] if (original_audio is not None and original_audio.size >= ri) else [])
        for src_arr in sources_p:
            try:
                p_res = asr.transcribe_chunk(
                    _normalize_vocal_rms(src_arr[li:ri]),
                    offset_sec=li / 16000.0,
                    total_duration=duration,
                    language_iso=language_iso,
                    want_timestamps=False,
                )
            except Cancelled:
                raise
            except Exception:
                continue
            cand = str((p_res or {}).get("text") or "").strip()
            if (
                cand
                and not NOISE_ONLY_PATTERN.match(cand)
                and not HALLUCINATION_PATTERNS.match(cand)
                and not _is_runaway_token_loop(cand, duration_sec=p_dur)
            ):
                p_txt = re.sub(r"[\s.,!?;:。！？]+$", "", cand).strip()
                if p_txt:
                    p_words: list[dict[str, Any]] = []
                    if len(_tokens(p_txt)) > 1 and callable(getattr(asr, "force_align_chunk", None)):
                        try:
                            aligned = asr.force_align_chunk(
                                _normalize_vocal_rms(src_arr[li:ri]),
                                p_txt,
                                offset_sec=li / 16000.0,
                                total_duration=duration,
                                language_iso=language_iso or (p_res or {}).get("language"),
                            )
                            if aligned and _tokens("".join(str(w.get("word") or "") for w in aligned)) == _tokens(p_txt):
                                p_words = [
                                    {**w, "start": round(max(ps, float(w["start"])), 2), "end": round(min(pe, float(w["end"])), 2)}
                                    for w in aligned
                                    if _word_interval(w) is not None
                                ]
                        except Exception:
                            p_words = []
                    recognized.append((ps, pe, p_txt, p_words))
                    break

    total_new_toks = sum(len(_tokens(txt)) for _, _, txt, _ in recognized)
    if len(recognized) < 2 or total_new_toks <= len(in_toks):
        return None

    parts: list[str] = []
    words_out: list[dict[str, Any]] = []
    all_timed = True

    for idx, (ps, pe, raw_syl, aligned_words) in enumerate(recognized):
        piece = raw_syl.strip()
        if idx == 0 and piece and piece[0].islower() and piece[0].isalpha():
            piece = piece[0].upper() + piece[1:]
        elif idx > 0 and piece and piece[0].isupper() and len(piece.split()) == 1 and piece.isalpha():
            piece = piece.lower()

        sep = ", " if (idx > 0 and (ps - recognized[idx - 1][1]) >= 0.08) else (" " if idx > 0 else "")
        parts.append(sep + piece)

        if len(_tokens(piece)) == 1:
            if idx > 0 and sep == ", " and words_out and not str(words_out[-1]["word"]).rstrip().endswith((",", ".", "!", "?", "、", "。")):
                words_out[-1]["word"] = str(words_out[-1]["word"]) + ","
            words_out.append(
                {
                    "word": piece if idx == 0 else f" {piece}",
                    "start": round(ps, 2),
                    "end": max(round(ps + 0.04, 2), round(pe, 2)),
                    "probability": None,
                    "uncertain": False,
                    "timing_source": "acoustic_region",
                }
            )
        elif aligned_words:
            if idx > 0 and sep == ", " and words_out and not str(words_out[-1]["word"]).rstrip().endswith((",", ".", "!", "?", "、", "。")):
                words_out[-1]["word"] = str(words_out[-1]["word"]) + ","
            for wi, w in enumerate(aligned_words):
                w_txt = str(w.get("word") or "")
                if idx == 0 and wi == 0:
                    w_txt = w_txt.lstrip()
                    if w_txt and w_txt[0].islower() and w_txt[0].isalpha():
                        w_txt = w_txt[0].upper() + w_txt[1:]
                elif wi == 0 and not w_txt.startswith(" "):
                    w_txt = " " + w_txt
                words_out.append({**w, "word": w_txt})
        else:
            all_timed = False

    full_text = re.sub(r"\s+", " ", "".join(parts)).strip()
    return (full_text, words_out if all_timed else []) if full_text else None


def recover_vocals_with_asr(
    *, asr: Any, segments: list[dict[str, Any]], full_vocal_16k: np.ndarray,
    rms_env: np.ndarray, vocal_floor: float, duration: float, primary_lang: str,
    vocal_isolated: bool, original_audio: np.ndarray | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retry missing/suspect singing crops on the loaded Qwen or Whisper engine.

    Try the vocal stem first and original mix only if it yields nothing new.
    Candidates remain marked for listening review: ASR cannot guarantee all scat
    syllables or establish singer identity from a mixed vocal track.
    """
    from .lyrics_service import (
        HALLUCINATION_PATTERNS,
        NOISE_ONLY_PATTERN,
        _is_runaway_token_loop,
        _normalize_vocal_rms,
        sanitize_runaway_repetition,
    )
    from .music_asr_engine import _build_segment_record, _group_words_into_lyric_lines

    def check_stop() -> None:
        if should_stop():
            raise Cancelled

    check_stop()
    # Keep language consistent on short unoccupied gaps for single-language songs so Qwen does
    # not randomly emit foreign-script single vowels (e.g. Chinese "啊。" inside an English song).
    seg_langs = {str(s.get("language") or "").strip().lower() for s in segments if str(s.get("language") or "").strip()}
    single_lang_hint = primary_lang if (primary_lang and len(seg_langs) <= 1) else None

    windows = find_adlib_review_windows(
        segments=segments, full_vocal_16k=full_vocal_16k, rms_env=rms_env,
        vocal_floor=vocal_floor, duration=duration,
    )
    stats: dict[str, Any] = {
        "status": "no_candidates", "candidate_windows": len(windows),
        "reviewed_windows": 0, "failed_attempts": 0, "recovered_count": 0,
        "unknown_count": 0, "recognition_attempts": 0,
    }
    recovered = []
    for index, window in enumerate(windows, 1):
        check_stop()
        start, end = window["start"], window["end"]
        crop_start, crop_end = (
            (window["crop_start"], window["crop_end"]) if window["occupied"] else (start, end)
        )
        baseline = [s for s in segments if float(s.get("end") or 0) > crop_start and float(s.get("start") or 0) < crop_end]
        sources = [("vocal_stem" if vocal_isolated else "original_mix", full_vocal_16k)]
        if vocal_isolated and original_audio is not None and original_audio.size:
            sources.append(("original_mix", original_audio))
        found, successful = [], False
        log(f"Rà soát giọng hát {index}/{len(windows)} ({start:.2f}–{end:.2f}s)...")
        for variant, source in sources:
            check_stop()
            left, right = int(crop_start * 16000), min(source.size, int(crop_end * 16000))
            if right - left < 1600:
                continue
            stats["recognition_attempts"] += 1
            crop_dur = (right - left) / 16000.0
            win_lang = single_lang_hint if (not window["occupied"] and crop_dur <= 6.0) else None
            try:
                result = asr.transcribe_chunk(
                    _normalize_vocal_rms(source[left:right]), offset_sec=left / 16000.,
                    total_duration=duration, language_iso=win_lang, want_timestamps=True,
                )
                check_stop()
                successful = True
            except Cancelled:
                raise
            except Exception as exc:
                stats["failed_attempts"] += 1
                log(f"Không đọc được vùng {start:.2f}–{end:.2f}s ({type(exc).__name__}); giữ để rà soát.")
                continue
            text = str(result.get("text") or "").strip()
            loop_trimmed = False
            if text and _is_runaway_token_loop(text, duration_sec=crop_dur):
                text, loop_trimmed = sanitize_runaway_repetition(text, duration_sec=crop_dur)
            if not text or NOISE_ONLY_PATTERN.match(text) or HALLUCINATION_PATTERNS.match(text):
                continue

            # If ASR collapsed a multi-syllable vocalise (e.g. "Ah — ah" -> "Ah."), re-recognize
            # each distinct acoustic syllable pulse individually with exact onset/offset timing!
            expanded_words: list[dict[str, Any]] | None = None
            if not window["occupied"] and not loop_trimmed:
                expanded = expand_collapsed_vocalise_with_asr(
                    asr=asr,
                    text=text,
                    full_vocal_16k=full_vocal_16k,
                    start=start,
                    end=end,
                    vocal_floor=vocal_floor,
                    duration=duration,
                    language_iso=win_lang,
                    original_audio=original_audio,
                    should_stop=should_stop,
                )
                if expanded is not None:
                    text, expanded_words = expanded

            valid_words = []
            if expanded_words is not None:
                valid_words = expanded_words
            elif not loop_trimmed:
                for word in result.get("words") or []:
                    interval = _word_interval(word)
                    if interval is None or not str(word.get("word") or "").strip():
                        continue
                    ws, we = interval
                    # Reject offsets outside the actual crop instead of double
                    # mapping, stretching or fabricating word times.
                    if ws < crop_start - .05 or we > crop_end + .05:
                        continue
                    if (ws + we) / 2 < start or (ws + we) / 2 >= end:
                        continue
                    valid_words.append({**word, "start": max(start, ws), "end": min(end, we)})
            if window["occupied"]:
                groups = _new_timed_words(valid_words, baseline)
                if _tokens(text) != _tokens("".join(str(w.get("word") or "") for w in result.get("words") or [])) or not valid_words:
                    # Keep the complete alternative when alignment omitted
                    # syllables; it remains a pending reading, not a new voice.
                    for original in baseline:
                        if original.get("review_status") in {"verified", "edited", "skipped"}:
                            continue
                        candidates = original.setdefault("candidates", [])
                        candidates.append({
                            "candidate_id": f"recovery-{index}-{variant}", "text": text,
                            "source_audio_variant": variant, "status": "pending",
                            "config": {"engine": str(getattr(asr, "repo_id", "asr")), "start": crop_start, "end": crop_end},
                        })
                        original["review_status"] = "needs_review"
                        original["review_reasons"] = list(dict.fromkeys([*(original.get("review_reasons") or []), "candidate_pending"]))
                        original["review_reason"] = format_review_reason(original["review_reasons"])
            else:
                aligned_text = "".join(str(w.get("word") or "") for w in valid_words)
                # Incomplete alignment must never truncate recognized text.
                if _tokens(aligned_text) != _tokens(text):
                    groups = [[]] if (expanded_words is not None or not result.get("words") or valid_words) else []
                else:
                    groups = [valid_words]
            for group in groups:
                phrase_text = "".join(str(w["word"]) for w in group).strip() if group else text
                if not phrase_text or HALLUCINATION_PATTERNS.match(phrase_text):
                    continue
                if _is_runaway_token_loop(phrase_text, duration_sec=crop_dur):
                    phrase_text, loop_trimmed = sanitize_runaway_repetition(phrase_text, duration_sec=crop_dur)
                    if not phrase_text:
                        continue
                    group = []
                for line, ls, le, line_words in _group_words_into_lyric_lines(
                    group, phrase_text, start, end, is_vocalise_chunk=True,
                ):
                    if le <= ls:
                        continue
                    coarse = any(w.get("_partial_word") for w in line_words)
                    eff_timing = (
                        "asr_segment"
                        if (coarse or not line_words)
                        else str(line_words[0].get("timing_source") or result.get("timing_source") or "asr_segment")
                    )
                    seg = _build_segment_record(
                        start=ls, end=le, text=line, text_raw=line,
                        stream_id=DEFAULT_STREAM_ID, primary_lang=primary_lang,
                        chunk_lang=result.get("language"), source_variant=variant,
                        words=[] if coarse else line_words,
                        timing_source=eff_timing,
                    )
                    reasons = ["asr_repetition_loop_trimmed"] if loop_trimmed else ["candidate_pending"]
                    if window["occupied"]:
                        reasons.append("possible_overlap")
                    if coarse:
                        reasons.append("vocal_timing_unresolved")
                    seg.update(vocal_role="unknown", confidence=None, low_confidence=True,
                               review_status="needs_review", review_reasons=reasons,
                               review_reason=format_review_reason(reasons))
                    found.append(seg)
            if found:
                # Further variants are alternative observations, not evidence
                # for additional identical occurrences at the same instant.
                break
        stats["reviewed_windows"] += int(successful)
        recovered.extend(found)
        if not found and not window["occupied"] and not any(
            s.get("event_type") == "unknown_vocal"
            and float(s.get("start") or 0) <= start + .1 and float(s.get("end") or 0) >= end - .1
            for s in baseline
        ):
            reasons = ["suspected_vocal_untranscribed", "vocal_like_heuristic_only"]
            recovered.append({
                "start": start, "end": end, "text": "", "text_raw": "", "text_edited": None,
                "event_type": "unknown_vocal", "vocal_role": "unknown", "is_adlib": None,
                "stream_id": DEFAULT_STREAM_ID, "language": None, "lang": "", "confidence": None,
                "low_confidence": True, "review_status": "needs_review", "review_reasons": reasons,
                "review_reason": format_review_reason(reasons), "timing_source": "acoustic_region",
                "timing_granularity": "region", "source_audio_variant": sources[0][0],
                "words": [], "candidates": [],
            })
            stats["unknown_count"] += 1
    check_stop()
    stats["recovered_count"] = sum(bool(s.get("text")) for s in recovered)
    if windows:
        stats["status"] = "partial" if stats["failed_attempts"] else "ok"
    return recovered, stats



def merge_recovered_vocal_events(
    segments: list[dict[str, Any]], recovered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine recognition passes without collapsing repetitions inside either pass.

    Match exact observations across the two inputs one-to-one, so feeding an existing
    ASR result back through recovery does not add it twice. Never match two entries
    from the recovered list to each other or use fuzzy timing/text containment.
    """

    def observation_key(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(event["start"]), float(event["end"]),
            re.sub(r"\s+", " ", str(event.get("text") or "")).strip().casefold(),
            event.get("stream_id") or DEFAULT_STREAM_ID,
            event.get("vocal_role") or "unknown",
        )

    # Retire a placeholder only when recovered text covers its entire region.
    # Partial/empty retries keep the original uncertainty visible.
    known_spans = sorted((float(s["start"]), float(s["end"])) for s in recovered if str(s.get("text") or "").strip())

    def resolved(event: dict[str, Any]) -> bool:
        if event.get("event_type") != "unknown_vocal":
            return False
        cursor, end = float(event["start"]), float(event["end"])
        for start, stop in known_spans:
            if start > cursor + .1:
                break
            if stop > cursor:
                cursor = stop
            if cursor >= end - .1:
                return True
        return False

    merged = [s for s in segments if not resolved(s)]
    unmatched: dict[tuple[Any, ...], int] = {}
    for original in segments:
        key = observation_key(original)
        unmatched[key] = unmatched.get(key, 0) + 1
    for event in recovered:
        start, end = float(event["start"]), float(event["end"])
        text = str(event.get("text") or "").strip()
        if (not text and event.get("event_type") != "unknown_vocal") or end <= start:
            continue
        key = observation_key(event)
        if unmatched.get(key, 0):
            unmatched[key] -= 1
            continue
        overlaps = [s for s in merged if min(end, float(s["end"])) > max(start, float(s["start"]))]
        used = {s.get("stream_id") or DEFAULT_STREAM_ID for s in overlaps}
        stream_id = event.get("stream_id") or DEFAULT_STREAM_ID
        if stream_id in used:
            lane = 2
            while True:
                stream_id = f"stream_{chr(96 + lane)}" if lane <= 26 else f"stream_{lane}"
                if stream_id not in used:
                    break
                lane += 1
        merged.append({**event, "stream_id": stream_id})
    return merged

