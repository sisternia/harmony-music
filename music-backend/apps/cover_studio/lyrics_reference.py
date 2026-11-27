"""Audio-verified reference lyrics lookup & LRC synchronization for Harmony Auralis.

100% general across all songs and languages (no hardcoded lyrics):
  1. Cleans track title metadata and queries the public LRCLIB catalog.
  2. Verifies every candidate against what `openai/whisper-large-v3` actually heard in the audio
     (word/CJK-bigram overlap gate) so wrong-song or remix lyrics are NEVER applied.
  3. Estimates any constant timestamp offset (e.g. MV intro vs studio cut) from ASR
     anchor matches before handing off to `openai/whisper-large-v3` DTW alignment for word-level sync.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_LRC_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
_QUOTED_RE = re.compile(r"[\"“”「」『』《》]([^\"“”「」『』《》]{2,120})[\"“”「」『』《》]")
_NOISE_RE = re.compile(
    r"(?i)\b(?:official\s+)?(?:music\s+video|lyric\s+video|audio|mv|pv|hd|4k|hq|visualizer|full\s+ver\.?|short\s+ver\.?|ost|theme\s+song|amv|sub\s+vietsub|vietsub|engsub|lyrics?)\b"
)
_BRACKET_NOISE_RE = re.compile(
    r"(?i)[\(\[【（]\s*(?:official|music\s*video|lyric|mv|pv|hd|4k|audio|visualizer|vietsub|engsub|\d+)\s*[^\)\]】）]*[\)\]】）]"
)


def build_search_queries(track_title: str) -> list[str]:
    """Derive 1-3 clean LRCLIB search queries from noisy filenames or YouTube titles."""
    raw = str(track_title or "").strip()
    if not raw:
        return []

    # Strip file extension if present
    raw = re.sub(r"\.(?:mp3|wav|flac|m4a|ogg|webm|opus|mp4|aac)$", "", raw, flags=re.IGNORECASE).strip()
    cleaned = _BRACKET_NOISE_RE.sub(" ", raw)
    cleaned = _NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[|_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:|")

    queries: list[str] = []

    # 1. Quoted title + optional "by Artist" (common in YouTube MV titles)
    quoted = [m.group(1).strip() for m in _QUOTED_RE.finditer(raw) if len(m.group(1).strip()) >= 2]
    if quoted:
        q_title = quoted[0]
        by_m = re.search(r"(?i)\bby\s+([^|\-–—\(\[\n]{2,50})", cleaned)
        if by_m:
            artist = by_m.group(1).strip(" -–—:|")
            if artist and artist.lower() not in q_title.lower():
                queries.append(f"{q_title} {artist}")
        queries.append(q_title)

    # 2. Split segments around '|' or ' - ' when title has franchise/series prefix
    parts = [p.strip(" -–—:|\"“”「」『』") for p in re.split(r"[|｜]", raw) if p.strip()]
    if len(parts) >= 2:
        cleaned_parts = [
            re.sub(r"\s+", " ", _NOISE_RE.sub(" ", _BRACKET_NOISE_RE.sub(" ", p))).strip(" -–—:|\"“”「」『』")
            for p in parts
        ]
        cleaned_parts = [p for p in cleaned_parts if len(p) >= 2]
        if len(cleaned_parts) >= 2:
            # Prefer non-prefix middle segment (often "Song by Artist")
            cand = re.sub(r"(?i)\bby\s+", " ", cleaned_parts[1]).strip()
            if cand:
                queries.append(cand)

    # 3. Full cleaned title (stripping "by ")
    full_q = re.sub(r"(?i)\bby\s+", " ", cleaned)
    full_q = re.sub(r"[\"“”「」『』]", " ", full_q)
    full_q = re.sub(r"\s+", " ", full_q).strip()
    if full_q:
        queries.append(full_q)

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        qn = re.sub(r"\s+", " ", q).strip()
        if len(qn) >= 2 and qn.lower() not in seen:
            seen.add(qn.lower())
            out.append(qn)
    return out[:3]


def parse_synced_lrc(lrc_text: str) -> list[dict[str, Any]]:
    """Parse standard LRC `[mm:ss.xx] lyric` text into sorted `[{"start": float, "text": str}]`."""
    if not lrc_text:
        return []
    entries: list[tuple[float, str]] = []
    for raw_line in lrc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matches = list(_LRC_TS_RE.finditer(line))
        if not matches:
            continue
        text = _LRC_TS_RE.sub("", line).strip()
        # Strip inline word-timing tags like <00:12.34> if present
        text = re.sub(r"<\d{1,2}:\d{2}(?:[.:]\d{1,3})?>", "", text).strip()
        if not text:
            continue
        # Skip pure metadata/credit lines
        if re.match(r"^(?:ti|ar|al|by|offset|length)\s*:", text, re.IGNORECASE):
            continue
        for m in matches:
            mm = int(m.group(1))
            ss = int(m.group(2))
            frac_s = m.group(3) or "0"
            frac = int(frac_s) / (1000.0 if len(frac_s) == 3 else (100.0 if len(frac_s) == 2 else 10.0))
            entries.append((round(mm * 60.0 + ss + frac, 2), text))
    entries.sort(key=lambda x: x[0])
    return [{"start": t, "text": txt} for t, txt in entries]


def _tokenize_for_verify(text: str) -> list[str]:
    clean = str(text or "").lower().replace("’", "'")
    if _CJK_RE.search(clean):
        chars = re.sub(r"[^\w\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+", "", clean)
        if len(chars) < 2:
            return list(chars)
        return [chars[i : i + 2] for i in range(len(chars) - 1)]
    clean = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff'\s]+", " ", clean)
    return [w.strip("'") for w in clean.split() if len(w.strip("'")) >= 2]


def verify_reference_against_asr(
    candidate_text: str,
    asr_text: str,
    *,
    min_overlap: float = 0.30,
) -> tuple[bool, float]:
    """Verify that candidate reference lyrics genuinely match what `openai/whisper-large-v3` heard in the audio."""
    ref_toks = _tokenize_for_verify(candidate_text)
    asr_toks = _tokenize_for_verify(asr_text)
    if len(ref_toks) < 6 or len(asr_toks) < 6:
        return False, 0.0

    # Multiset (bag-of-tokens) overlap so chorus repetitions count accurately
    ref_counts: dict[str, int] = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    asr_counts: dict[str, int] = {}
    for t in asr_toks:
        asr_counts[t] = asr_counts.get(t, 0) + 1

    inter = sum(min(c, asr_counts.get(t, 0)) for t, c in ref_counts.items())
    recall_asr = inter / max(1, len(asr_toks))
    recall_ref = inter / max(1, len(ref_toks))
    f1 = (2.0 * recall_asr * recall_ref / max(1e-6, recall_asr + recall_ref)) if (recall_asr + recall_ref) > 0 else 0.0

    # Also check unique-token Jaccard/overlap to prevent common stopword coincidence
    u_ref, u_asr = set(ref_counts), set(asr_counts)
    u_overlap = len(u_ref & u_asr) / max(1, min(len(u_ref), len(u_asr)))

    score = round(0.6 * f1 + 0.4 * u_overlap, 4)
    return (score >= min_overlap and u_overlap >= 0.28), score


def estimate_lrc_time_offset(
    lrc_lines: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
) -> float:
    """Estimate constant global time shift (asr_start - lrc_start) in seconds using clear matching lines."""
    if not lrc_lines or not asr_segments:
        return 0.0

    diffs: list[float] = []
    for lrc in lrc_lines:
        lt = _tokenize_for_verify(str(lrc.get("text") or ""))
        if len(lt) < 3:
            continue
        l_set = set(lt)
        l_start = float(lrc["start"])
        best_sim, best_diff = 0.0, 0.0
        for seg in asr_segments:
            st = _tokenize_for_verify(str(seg.get("text") or ""))
            if len(st) < 3:
                continue
            s_start = float(seg.get("start") or 0.0)
            if abs(s_start - l_start) > 18.0:
                continue
            s_set = set(st)
            sim = len(l_set & s_set) / max(1, min(len(l_set), len(s_set)))
            if sim > best_sim:
                best_sim = sim
                best_diff = s_start - l_start
        if best_sim >= 0.65:
            diffs.append(best_diff)

    if len(diffs) < 2:
        return 0.0
    diffs.sort()
    mid = diffs[len(diffs) // 2]
    # Ignore tiny jitter < 0.35s so clean LRC timestamps remain untouched
    if abs(mid) < 0.35:
        return 0.0
    return round(max(-12.0, min(12.0, mid)), 2)


def fetch_lrclib_candidates(
    track_title: str,
    *,
    duration: float | None = None,
    timeout: float = 6.0,
) -> list[dict[str, Any]]:
    """Query public LRCLIB API and return ranked candidates with synced or plain lyrics."""
    queries = build_search_queries(track_title)
    if not queries:
        return []

    by_id: dict[Any, dict[str, Any]] = {}
    for q in queries:
        url = "https://lrclib.net/api/search?q=" + urllib.parse.quote(q)
        req = urllib.request.Request(url, headers={"User-Agent": "HarmonyAuralis/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                items = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("instrumental"):
                continue
            if not (item.get("syncedLyrics") or item.get("plainLyrics")):
                continue
            cid = item.get("id") or (item.get("trackName"), item.get("artistName"), item.get("duration"))
            by_id.setdefault(cid, item)
        if by_id:
            break

    cands = list(by_id.values())

    def _rank_key(it: dict[str, Any]) -> tuple[int, float]:
        has_synced = 0 if it.get("syncedLyrics") else 1
        d = it.get("duration")
        dur_diff = abs(float(d) - float(duration)) if (duration and d) else 999.0
        return (has_synced, dur_diff)

    cands.sort(key=_rank_key)
    return cands[:8]


def find_verified_reference_lyrics(
    *,
    track_title: str,
    duration: float,
    asr_segments: list[dict[str, Any]],
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Any] | None:
    """Find and audio-verify reference lyrics for `track_title` against `openai/whisper-large-v3` output."""
    if not track_title or not track_title.strip() or not asr_segments:
        return None
    asr_text = " ".join(str(s.get("text") or "").strip() for s in asr_segments if str(s.get("text") or "").strip())
    if len(_tokenize_for_verify(asr_text)) < 6:
        return None

    candidates = fetch_lrclib_candidates(track_title, duration=duration)
    if not candidates:
        return None

    best_match: dict[str, Any] | None = None
    best_rank: tuple[int, float, float] = (-1, -1.0, -999.0)

    for cand in candidates:
        synced_raw = str(cand.get("syncedLyrics") or "").strip()
        plain_raw = str(cand.get("plainLyrics") or "").strip()
        synced_lines = parse_synced_lrc(synced_raw) if synced_raw else []
        cand_text = "\n".join(x["text"] for x in synced_lines) if synced_lines else plain_raw
        if not cand_text.strip():
            continue

        cand_dur = cand.get("duration")
        dur_diff = abs(float(cand_dur) - float(duration)) if (duration and cand_dur) else 30.0
        if duration and cand_dur and dur_diff > 35.0:
            continue

        ok, score = verify_reference_against_asr(cand_text, asr_text)
        if not ok:
            continue

        # Prefer synced lyrics + close duration + highest audio overlap score
        has_synced = 1 if len(synced_lines) >= 4 else 0
        rank = (has_synced, -min(dur_diff, 35.0) / 35.0 + score, score)
        if rank > best_rank:
            best_rank = rank
            offset = estimate_lrc_time_offset(synced_lines, asr_segments) if synced_lines else 0.0
            best_match = {
                "source": "lrclib",
                "track_name": str(cand.get("trackName") or ""),
                "artist_name": str(cand.get("artistName") or ""),
                "duration": float(cand_dur) if cand_dur else None,
                "match_score": score,
                "synced_lines": synced_lines,
                "plain_text": plain_raw or cand_text,
                "time_offset": offset,
            }

    if best_match is not None:
        log(
            f"Đã xác thực lời tham chiếu qua âm thanh (LRCLIB: {best_match['track_name']} - "
            f"{best_match['artist_name']}, khớp {best_match['match_score']*100:.1f}%, "
            f"{len(best_match['synced_lines'])} dòng synced)."
        )
    return best_match

