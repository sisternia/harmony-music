"""Smart MIDI Repair engine for Harmony Auralis Cover Studio.

Analyzes piano cover MIDI (cover.mid) and source MIDI (source.mid) for common
transcription / generation artifacts, previews non-destructive repairs with
note-level diffs and musical quality scores, and writes clean metadata-preserving
MIDI files via symusic.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MIN_PIANO_PITCH = 21  # A0
MAX_PIANO_PITCH = 108  # C8
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def pitch_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{(pitch // 12) - 1}"


@dataclass
class RepairOptions:
    remove_ghost_notes: bool = True
    min_duration_ms: float = 35.0
    min_velocity: int = 18

    fix_overlaps: bool = True
    duplicate_window_ms: float = 30.0
    rearticulate_gap_ms: float = 15.0

    clamp_piano_range: bool = True

    fix_hand_span: bool = True
    split_pitch: int = 60
    max_hand_span: int = 15
    max_polyphony: int = 10

    smart_quantize: bool = True
    quantize_strength: float = 0.5
    grid_subdivision: str = "16th"  # "8th" | "16th" | "triplet"

    normalize_velocity: bool = True
    vel_min: int = 34
    vel_max: int = 114
    melody_boost: int = 8

    clean_pedals: bool = True
    max_pedal_sec: float = 4.0
    repedal_on_chord: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RepairOptions:
        data = raw or {}
        sub = str(data.get("grid_subdivision", "16th"))
        if sub not in ("8th", "16th", "triplet"):
            sub = "16th"
        vel_min = int(np.clip(int(data.get("vel_min", 34)), 10, 90))
        vel_max = int(np.clip(int(data.get("vel_max", 114)), max(vel_min + 10, 40), 127))
        return cls(
            remove_ghost_notes=bool(data.get("remove_ghost_notes", True)),
            min_duration_ms=float(np.clip(float(data.get("min_duration_ms", 35.0)), 10.0, 140.0)),
            min_velocity=int(np.clip(int(data.get("min_velocity", 18)), 1, 60)),
            fix_overlaps=bool(data.get("fix_overlaps", True)),
            duplicate_window_ms=float(np.clip(float(data.get("duplicate_window_ms", 30.0)), 5.0, 90.0)),
            rearticulate_gap_ms=float(np.clip(float(data.get("rearticulate_gap_ms", 15.0)), 0.0, 50.0)),
            clamp_piano_range=bool(data.get("clamp_piano_range", True)),
            fix_hand_span=bool(data.get("fix_hand_span", True)),
            split_pitch=int(np.clip(int(data.get("split_pitch", 60)), 48, 72)),
            max_hand_span=int(np.clip(int(data.get("max_hand_span", 15)), 12, 19)),
            max_polyphony=int(np.clip(int(data.get("max_polyphony", 10)), 4, 14)),
            smart_quantize=bool(data.get("smart_quantize", True)),
            quantize_strength=float(np.clip(float(data.get("quantize_strength", 0.5)), 0.0, 1.0)),
            grid_subdivision=sub,
            normalize_velocity=bool(data.get("normalize_velocity", True)),
            vel_min=vel_min,
            vel_max=vel_max,
            melody_boost=int(np.clip(int(data.get("melody_boost", 8)), 0, 28)),
            clean_pedals=bool(data.get("clean_pedals", True)),
            max_pedal_sec=float(np.clip(float(data.get("max_pedal_sec", 4.0)), 1.5, 10.0)),
            repedal_on_chord=bool(data.get("repedal_on_chord", True)),
        )


@dataclass
class _NoteItem:
    orig_idx: int
    onset: float
    duration: float
    pitch: int
    velocity: int
    track_idx: int = 0
    removed: bool = False
    reasons: list[str] = field(default_factory=list)
    orig_onset: float = 0.0
    orig_duration: float = 0.0
    orig_pitch: int = 0
    orig_velocity: int = 0

    def __post_init__(self) -> None:
        self.orig_onset = self.onset
        self.orig_duration = self.duration
        self.orig_pitch = self.pitch
        self.orig_velocity = self.velocity


def _load_midi_elements(
    path: Path,
) -> tuple[list[_NoteItem], list[tuple[float, float]], list[float], list[tuple[float, str]]]:
    """Load non-drum notes, pedals, tempo-derived beats, and chord markers in seconds."""
    from symusic import Score

    score_ticks = Score(str(path))
    score = score_ticks.to("second")
    items: list[_NoteItem] = []
    pedals: list[tuple[float, float]] = []

    raw_notes: list[tuple[float, float, int, int, int]] = []
    for t_idx, track in enumerate(score.tracks):
        if track.is_drum:
            continue
        for n in track.notes:
            onset = max(0.0, float(n.time))
            dur = max(0.005, float(n.duration))
            raw_notes.append((onset, dur, int(n.pitch), int(np.clip(int(n.velocity), 1, 127)), t_idx))
        for p in track.pedals:
            start = max(0.0, float(p.time))
            end = max(start, float(p.time + p.duration))
            if end > start:
                pedals.append((start, end))

    raw_notes.sort(key=lambda r: (r[0], r[2], -r[3]))
    for idx, (onset, dur, pitch, vel, t_idx) in enumerate(raw_notes):
        items.append(_NoteItem(orig_idx=idx, onset=onset, duration=dur, pitch=pitch, velocity=vel, track_idx=t_idx))

    pedals.sort()
    end_sec = max((n.onset + n.duration for n in items), default=0.0)
    if pedals:
        end_sec = max(end_sec, max(p[1] for p in pedals))

    # Build beat times from MIDI tempo map as fallback when source beats are absent
    beats = _beats_from_score(score, end_sec)
    chords = _chords_from_score(score)
    return items, pedals, beats, chords


def _beats_from_score(score, end_sec: float) -> list[float]:
    if end_sec <= 0:
        return []
    tempos = sorted(((max(0.0, float(t.time)), max(20.0, min(300.0, float(t.qpm)))) for t in score.tempos), key=lambda x: x[0])
    if not tempos or tempos[0][0] > 0.05:
        tempos.insert(0, (0.0, tempos[0][1] if tempos else 120.0))
    beats: list[float] = []
    for i, (t_start, qpm) in enumerate(tempos):
        t_end = tempos[i + 1][0] if i + 1 < len(tempos) else end_sec + 1.0
        step = 60.0 / qpm
        cur = t_start
        while cur < t_end - 1e-4 and cur <= end_sec + 0.5:
            beats.append(round(cur, 4))
            cur += step
    return beats


def _chords_from_score(score) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for marker in score.markers:
        text = str(marker.text).strip()
        if text and text != "N":
            out.append((round(float(marker.time), 3), text))
    out.sort(key=lambda x: x[0])
    return out


def _build_subdivision_grid(beats: list[float], subdivision: str, end_sec: float) -> np.ndarray:
    if len(beats) < 2:
        if end_sec <= 0:
            return np.zeros((0,), dtype=np.float64)
        beats = list(np.arange(0.0, end_sec + 1.0, 0.5))
    div = {"8th": 2, "16th": 4, "triplet": 3}.get(subdivision, 4)
    grid: list[float] = []
    for i in range(len(beats) - 1):
        a, b = beats[i], beats[i + 1]
        if b <= a + 1e-4:
            continue
        step = (b - a) / div
        for k in range(div):
            grid.append(a + k * step)
    if beats:
        last_step = (beats[-1] - beats[-2]) / div if len(beats) >= 2 and beats[-1] > beats[-2] else 0.125
        cur = beats[-1]
        while cur <= end_sec + last_step:
            grid.append(cur)
            cur += max(0.05, last_step)
    return np.asarray(sorted(set(round(g, 5) for g in grid)), dtype=np.float64)


def _compute_releases(
    notes: list[_NoteItem],
    pedals: list[tuple[float, float]],
) -> list[list[float | int]]:
    """Convert active notes to CoverView [onset, duration, pitch, velocity, release] rows."""
    sorted_pedals = sorted(pedals)
    starts = np.asarray([p[0] for p in sorted_pedals], dtype=np.float64) if sorted_pedals else np.zeros((0,), dtype=np.float64)
    out: list[list[float | int]] = []
    for n in sorted((x for x in notes if not x.removed), key=lambda x: (x.onset, x.pitch)):
        release = n.onset + n.duration
        if len(starts) > 0:
            i = int(np.searchsorted(starts, release, side="right")) - 1
            if i >= 0 and sorted_pedals[i][0] <= release < sorted_pedals[i][1]:
                release = sorted_pedals[i][1]
        out.append([round(n.onset, 3), round(n.duration, 3), int(n.pitch), int(n.velocity), round(release, 3)])
    return out


def _group_simultaneous(notes: list[_NoteItem], window_sec: float = 0.035) -> list[list[_NoteItem]]:
    active = sorted((n for n in notes if not n.removed), key=lambda x: (x.onset, x.pitch))
    if not active:
        return []
    groups: list[list[_NoteItem]] = [[active[0]]]
    for n in active[1:]:
        if n.onset - groups[-1][0].onset <= window_sec:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def analyze_elements(
    notes: list[_NoteItem],
    pedals: list[tuple[float, float]],
    beats: list[float],
    chords: list[tuple[float, str]],
    options: RepairOptions,
) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    """Analyze notes and pedals, returning (issue_counts, quality_scores, issue_items)."""
    active = [n for n in notes if not n.removed]
    total = len(active)
    issues: list[dict[str, Any]] = []
    counts = {
        "ghost_notes": 0,
        "overlapping_notes": 0,
        "out_of_range": 0,
        "hand_span": 0,
        "timing_jitter": 0,
        "velocity_spikes": 0,
        "pedal_glitches": 0,
    }
    if total == 0:
        scores = {"overall": 100, "cleanliness": 100, "playability": 100, "timing": 100, "expression": 100}
        return counts, scores, issues

    min_dur_sec = options.min_duration_ms / 1000.0
    dup_win_sec = options.duplicate_window_ms / 1000.0
    end_sec = max(n.onset + n.duration for n in active)
    grid = _build_subdivision_grid(beats, options.grid_subdivision, end_sec)

    # 1. Ghost notes, out-of-range, velocity outliers
    for n in active:
        if n.pitch < MIN_PIANO_PITCH or n.pitch > MAX_PIANO_PITCH:
            counts["out_of_range"] += 1
            issues.append(
                {
                    "id": f"range-{n.orig_idx}",
                    "type": "out_of_range",
                    "severity": "high",
                    "time": round(n.onset, 3),
                    "pitch": n.pitch,
                    "note_idx": n.orig_idx,
                    "label": f"{pitch_name(n.pitch)} ngoài dải 88 phím (A0–C8)",
                }
            )
        if (n.duration < min_dur_sec and n.velocity < max(options.min_velocity + 28, 46)) or n.velocity < options.min_velocity:
            counts["ghost_notes"] += 1
            issues.append(
                {
                    "id": f"ghost-{n.orig_idx}",
                    "type": "ghost_notes",
                    "severity": "medium",
                    "time": round(n.onset, 3),
                    "pitch": n.pitch,
                    "note_idx": n.orig_idx,
                    "label": f"Nốt rác {pitch_name(n.pitch)} ({round(n.duration * 1000)}ms · vel {n.velocity})",
                }
            )
        if n.velocity < max(16, options.vel_min - 10) or n.velocity > min(125, options.vel_max + 8):
            counts["velocity_spikes"] += 1
            if counts["velocity_spikes"] <= 40:
                issues.append(
                    {
                        "id": f"vel-{n.orig_idx}",
                        "type": "velocity_spikes",
                        "severity": "low",
                        "time": round(n.onset, 3),
                        "pitch": n.pitch,
                        "note_idx": n.orig_idx,
                        "label": f"Lực đánh cực đoan {pitch_name(n.pitch)} (vel {n.velocity})",
                    }
                )

    # 2. Same-pitch duplicates & overlaps
    by_pitch: dict[int, list[_NoteItem]] = {}
    for n in active:
        by_pitch.setdefault(n.pitch, []).append(n)
    for pitch, plist in by_pitch.items():
        plist.sort(key=lambda x: x.onset)
        for i in range(len(plist) - 1):
            cur, nxt = plist[i], plist[i + 1]
            if nxt.onset - cur.onset <= dup_win_sec:
                counts["overlapping_notes"] += 1
                issues.append(
                    {
                        "id": f"dup-{nxt.orig_idx}",
                        "type": "overlapping_notes",
                        "severity": "high",
                        "time": round(nxt.onset, 3),
                        "pitch": pitch,
                        "note_idx": nxt.orig_idx,
                        "label": f"Nốt trùng {pitch_name(pitch)} (cách {round((nxt.onset - cur.onset) * 1000)}ms)",
                    }
                )
            elif cur.onset + cur.duration > nxt.onset + 0.005:
                counts["overlapping_notes"] += 1
                issues.append(
                    {
                        "id": f"ovl-{cur.orig_idx}",
                        "type": "overlapping_notes",
                        "severity": "medium",
                        "time": round(cur.onset, 3),
                        "pitch": pitch,
                        "note_idx": cur.orig_idx,
                        "label": f"Chồng phím {pitch_name(pitch)} tràn sang nốt kế tiếp",
                    }
                )

    # 3. Hand span & polyphony bursts
    for group in _group_simultaneous(active, window_sec=0.035):
        lh = [n for n in group if n.pitch < options.split_pitch]
        rh = [n for n in group if n.pitch >= options.split_pitch]
        lh_span = (max(n.pitch for n in lh) - min(n.pitch for n in lh)) if len(lh) >= 2 else 0
        rh_span = (max(n.pitch for n in rh) - min(n.pitch for n in rh)) if len(rh) >= 2 else 0
        if lh_span > options.max_hand_span:
            counts["hand_span"] += 1
            issues.append(
                {
                    "id": f"span-lh-{group[0].orig_idx}",
                    "type": "hand_span",
                    "severity": "high",
                    "time": round(group[0].onset, 3),
                    "pitch": min(n.pitch for n in lh),
                    "note_idx": group[0].orig_idx,
                    "label": f"Tay trái giãn rộng {lh_span} nửa cung ({pitch_name(min(n.pitch for n in lh))}–{pitch_name(max(n.pitch for n in lh))})",
                }
            )
        if rh_span > options.max_hand_span:
            counts["hand_span"] += 1
            issues.append(
                {
                    "id": f"span-rh-{group[-1].orig_idx}",
                    "type": "hand_span",
                    "severity": "high",
                    "time": round(group[0].onset, 3),
                    "pitch": max(n.pitch for n in rh),
                    "note_idx": group[-1].orig_idx,
                    "label": f"Tay phải giãn rộng {rh_span} nửa cung ({pitch_name(min(n.pitch for n in rh))}–{pitch_name(max(n.pitch for n in rh))})",
                }
            )
        if len(group) > options.max_polyphony:
            counts["hand_span"] += 1
            issues.append(
                {
                    "id": f"poly-{group[0].orig_idx}",
                    "type": "hand_span",
                    "severity": "high",
                    "time": round(group[0].onset, 3),
                    "pitch": group[-1].pitch,
                    "note_idx": group[0].orig_idx,
                    "label": f"Quá nhiều nốt đồng thời ({len(group)} nốt > {options.max_polyphony} ngón)",
                }
            )

    # 4. Timing jitter against beat subdivision grid
    if len(grid) >= 2:
        onsets = np.asarray([n.onset for n in active], dtype=np.float64)
        idx = np.clip(np.searchsorted(grid, onsets), 0, len(grid) - 1)
        prev_idx = np.maximum(0, idx - 1)
        d1 = np.abs(onsets - grid[idx])
        d2 = np.abs(onsets - grid[prev_idx])
        nearest_dist = np.minimum(d1, d2)
        step_est = float(np.median(np.diff(grid))) if len(grid) > 2 else 0.125
        jitter_lo = 0.018
        jitter_hi = min(0.065, step_est * 0.42)
        jitter_mask = (nearest_dist > jitter_lo) & (nearest_dist < jitter_hi)
        jitter_indices = np.flatnonzero(jitter_mask)
        counts["timing_jitter"] = int(len(jitter_indices))
        for j in jitter_indices[:30]:
            n = active[int(j)]
            issues.append(
                {
                    "id": f"jit-{n.orig_idx}",
                    "type": "timing_jitter",
                    "severity": "low",
                    "time": round(n.onset, 3),
                    "pitch": n.pitch,
                    "note_idx": n.orig_idx,
                    "label": f"Lệch lưới nhịp {round(float(nearest_dist[j]) * 1000)}ms ({pitch_name(n.pitch)})",
                }
            )

    # 5. Sustain pedal glitches
    chord_times = [c[0] for c in chords]
    for p_idx, (p_start, p_end) in enumerate(pedals):
        dur = p_end - p_start
        if dur < 0.08:
            counts["pedal_glitches"] += 1
            issues.append(
                {
                    "id": f"ped-short-{p_idx}",
                    "type": "pedal_glitches",
                    "severity": "low",
                    "time": round(p_start, 3),
                    "pitch": 48,
                    "note_idx": None,
                    "label": f"Pedal chớp nhiễu ({round(dur * 1000)}ms)",
                }
            )
        elif dur > options.max_pedal_sec:
            counts["pedal_glitches"] += 1
            issues.append(
                {
                    "id": f"ped-long-{p_idx}",
                    "type": "pedal_glitches",
                    "severity": "medium",
                    "time": round(p_start, 3),
                    "pitch": 48,
                    "note_idx": None,
                    "label": f"Pedal giữ quá dài ({dur:.1f}s gây ù tiếng)",
                }
            )
        elif options.repedal_on_chord and chord_times:
            crossed = [ct for ct in chord_times if p_start + 0.25 < ct < p_end - 0.25]
            if len(crossed) >= 2:
                counts["pedal_glitches"] += 1
                issues.append(
                    {
                        "id": f"ped-chord-{p_idx}",
                        "type": "pedal_glitches",
                        "severity": "medium",
                        "time": round(crossed[0], 3),
                        "pitch": 48,
                        "note_idx": None,
                        "label": f"Pedal giữ qua {len(crossed)} lần đổi hợp âm",
                    }
                )

    # Sort issues chronologically and cap list for snappy UI payload
    issues.sort(key=lambda x: (x["time"], 0 if x["severity"] == "high" else 1 if x["severity"] == "medium" else 2))

    # Compute transparent 0-100 quality metrics
    clean_ratio = (counts["ghost_notes"] * 1.4 + counts["overlapping_notes"] * 1.6 + counts["out_of_range"] * 2.5) / max(total, 1)
    cleanliness = int(round(max(20.0, min(100.0, 100.0 - clean_ratio * 220.0))))

    chord_count = max(len(_group_simultaneous(active, 0.035)), 1)
    play_ratio = counts["hand_span"] / chord_count
    playability = int(round(max(25.0, min(100.0, 100.0 - play_ratio * 240.0))))

    jitter_ratio = counts["timing_jitter"] / max(total, 1)
    timing = int(round(max(30.0, min(100.0, 100.0 - jitter_ratio * 135.0))))

    pedal_denom = max(len(pedals), 8)
    expr_penalty = (counts["velocity_spikes"] / max(total, 1)) * 90.0 + (counts["pedal_glitches"] / pedal_denom) * 45.0
    expression = int(round(max(30.0, min(100.0, 100.0 - expr_penalty))))

    overall = int(round(cleanliness * 0.30 + playability * 0.30 + timing * 0.20 + expression * 0.20))
    scores = {
        "overall": overall,
        "cleanliness": cleanliness,
        "playability": playability,
        "timing": timing,
        "expression": expression,
    }
    return counts, scores, issues[:220]


def _repair_elements(
    orig_notes: list[_NoteItem],
    orig_pedals: list[tuple[float, float]],
    beats: list[float],
    chords: list[tuple[float, str]],
    options: RepairOptions,
    manual_deletes: set[int] | None = None,
    manual_keeps: set[int] | None = None,
    manual_updates: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[_NoteItem], list[tuple[float, float]], dict[str, int]]:
    """Run all enabled repair stages non-destructively on cloned notes & pedals."""
    manual_deletes = manual_deletes or set()
    manual_keeps = manual_keeps or set()
    manual_updates = manual_updates or {}

    notes = [
        _NoteItem(
            orig_idx=n.orig_idx,
            onset=n.onset,
            duration=n.duration,
            pitch=n.pitch,
            velocity=n.velocity,
            track_idx=n.track_idx,
        )
        for n in orig_notes
    ]
    pedals = list(orig_pedals)
    actions = {
        "ghosts_removed": 0,
        "overlaps_fixed": 0,
        "range_clamped": 0,
        "hand_spans_fixed": 0,
        "notes_quantized": 0,
        "velocities_balanced": 0,
        "pedals_cleaned": 0,
        "manual_edits": 0,
    }

    # 1. Clamp out-of-range notes into 88-key piano bounds (A0..C8)
    if options.clamp_piano_range:
        for n in notes:
            if n.pitch < MIN_PIANO_PITCH:
                while n.pitch < MIN_PIANO_PITCH:
                    n.pitch += 12
                n.reasons.append("range_clamped")
                actions["range_clamped"] += 1
            elif n.pitch > MAX_PIANO_PITCH:
                while n.pitch > MAX_PIANO_PITCH:
                    n.pitch -= 12
                n.reasons.append("range_clamped")
                actions["range_clamped"] += 1

    # 2. Remove ghost / stray micro-notes (preserving protected manual_keeps)
    if options.remove_ghost_notes:
        min_dur_sec = options.min_duration_ms / 1000.0
        for n in notes:
            if n.orig_idx in manual_keeps:
                continue
            is_short_weak = n.duration < min_dur_sec and n.velocity < max(options.min_velocity + 28, 46)
            is_inaudible = n.velocity < options.min_velocity
            if is_short_weak or is_inaudible:
                n.removed = True
                n.reasons.append("ghost_removed")
                actions["ghosts_removed"] += 1

    # 3. Fix same-pitch duplicates & overlaps
    if options.fix_overlaps:
        dup_win_sec = options.duplicate_window_ms / 1000.0
        gap_sec = options.rearticulate_gap_ms / 1000.0
        by_pitch: dict[int, list[_NoteItem]] = {}
        for n in notes:
            if not n.removed:
                by_pitch.setdefault(n.pitch, []).append(n)
        for plist in by_pitch.values():
            plist.sort(key=lambda x: (x.onset, -x.velocity))
            kept: list[_NoteItem] = []
            for n in plist:
                if kept and (n.onset - kept[-1].onset) <= dup_win_sec:
                    prev = kept[-1]
                    # Keep the louder/longer note unless one is explicitly manual-kept
                    if n.orig_idx in manual_keeps and prev.orig_idx not in manual_keeps:
                        prev.removed = True
                        prev.reasons.append("duplicate_removed")
                        kept[-1] = n
                    elif n.orig_idx not in manual_keeps:
                        # Absorb duration if new note was longer
                        prev.duration = max(prev.duration, (n.onset + n.duration) - prev.onset)
                        prev.velocity = max(prev.velocity, n.velocity)
                        n.removed = True
                        n.reasons.append("duplicate_removed")
                    else:
                        kept.append(n)
                    actions["overlaps_fixed"] += 1
                    continue
                if kept:
                    prev = kept[-1]
                    max_end = n.onset - gap_sec
                    if prev.onset + prev.duration > max_end + 0.002:
                        new_dur = max(0.025, max_end - prev.onset)
                        if new_dur < prev.duration - 0.002:
                            prev.duration = new_dur
                            if "overlap_trimmed" not in prev.reasons:
                                prev.reasons.append("overlap_trimmed")
                            actions["overlaps_fixed"] += 1
                kept.append(n)

    # 4. Smart Quantize gently toward beat/subdivision grid
    if options.smart_quantize and options.quantize_strength > 0.01:
        active = [n for n in notes if not n.removed]
        if active:
            end_sec = max(n.onset + n.duration for n in active)
            grid = _build_subdivision_grid(beats, options.grid_subdivision, end_sec)
            if len(grid) >= 2:
                step_est = float(np.median(np.diff(grid))) if len(grid) > 2 else 0.125
                max_pull = min(0.065, step_est * 0.42)
                for n in active:
                    idx = int(np.searchsorted(grid, n.onset))
                    cands = []
                    if idx < len(grid):
                        cands.append(float(grid[idx]))
                    if idx > 0:
                        cands.append(float(grid[idx - 1]))
                    nearest = min(cands, key=lambda g: abs(g - n.onset))
                    diff = nearest - n.onset
                    if 0.004 <= abs(diff) <= max_pull:
                        n.onset = max(0.0, n.onset + diff * options.quantize_strength)
                        n.reasons.append("quantized")
                        actions["notes_quantized"] += 1

    # 5. Fix unplayable hand spans & excess simultaneous polyphony (preserving melody top & bass bottom)
    if options.fix_hand_span:
        for group in _group_simultaneous(notes, window_sec=0.035):
            if len(group) < 2:
                continue
            lh = sorted((n for n in group if n.pitch < options.split_pitch), key=lambda x: x.pitch)
            rh = sorted((n for n in group if n.pitch >= options.split_pitch), key=lambda x: x.pitch)
            used_pitches = {n.pitch for n in group}

            # Left hand: anchor on lowest bass note
            if len(lh) >= 2:
                bass = lh[0].pitch
                for n in lh[1:]:
                    if n.pitch - bass > options.max_hand_span:
                        folded = n.pitch - 12
                        if bass <= folded <= bass + options.max_hand_span and folded not in used_pitches:
                            used_pitches.discard(n.pitch)
                            n.pitch = folded
                            used_pitches.add(folded)
                            n.reasons.append("hand_span_folded")
                            actions["hand_spans_fixed"] += 1
                        elif n.orig_idx not in manual_keeps:
                            n.removed = True
                            n.reasons.append("hand_span_trimmed")
                            actions["hand_spans_fixed"] += 1

            # Right hand: anchor on highest melody note
            if len(rh) >= 2:
                melody = rh[-1].pitch
                for n in rh[:-1]:
                    if melody - n.pitch > options.max_hand_span:
                        folded = n.pitch + 12
                        if melody - options.max_hand_span <= folded <= melody and folded not in used_pitches:
                            used_pitches.discard(n.pitch)
                            n.pitch = folded
                            used_pitches.add(folded)
                            n.reasons.append("hand_span_folded")
                            actions["hand_spans_fixed"] += 1
                        elif n.orig_idx not in manual_keeps:
                            n.removed = True
                            n.reasons.append("hand_span_trimmed")
                            actions["hand_spans_fixed"] += 1

            # Enforce max simultaneous polyphony (protect bass + top melody)
            alive = sorted((n for n in group if not n.removed), key=lambda x: x.pitch)
            if len(alive) > options.max_polyphony:
                protected = {alive[0].orig_idx, alive[-1].orig_idx} | manual_keeps
                removable = [n for n in alive if n.orig_idx not in protected]
                # Prefer removing lowest-velocity inner doublings first
                removable.sort(key=lambda x: (x.velocity, x.duration))
                excess = len(alive) - options.max_polyphony
                for n in removable[:excess]:
                    n.removed = True
                    n.reasons.append("polyphony_trimmed")
                    actions["hand_spans_fixed"] += 1

    # 6. Balance dynamics / velocity & lift top melody
    if options.normalize_velocity:
        active = [n for n in notes if not n.removed]
        if active:
            vels = np.asarray([n.velocity for n in active], dtype=np.float64)
            p5, p95 = float(np.percentile(vels, 5)), float(np.percentile(vels, 95))
            span = max(p95 - p5, 1.0)
            # Detect melodic top-line notes per 60ms slice
            melody_ids: set[int] = set()
            for group in _group_simultaneous(active, window_sec=0.06):
                top = max(group, key=lambda x: x.pitch)
                if top.pitch >= options.split_pitch - 2:
                    melody_ids.add(top.orig_idx)

            for n in active:
                norm = (n.velocity - p5) / span
                norm = max(0.0, min(1.0, norm))
                target_v = options.vel_min + norm * (options.vel_max - options.vel_min)
                # Blend 70% toward normalized range to retain organic phrasing
                blended = 0.30 * n.velocity + 0.70 * target_v
                if n.orig_idx in melody_ids and options.melody_boost > 0:
                    blended = min(float(options.vel_max), blended + options.melody_boost)
                new_v = int(round(max(options.vel_min, min(options.vel_max, blended))))
                if abs(new_v - n.velocity) >= 4:
                    n.velocity = new_v
                    n.reasons.append("velocity_balanced")
                    actions["velocities_balanced"] += 1

    # 7. Clean sustain pedals
    if options.clean_pedals and pedals:
        cleaned: list[tuple[float, float]] = []
        # Step A: Drop tiny bounces & merge near-touching pedals
        for start, end in sorted(pedals):
            if end - start < 0.08:
                actions["pedals_cleaned"] += 1
                continue
            if cleaned and start - cleaned[-1][1] < 0.04:
                cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], end))
                actions["pedals_cleaned"] += 1
            else:
                cleaned.append((start, end))

        # Step B: Re-pedal on chord changes or max_pedal_sec
        chord_times = [c[0] for c in chords]
        final_pedals: list[tuple[float, float]] = []
        lift = 0.05
        for start, end in cleaned:
            splits: list[float] = []
            if options.repedal_on_chord and chord_times:
                for ct in chord_times:
                    if start + 0.35 < ct < end - 0.25:
                        if not splits or ct - splits[-1] >= 0.35:
                            splits.append(ct)
            cur = start
            for sp in splits:
                # Also honor max_pedal_sec within each segment
                while sp - cur > options.max_pedal_sec:
                    cut = cur + options.max_pedal_sec
                    final_pedals.append((cur, cut - lift))
                    cur = cut
                    actions["pedals_cleaned"] += 1
                if sp - cur >= 0.12:
                    final_pedals.append((cur, sp - lift))
                    cur = sp
                    actions["pedals_cleaned"] += 1
            while end - cur > options.max_pedal_sec:
                cut = cur + options.max_pedal_sec
                final_pedals.append((cur, cut - lift))
                cur = cut
                actions["pedals_cleaned"] += 1
            if end - cur >= 0.08:
                final_pedals.append((cur, end))
        pedals = [(round(a, 3), round(b, 3)) for a, b in final_pedals if b > a + 0.05]

    # 8. Final safety re-pass on same-pitch overlaps after quantization/hand-folding
    if options.fix_overlaps:
        gap_sec = options.rearticulate_gap_ms / 1000.0
        by_pitch_final: dict[int, list[_NoteItem]] = {}
        for n in notes:
            if not n.removed:
                by_pitch_final.setdefault(n.pitch, []).append(n)
        for plist in by_pitch_final.values():
            plist.sort(key=lambda x: x.onset)
            for i in range(len(plist) - 1):
                cur, nxt = plist[i], plist[i + 1]
                max_end = nxt.onset - gap_sec
                if cur.onset + cur.duration > max_end:
                    cur.duration = max(0.025, max_end - cur.onset)

    # 9. Apply explicit user manual deletes / updates last
    by_idx = {n.orig_idx: n for n in notes}
    for idx in manual_deletes:
        item = by_idx.get(idx)
        if item is not None and not item.removed:
            item.removed = True
            item.reasons.append("manual_deleted")
            actions["manual_edits"] += 1
    for idx, patch in manual_updates.items():
        item = by_idx.get(idx)
        if item is None:
            continue
        changed = False
        if "onset" in patch and patch["onset"] is not None:
            item.onset = max(0.0, float(patch["onset"]))
            changed = True
        if "duration" in patch and patch["duration"] is not None:
            item.duration = max(0.02, float(patch["duration"]))
            changed = True
        if "pitch" in patch and patch["pitch"] is not None:
            item.pitch = int(np.clip(int(patch["pitch"]), 0, 127))
            changed = True
        if "velocity" in patch and patch["velocity"] is not None:
            item.velocity = int(np.clip(int(patch["velocity"]), 1, 127))
            changed = True
        if changed:
            item.removed = False
            if "manual_edited" not in item.reasons:
                item.reasons.append("manual_edited")
            actions["manual_edits"] += 1

    return notes, pedals, actions


def preview_midi_repair(
    midi_path: Path,
    options: RepairOptions,
    *,
    source_beats: list[float] | None = None,
    source_chords: list[tuple[float, str]] | None = None,
    manual_deletes: list[int] | None = None,
    manual_keeps: list[int] | None = None,
    manual_updates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute non-destructive Before/After analysis, note diffs, and playback views."""
    orig_notes, orig_pedals, midi_beats, midi_chords = _load_midi_elements(midi_path)
    beats = source_beats if (source_beats and len(source_beats) >= 2) else midi_beats
    chords = source_chords if source_chords else midi_chords

    before_counts, before_scores, before_issues = analyze_elements(orig_notes, orig_pedals, beats, chords, options)
    before_cover_notes = _compute_releases(orig_notes, orig_pedals)
    before_duration = max((float(n[4]) for n in before_cover_notes), default=0.0)

    upd_map = {int(u["idx"]): u for u in (manual_updates or []) if "idx" in u}
    repaired_notes, repaired_pedals, actions = _repair_elements(
        orig_notes,
        orig_pedals,
        beats,
        chords,
        options,
        manual_deletes=set(int(i) for i in (manual_deletes or [])),
        manual_keeps=set(int(i) for i in (manual_keeps or [])),
        manual_updates=upd_map,
    )

    after_counts, after_scores, after_issues = analyze_elements(repaired_notes, repaired_pedals, beats, chords, options)
    after_cover_notes = _compute_releases(repaired_notes, repaired_pedals)
    after_duration = max((float(n[4]) for n in after_cover_notes), default=before_duration)

    # Compact note-level diff entries for visual Piano Roll overlay:
    # Each entry: { idx, onset, duration, pitch, velocity, orig_onset, orig_pitch, status, reasons }
    diff_notes: list[dict[str, Any]] = []
    removed_count = 0
    modified_count = 0
    kept_count = 0
    for n in repaired_notes:
        structural_mod = (
            abs(n.onset - n.orig_onset) >= 0.008
            or abs(n.duration - n.orig_duration) >= 0.015
            or n.pitch != n.orig_pitch
            or abs(n.velocity - n.orig_velocity) >= 18
            or "manual_edited" in n.reasons
        )
        is_mod = not n.removed and structural_mod
        if n.removed:
            status = "manual_deleted" if "manual_deleted" in n.reasons else "removed"
            removed_count += 1
        elif is_mod:
            status = "modified"
            modified_count += 1
        else:
            status = "kept"
            kept_count += 1
        diff_notes.append(
            {
                "idx": n.orig_idx,
                "onset": round(n.onset, 3),
                "duration": round(n.duration, 3),
                "pitch": int(n.pitch),
                "velocity": int(n.velocity),
                "orig_onset": round(n.orig_onset, 3),
                "orig_duration": round(n.orig_duration, 3),
                "orig_pitch": int(n.orig_pitch),
                "status": status,
                "reasons": n.reasons,
            }
        )

    return {
        "options": asdict(options),
        "beats": [round(float(b), 3) for b in beats],
        "chords": [[round(float(t), 3), name] for t, name in chords],
        "before": {
            "duration": round(before_duration, 3),
            "note_count": len(orig_notes),
            "pedal_count": len(orig_pedals),
            "notes": before_cover_notes,
            "pedals": [[round(a, 3), round(b, 3)] for a, b in orig_pedals],
            "issue_counts": before_counts,
            "total_issues": sum(before_counts.values()),
            "scores": before_scores,
            "issues": before_issues,
        },
        "after": {
            "duration": round(after_duration, 3),
            "note_count": kept_count + modified_count,
            "pedal_count": len(repaired_pedals),
            "notes": after_cover_notes,
            "pedals": [[round(a, 3), round(b, 3)] for a, b in repaired_pedals],
            "issue_counts": after_counts,
            "total_issues": sum(after_counts.values()),
            "scores": after_scores,
            "issues": after_issues,
        },
        "diff": {
            "kept_count": kept_count,
            "modified_count": modified_count,
            "removed_count": removed_count,
            "actions": actions,
            "notes": diff_notes,
        },
    }


def write_repaired_midi(
    input_path: Path,
    output_path: Path,
    options: RepairOptions,
    *,
    is_source_midi: bool = False,
    metadata_source_path: Path | None = None,
    source_beats: list[float] | None = None,
    source_chords: list[tuple[float, str]] | None = None,
    manual_deletes: list[int] | None = None,
    manual_keeps: list[int] | None = None,
    manual_updates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repair MIDI and write atomically to output_path while preserving all metadata."""
    from piano_cover.metadata import apply_metadata
    from symusic import ControlChange, Note, Score, Track

    orig_notes, orig_pedals, midi_beats, midi_chords = _load_midi_elements(input_path)
    beats = source_beats if (source_beats and len(source_beats) >= 2) else midi_beats
    chords = source_chords if source_chords else midi_chords

    upd_map = {int(u["idx"]): u for u in (manual_updates or []) if "idx" in u}
    repaired_notes, repaired_pedals, actions = _repair_elements(
        orig_notes,
        orig_pedals,
        beats,
        chords,
        options,
        manual_deletes=set(int(i) for i in (manual_deletes or [])),
        manual_keeps=set(int(i) for i in (manual_keeps or [])),
        manual_updates=upd_map,
    )
    after_counts, after_scores, _ = analyze_elements(repaired_notes, repaired_pedals, beats, chords, options)
    alive_notes = [n for n in repaired_notes if not n.removed]

    orig_score = Score(str(input_path)).to("second")

    if is_source_midi:
        # Preserve multi-track structure and drum tracks for source.mid
        by_track: dict[int, list[_NoteItem]] = {}
        for n in alive_notes:
            by_track.setdefault(n.track_idx, []).append(n)
        for t_idx, track in enumerate(orig_score.tracks):
            if track.is_drum:
                continue
            track.notes.clear()
            for n in sorted(by_track.get(t_idx, []), key=lambda x: (x.onset, x.pitch)):
                track.notes.append(Note(n.onset, max(0.02, n.duration), int(n.pitch), int(n.velocity), "second"))
        out_score = orig_score
    else:
        out_score = Score(ttype="second")
        piano_track = Track(name="Piano", program=0, ttype="second")
        for n in sorted(alive_notes, key=lambda x: (x.onset, x.pitch)):
            piano_track.notes.append(Note(n.onset, max(0.02, n.duration), int(n.pitch), int(n.velocity), "second"))
        for p_start, p_end in sorted(repaired_pedals):
            if p_end > p_start + 0.02:
                piano_track.controls.append(ControlChange(p_start, 64, 127, "second"))
                piano_track.controls.append(ControlChange(p_end, 64, 0, "second"))
        out_score.tracks.append(piano_track)
        meta_ref = orig_score
        if metadata_source_path is not None and metadata_source_path.is_file():
            try:
                meta_ref = Score(str(metadata_source_path)).to("second")
            except Exception:
                meta_ref = orig_score
        apply_metadata(out_score, meta_ref)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".mid.tmp")
    out_score.dump_midi(str(tmp))
    os.replace(tmp, output_path)

    duration = max((n.onset + n.duration for n in alive_notes), default=0.0)
    if repaired_pedals:
        duration = max(duration, max(p[1] for p in repaired_pedals))
    if not math.isfinite(duration):
        duration = 0.0

    return {
        "note_count": len(alive_notes),
        "pedal_count": len(repaired_pedals),
        "duration": round(duration, 3),
        "actions": actions,
        "issue_counts": after_counts,
        "scores": after_scores,
    }

