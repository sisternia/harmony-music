import type { SongscriptionSegment, SongscriptionWord } from '../api';

/** Perceptual audio-visual lead (seconds) compensating for browser audio buffer + rAF paint latency. */
export const KARAOKE_SYNC_LEAD_SEC = 0.08;

const CJK_CHAR_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;

function wordSyllableWeight(text: string): number {
  const clean = text.replace(/[\s.,!?;:'"~…♪♫()[\]【】「」『』、。！？・-]+/g, '');
  if (!clean) return 1;
  if (CJK_CHAR_RE.test(clean)) return Math.max(1, clean.length);
  return Math.max(1, clean.length / 3);
}

export function getPlaybackPosition(time: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  const position = Number.isFinite(time) ? time : 0;
  return Math.min(duration, Math.max(0, position));
}

export function getPlaybackPercent(time: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return getPlaybackPosition(time, duration) / duration * 100;
}

/**
 * Repair collapsed/stacked forced-aligner word timestamps (e.g. 4 words stacked at 38.48-38.53
 * or an over-stretched word swallowing the next 4 words) at playback time without mutating saved data.
 * Well-aligned words pass through unchanged.
 */
export function prepareWordsForKaraoke(
  words: readonly SongscriptionWord[] | undefined | null,
  segStart: number,
  segEnd: number,
  nextSegStart?: number | null,
): SongscriptionWord[] {
  if (!words || words.length === 0) return [];
  const n = words.length;
  const out: SongscriptionWord[] = words.map(w => ({ ...w }));
  const streamCeiling =
    nextSegStart != null && Number.isFinite(nextSegStart) && nextSegStart > segStart + 0.08
      ? nextSegStart
      : null;
  const boundedSegEnd = streamCeiling !== null ? Math.min(segEnd, streamCeiling) : segEnd;

  const isCollapsed = (idx: number): boolean => {
    const s = out[idx].start;
    const e = out[idx].end;
    if (!Number.isFinite(s) || !Number.isFinite(e)) return true;
    if (e - s <= 0.065) return true;
    if (idx > 0 && s <= out[idx - 1].start + 0.025) return true;
    if (idx + 1 < n && out[idx + 1].start <= s + 0.025) return true;
    return false;
  };

  let i = 0;
  while (i < n) {
    if (!isCollapsed(i)) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < n && isCollapsed(j + 1)) {
      j++;
    }

    let spanA = i;
    const spanB = j;
    let spanStart = Number.isFinite(out[spanA].start) ? out[spanA].start : segStart;
    let spanEnd = spanStart;
    for (let k = spanA; k <= spanB; k++) {
      if (Number.isFinite(out[k].end)) spanEnd = Math.max(spanEnd, out[k].end);
    }

    // 1. Stalled predecessor swallowed the collapsed run (e.g. "go" 74.88..78.72 + 5 stacked words)
    if (spanA > 0) {
      const prev = out[spanA - 1];
      const prevDur = prev.end - prev.start;
      const expectedPrev = Math.max(0.55, 0.34 * wordSyllableWeight(prev.word));
      const gapFromPrev = spanStart - prev.end;
      if (prevDur > expectedPrev * 1.45 && gapFromPrev <= 0.16) {
        spanA -= 1;
        spanStart = prev.start;
      } else if (gapFromPrev >= 0.32 && (spanB - spanA >= 1 || spanEnd - spanStart <= 0.08)) {
        // 2. Dead gap before stacked words (e.g. "Another" 32.48..32.96 -> hole -> "way to feel." at 34.32)
        spanStart = prev.end + 0.04;
      }
    }

    // 3. Collapsed tail/final word — extend cleanly up to segEnd (or before nextSegStart if segEnd itself was truncated)
    const count = spanB - spanA + 1;
    let minNatural = 0;
    for (let k = spanA; k <= spanB; k++) {
      minNatural += Math.max(0.14, 0.16 * wordSyllableWeight(out[k].word));
    }
    const hardLimit =
      spanB + 1 < n
        ? Math.max(spanEnd, out[spanB + 1].start - 0.02)
        : streamCeiling !== null
          ? Math.min(streamCeiling, Math.max(boundedSegEnd, spanStart + Math.max(minNatural * 1.35, 0.6 * count)))
          : Math.max(boundedSegEnd, spanStart + Math.max(minNatural * 1.25, 0.55 * count));
    if (spanEnd - spanStart < minNatural && hardLimit > spanEnd + 0.04) {
      spanEnd = Math.min(hardLimit, Math.max(spanEnd, spanStart + minNatural));
    }

    // 4. If phrase already ended right at spanEnd (e.g. 5 words piled up at end of line),
    // borrow back across contiguous preceding words so every word flows smoothly before line end.
    while (spanA > 0 && spanEnd - spanStart < 0.13 * (spanB - spanA + 1)) {
      const prevEnd = out[spanA - 1].end;
      if (spanStart - prevEnd > 0.25) break;
      spanA -= 1;
      spanStart = out[spanA].start;
    }

    const finalCount = spanB - spanA + 1;
    const totalDur = Math.max(0.06 * finalCount, spanEnd - spanStart);
    const weights: number[] = [];
    let wSum = 0;
    for (let k = spanA; k <= spanB; k++) {
      const wt = wordSyllableWeight(out[k].word);
      weights.push(wt);
      wSum += wt;
    }
    let cur = spanStart;
    for (let idx = 0; idx < finalCount; idx++) {
      const k = spanA + idx;
      const durK = totalDur * (weights[idx] / Math.max(1e-6, wSum));
      const nxt = idx === finalCount - 1 ? spanStart + totalDur : cur + durK;
      out[k].start = Number(cur.toFixed(2));
      out[k].end = Number(Math.max(cur + 0.04, nxt).toFixed(2));
      cur = nxt;
    }

    i = spanB + 1;
  }

  // Enforce strict monotonicity & clamp strictly within same-stream ceiling so a line finishes
  // cleanly BEFORE the next sentence on the same stream begins.
  if (streamCeiling !== null && n > 0 && out[n - 1].end > streamCeiling) {
    const firstS = Math.min(segStart, out[0].start);
    const oldSpan = Math.max(0.05, out[n - 1].end - firstS);
    const newSpan = Math.max(0.05, streamCeiling - firstS);
    const scale = newSpan / oldSpan;
    for (let k = 0; k < n; k++) {
      out[k].start = Number((firstS + (out[k].start - firstS) * scale).toFixed(2));
      out[k].end = Number((firstS + (out[k].end - firstS) * scale).toFixed(2));
    }
    out[n - 1].end = Number(streamCeiling.toFixed(2));
  }

  for (let k = 1; k < n; k++) {
    const prev = out[k - 1];
    const cur = out[k];
    if (cur.start < prev.end) {
      if (cur.end > prev.end + 0.04) {
        cur.start = prev.end;
      } else {
        const mid = Number(Math.max(prev.start + 0.04, (prev.start + Math.max(prev.end, cur.end)) * 0.5).toFixed(2));
        prev.end = mid;
        cur.start = mid;
        cur.end = Number(Math.max(mid + 0.04, cur.end).toFixed(2));
      }
    }
  }
  if (streamCeiling !== null && n > 0 && out[n - 1].end > streamCeiling) {
    out[n - 1].end = Number(streamCeiling.toFixed(2));
  }
  return out;
}

export function getWordPlaybackState(
  word: Pick<SongscriptionWord, 'start' | 'end'>,
  time: number,
  nextWordStart?: number | null,
): 'upcoming' | 'active' | 'sung' {
  if (!Number.isFinite(time) || !Number.isFinite(word.start) || time < word.start) return 'upcoming';
  const baseEnd = Number.isFinite(word.end) ? Math.max(word.start, word.end) : word.start;
  // Bridge tiny <= 0.16s inter-word gaps so active highlight flows smoothly without blinking off
  const bridgedEnd =
    nextWordStart != null &&
    Number.isFinite(nextWordStart) &&
    nextWordStart > baseEnd &&
    nextWordStart - baseEnd <= 0.16
      ? nextWordStart
      : baseEnd;
  return time < bridgedEnd ? 'active' : 'sung';
}

export function getActiveSegmentIds(
  segments: readonly Pick<SongscriptionSegment, 'id' | 'start' | 'end' | 'stream_id'>[],
  time: number,
): Set<number> {
  const active = new Set<number>();
  if (!Number.isFinite(time) || segments.length === 0) return active;
  // At most ONE segment per stream can be active at any instant (two lines on the same stream
  // must never highlight simultaneously; different streams stream_a & stream_b can overlap freely).
  const activeByStream = new Map<string, { id: number; start: number }>();
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (!Number.isFinite(seg.start) || !Number.isFinite(seg.end)) continue;
    const end = Math.max(seg.start, seg.end);
    if (time >= seg.start && time < end) {
      const streamKey = seg.stream_id || 'stream_a';
      const prev = activeByStream.get(streamKey);
      if (!prev || seg.start >= prev.start) {
        activeByStream.set(streamKey, { id: seg.id, start: seg.start });
      }
    }
  }
  for (const item of activeByStream.values()) {
    active.add(item.id);
  }
  return active;
}
