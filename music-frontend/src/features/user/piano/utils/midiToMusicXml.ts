import type { PianoSong, SongNote } from './transcription';

export type EngravedMeasureTiming = {
  measureNumber: number;
  start: number;
  end: number;
  quarters: number;
  beats: number;
  beatType: number;
  bpm: number;
  beatTimes?: number[];
};

export type EngravedScore = {
  xml: string;
  measures: EngravedMeasureTiming[];
  eventTimes: Map<string, number>;
};

const DIVISIONS = 4; // 4 divisions per quarter note = 16th-note grid

const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
// Major tonic pitch class -> fifths (-6..+6)
const MAJOR_FIFTHS = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5];
// Minor tonic pitch class -> fifths (-6..+6)
const MINOR_FIFTHS = [-3, 4, -1, -6, 1, -4, 3, -2, 5, 0, -5, 2];

const SHARP_SPELL: [string, number][] = [
  ['C', 0], ['C', 1], ['D', 0], ['D', 1], ['E', 0], ['F', 0],
  ['F', 1], ['G', 0], ['G', 1], ['A', 0], ['A', 1], ['B', 0],
];
const FLAT_SPELL: [string, number][] = [
  ['C', 0], ['D', -1], ['D', 0], ['E', -1], ['E', 0], ['F', 0],
  ['G', -1], ['G', 0], ['A', -1], ['A', 0], ['B', -1], ['B', 0],
];

const FIFTHS_SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];
const FIFTHS_FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F'];
const STEP_SEMITONE: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function keyAlters(fifths: number): Record<string, number> {
  const out: Record<string, number> = { C: 0, D: 0, E: 0, F: 0, G: 0, A: 0, B: 0 };
  if (fifths > 0) {
    for (let i = 0; i < Math.min(7, fifths); i++) out[FIFTHS_SHARP_ORDER[i]] = 1;
  } else if (fifths < 0) {
    for (let i = 0; i < Math.min(7, -fifths); i++) out[FIFTHS_FLAT_ORDER[i]] = -1;
  }
  return out;
}

function estimateFifths(notes: SongNote[], explicitFifths?: number): number {
  if (explicitFifths !== undefined && explicitFifths !== 0) {
    return Math.max(-6, Math.min(6, explicitFifths));
  }
  const pcWeight = new Float64Array(12);
  for (const n of notes) {
    const dur = Math.max(0.05, Math.min(2.5, (n.keyEnd ?? n.end) - n.start));
    pcWeight[((n.pitch % 12) + 12) % 12] += dur * (0.5 + n.velocity / 254);
  }
  let bestScore = -Infinity;
  let bestFifths = 0;
  let cScore = -Infinity;
  for (let root = 0; root < 12; root++) {
    let maj = 0;
    let min = 0;
    for (let i = 0; i < 12; i++) {
      const w = pcWeight[(root + i) % 12];
      maj += w * MAJOR_PROFILE[i];
      min += w * MINOR_PROFILE[i];
    }
    if (root === 0) cScore = Math.max(cScore, maj);
    if (root === 9) cScore = Math.max(cScore, min);
    if (maj > bestScore) {
      bestScore = maj;
      bestFifths = MAJOR_FIFTHS[root];
    }
    if (min > bestScore) {
      bestScore = min;
      bestFifths = MINOR_FIFTHS[root];
    }
  }
  if (explicitFifths === 0 && bestScore < cScore * 1.08) return 0;
  return bestFifths;
}

function spellPitch(pitch: number, fifths: number, alters: Record<string, number>): { step: string; alter: number; octave: number } {
  const pc = ((pitch % 12) + 12) % 12;
  // Prefer a diatonic scale step of the active key if one matches this pitch class
  for (const step of ['C', 'D', 'E', 'F', 'G', 'A', 'B']) {
    const alt = alters[step];
    if (((STEP_SEMITONE[step] + alt) % 12 + 12) % 12 === pc) {
      const baseMidi = pitch - alt;
      const octave = Math.floor(baseMidi / 12) - 1;
      return { step, alter: alt, octave };
    }
  }
  const table = fifths >= 0 ? SHARP_SPELL : FLAT_SPELL;
  const [step, alter] = table[pc];
  const octave = Math.floor((pitch - alter) / 12) - 1;
  return { step, alter, octave };
}

function parseMeter(meter: string): [number, number] {
  const match = meter.match(/^(\d+)\s*\/\s*(\d+)$/);
  if (!match) return [4, 4];
  const beats = Math.max(1, Math.min(12, Number(match[1]) || 4));
  const beatType = [2, 4, 8, 16].includes(Number(match[2])) ? Number(match[2]) : 4;
  return [beats, beatType];
}

export function buildMeasureTimings(song: PianoSong): EngravedMeasureTiming[] {
  const duration = Math.max(1, song.duration, ...song.notes.map(n => n.keyEnd ?? n.end));

  if (song.midiBars && song.midiBars.length > 0) {
    const out: EngravedMeasureTiming[] = song.midiBars.map((bar, idx) => ({
      measureNumber: idx + 1,
      start: bar.start,
      end: bar.end,
      quarters: (bar.beats * 4) / bar.beatType,
      beats: bar.beats,
      beatType: bar.beatType,
      bpm: bar.bpm,
    }));
    while (out[out.length - 1].end < duration - 0.02 && out.length < 1200) {
      const last = out[out.length - 1];
      const barDur = Math.max(0.4, last.end - last.start);
      out.push({
        measureNumber: out.length + 1,
        start: last.end,
        end: last.end + barDur,
        quarters: last.quarters,
        beats: last.beats,
        beatType: last.beatType,
        bpm: last.bpm,
      });
    }
    return out;
  }

  const [defBeats, defBeatType] = parseMeter(song.meter || '4/4');
  const defQuarters = (defBeats * 4) / defBeatType;

  if (song.downbeats && song.downbeats.length >= 2) {
    const downbeats = [...song.downbeats].filter(t => Number.isFinite(t) && t >= 0).sort((a, b) => a - b);
    const beats = song.beats ? [...song.beats].filter(t => Number.isFinite(t) && t >= 0).sort((a, b) => a - b) : [];

    const beatIntervals: number[] = [];
    for (let i = 1; i < Math.min(beats.length, 200); i++) {
      const dt = beats[i] - beats[i - 1];
      if (dt > 0.18 && dt < 2.0) beatIntervals.push(dt);
    }
    beatIntervals.sort((a, b) => a - b);
    const medianBeat = beatIntervals.length
      ? beatIntervals[beatIntervals.length >> 1]
      : Math.max(0.25, (downbeats[1] - downbeats[0]) / defBeats);

    const rawBars: Omit<EngravedMeasureTiming, 'measureNumber'>[] = [];

    // Handle notes or beats before the first downbeat
    const firstNoteStart = song.notes.length ? song.notes[0].start : 0;
    const firstDownbeat = downbeats[0];
    let firstBarStart = firstDownbeat;
    if (firstDownbeat <= medianBeat * 0.65) {
      firstBarStart = 0;
    } else if (firstNoteStart < firstDownbeat - 0.06 || firstDownbeat > medianBeat * 0.85) {
      const fullBarDur = defBeats * medianBeat;
      const introStarts: number[] = [];
      let cursor = firstDownbeat;
      while (cursor > 0.08) {
        const prev = Math.max(0, cursor - fullBarDur);
        introStarts.unshift(prev);
        if (prev === 0) break;
        cursor = prev;
      }
      for (let i = 0; i < introStarts.length; i++) {
        const s = introStarts[i];
        const e = i + 1 < introStarts.length ? introStarts[i + 1] : firstDownbeat;
        const approxBeats = Math.max(1, Math.min(defBeats, Math.round((e - s) / medianBeat)));
        rawBars.push({
          start: s,
          end: e,
          quarters: (approxBeats * 4) / defBeatType,
          beats: approxBeats,
          beatType: defBeatType,
          bpm: Math.max(30, Math.min(260, (approxBeats * 60) / Math.max(0.2, e - s))),
        });
      }
    }

    for (let i = 0; i < downbeats.length; i++) {
      const start = i === 0 ? firstBarStart : downbeats[i];
      const nextDownbeat = i + 1 < downbeats.length ? downbeats[i + 1] : start + defBeats * medianBeat;
      if (nextDownbeat - start < 0.15) continue;

      const innerBeats = beats.filter(b => b >= downbeats[i] - 0.05 && b < nextDownbeat - 0.05);
      let barBeats = defBeats;
      let barBeatType = defBeatType;
      if (innerBeats.length >= 2 && innerBeats.length <= 12) {
        const avgBeat = (nextDownbeat - downbeats[i]) / innerBeats.length;
        if (avgBeat < medianBeat * 0.62 && innerBeats.length % 2 === 0) {
          barBeats = innerBeats.length / 2;
          barBeatType = 4;
        } else {
          barBeats = innerBeats.length;
          barBeatType = 4;
        }
      } else {
        const approx = Math.round((nextDownbeat - start) / medianBeat);
        if (approx >= 2 && approx <= 8) barBeats = approx;
      }
      const quarters = (barBeats * 4) / barBeatType;
      const bpm = Math.max(30, Math.min(260, (quarters * 60) / (nextDownbeat - start)));
      rawBars.push({
        start,
        end: nextDownbeat,
        quarters,
        beats: barBeats,
        beatType: barBeatType,
        bpm,
        beatTimes: innerBeats.length === barBeats ? innerBeats : undefined,
      });
    }

    const tailBarDur = defBeats * medianBeat;
    while (rawBars.length > 0 && rawBars[rawBars.length - 1].end < duration - 0.05 && rawBars.length < 1200) {
      const last = rawBars[rawBars.length - 1];
      rawBars.push({
        start: last.end,
        end: last.end + tailBarDur,
        quarters: defQuarters,
        beats: defBeats,
        beatType: defBeatType,
        bpm: Math.max(30, Math.min(260, 60 / medianBeat)),
      });
    }

    if (rawBars.length > 0) {
      return rawBars.map((bar, idx) => ({ ...bar, measureNumber: idx + 1 }));
    }
  }

  const safeBpm = Math.max(30, Math.min(260, song.bpm || 120));
  const barDur = (defQuarters * 60) / safeBpm;
  const out: EngravedMeasureTiming[] = [];
  for (let start = 0; start < duration - 0.01 && out.length < 1200; start += barDur) {
    out.push({
      measureNumber: out.length + 1,
      start,
      end: start + barDur,
      quarters: defQuarters,
      beats: defBeats,
      beatType: defBeatType,
      bpm: safeBpm,
    });
  }
  if (!out.length) {
    out.push({ measureNumber: 1, start: 0, end: barDur, quarters: defQuarters, beats: defBeats, beatType: defBeatType, bpm: safeBpm });
  }
  return out;
}

type HandNote = {
  pitch: number;
  start: number;
  end: number;
  velocity: number;
  staff: 1 | 2;
};

function assignHands(notes: SongNote[]): HandNote[] {
  return notes
    .filter(n => n.pitch >= 21 && n.pitch <= 108)
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch)
    .map(n => {
      const pressEnd = Math.max(n.start + 0.04, n.keyEnd ?? n.end);
      const effectiveEnd = Math.min(n.end, pressEnd + Math.min(0.22, (n.end - pressEnd) * 0.35));
      return {
        pitch: n.pitch,
        start: n.start,
        end: effectiveEnd,
        velocity: n.velocity,
        staff: n.hand === 'L' ? 2 : 1,
      };
    });
}

function timeToGrid(measure: EngravedMeasureTiming, time: number, totalGrid: number): number {
  const { start, end, beatTimes, beats } = measure;
  if (time <= start) return 0;
  if (time >= end) return totalGrid;
  if (beatTimes && beatTimes.length === beats && beats >= 2) {
    const gridPerBeat = totalGrid / beats;
    const bounds = [start, ...beatTimes.slice(1), end];
    for (let b = 0; b < beats; b++) {
      const b0 = bounds[b];
      const b1 = bounds[b + 1];
      if (time <= b1 || b === beats - 1) {
        const frac = b1 > b0 ? Math.max(0, Math.min(1, (time - b0) / (b1 - b0))) : 0;
        return Math.round((b + frac) * gridPerBeat);
      }
    }
  }
  return Math.round(((time - start) / Math.max(0.05, end - start)) * totalGrid);
}

export function gridToTime(measure: EngravedMeasureTiming, grid: number, totalGrid: number): number {
  const { start, end, beatTimes, beats } = measure;
  const clamped = Math.max(0, Math.min(totalGrid, grid));
  if (beatTimes && beatTimes.length === beats && beats >= 2) {
    const gridPerBeat = totalGrid / beats;
    const beatFloat = clamped / gridPerBeat;
    const idx = Math.min(beats - 1, Math.floor(beatFloat));
    const frac = beatFloat - idx;
    const bounds = [start, ...beatTimes.slice(1), end];
    return bounds[idx] + frac * (bounds[idx + 1] - bounds[idx]);
  }
  return start + (clamped / totalGrid) * (end - start);
}

type Atom = {
  startGrid: number;
  grid: number;
  type: 'whole' | 'half' | 'quarter' | 'eighth' | '16th';
  dots: number;
  pitches: number[]; // empty = rest
  tieStart: Set<number>;
  tieStop: Set<number>;
  wholeMeasureRest?: boolean;
};

const DIRECT_ATOMS = new Map<number, { type: Atom['type']; dots: number }>([
  [16, { type: 'whole', dots: 0 }],
  [12, { type: 'half', dots: 1 }],
  [8, { type: 'half', dots: 0 }],
  [6, { type: 'quarter', dots: 1 }],
  [4, { type: 'quarter', dots: 0 }],
  [3, { type: 'eighth', dots: 1 }],
  [2, { type: 'eighth', dots: 0 }],
  [1, { type: '16th', dots: 0 }],
]);

function splitSpan(pos: number, end: number, totalGrid: number, beatGrid: number, isRest: boolean): { startGrid: number; grid: number; type: Atom['type']; dots: number }[] {
  const len = end - pos;
  if (len <= 0) return [];

  if (!isRest) {
    if (len === 16 && pos === 0 && totalGrid === 16) return [{ startGrid: pos, grid: 16, type: 'whole', dots: 0 }];
    if (len === 12 && totalGrid >= 12 && (pos === 0 || pos === beatGrid)) return [{ startGrid: pos, grid: 12, type: 'half', dots: 1 }];
    if (len === 8 && pos % beatGrid === 0) return [{ startGrid: pos, grid: 8, type: 'half', dots: 0 }];
    if (len === 6 && pos % beatGrid === 0) return [{ startGrid: pos, grid: 6, type: 'quarter', dots: 1 }];
  } else {
    if (len === 8 && totalGrid === 16 && (pos === 0 || pos === 8)) return [{ startGrid: pos, grid: 8, type: 'half', dots: 0 }];
  }

  const halfBar = totalGrid === 16 ? 8 : totalGrid;
  if (pos < halfBar && end > halfBar) {
    return [
      ...splitSpan(pos, halfBar, totalGrid, beatGrid, isRest),
      ...splitSpan(halfBar, end, totalGrid, beatGrid, isRest),
    ];
  }

  const beatEnd = (Math.floor(pos / beatGrid) + 1) * beatGrid;
  if (end > beatEnd) {
    return [
      ...splitSpan(pos, beatEnd, totalGrid, beatGrid, isRest),
      ...splitSpan(beatEnd, end, totalGrid, beatGrid, isRest),
    ];
  }

  // Now [pos, end] lies completely inside a single beat
  const direct = DIRECT_ATOMS.get(len);
  if (direct) {
    // Avoid dotted eighth starting on a 16th offbeat (e.g. pos % 4 === 1, len === 3 -> 16th + eighth)
    const awkwardDottedEighth = len === 3 && pos % 2 === 1;
    const awkwardRest = isRest && len === 3;
    if (!awkwardDottedEighth && !awkwardRest) {
      return [{ startGrid: pos, grid: len, type: direct.type, dots: direct.dots }];
    }
  }

  const subBeat = (Math.floor(pos / 2) + 1) * 2;
  if (subBeat > pos && subBeat < end) {
    return [
      ...splitSpan(pos, subBeat, totalGrid, beatGrid, isRest),
      ...splitSpan(subBeat, end, totalGrid, beatGrid, isRest),
    ];
  }

  // Greedy fallback guaranteed to terminate
  for (const g of [4, 2, 1]) {
    if (g <= len) {
      const atom = DIRECT_ATOMS.get(g)!;
      return [
        { startGrid: pos, grid: g, type: atom.type, dots: atom.dots },
        ...splitSpan(pos + g, end, totalGrid, beatGrid, isRest),
      ];
    }
  }
  return [];
}

function computeBeams(atoms: Atom[], beatGrid: number): Map<number, Record<number, string>> {
  const beams = new Map<number, Record<number, string>>();
  let idx = 0;
  while (idx < atoms.length) {
    const first = atoms[idx];
    if (!first.pitches.length || first.grid >= 4) {
      idx++;
      continue;
    }
    const beatBucket = Math.floor(first.startGrid / beatGrid);
    const run: number[] = [idx++];
    while (
      idx < atoms.length &&
      atoms[idx].pitches.length > 0 &&
      atoms[idx].grid < 4 &&
      Math.floor(atoms[idx].startGrid / beatGrid) === beatBucket
    ) {
      run.push(idx++);
    }
    if (run.length < 2) continue;
    for (let k = 0; k < run.length; k++) {
      const atomIdx = run[k];
      const entry: Record<number, string> = {
        1: k === 0 ? 'begin' : k === run.length - 1 ? 'end' : 'continue',
      };
      if (atoms[atomIdx].type === '16th') {
        const prev16 = k > 0 && atoms[run[k - 1]].type === '16th';
        const next16 = k + 1 < run.length && atoms[run[k + 1]].type === '16th';
        if (prev16 && next16) entry[2] = 'continue';
        else if (!prev16 && next16) entry[2] = 'begin';
        else if (prev16 && !next16) entry[2] = 'end';
        else entry[2] = k === 0 ? 'forward hook' : 'backward hook';
      }
      beams.set(atomIdx, entry);
    }
  }
  return beams;
}

type ChordMark = { grid: number; symbol: string };

const CHORD_PATTERNS: [string, number[], number][] = [
  ['maj7', [0, 4, 7, 11], 0.08],
  ['m7', [0, 3, 7, 10], 0.06],
  ['7', [0, 4, 7, 10], 0.05],
  ['sus4', [0, 5, 7], -0.04],
  ['sus2', [0, 2, 7], -0.08],
  ['dim', [0, 3, 6], -0.1],
  ['m', [0, 3, 7], 0],
  ['', [0, 4, 7], 0.02],
];

function pcName(pc: number, fifths: number): string {
  const sharpNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const flatNames = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
  return (fifths >= 0 ? sharpNames : flatNames)[((pc % 12) + 12) % 12];
}

function detectChordInWindow(notes: HandNote[], wStart: number, wEnd: number, fifths: number): string | null {
  const active = notes.filter(n => n.start < wEnd - 0.02 && n.end > wStart + 0.02);
  if (active.length < 3) return null;
  const weights = new Float64Array(12);
  let lowestPitch = 128;
  let lowestWeight = 0;
  for (const n of active) {
    const overlap = Math.max(0.02, Math.min(n.end, wEnd) - Math.max(n.start, wStart));
    const bassBoost = n.staff === 2 ? 1.35 : 1.0;
    const w = overlap * bassBoost;
    weights[((n.pitch % 12) + 12) % 12] += w;
    if (n.pitch < lowestPitch || (n.pitch === lowestPitch && overlap > lowestWeight)) {
      lowestPitch = n.pitch;
      lowestWeight = overlap;
    }
  }
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0.15) return null;

  let bestSymbol = '';
  let bestScore = total * 0.36;
  const bassPc = lowestPitch <= 57 ? ((lowestPitch % 12) + 12) % 12 : -1;

  for (let root = 0; root < 12; root++) {
    for (const [suffix, intervals, bias] of CHORD_PATTERNS) {
      let matched = 0;
      for (const iv of intervals) matched += weights[(root + iv) % 12];
      // Require root and defining interval to be actually present
      if (weights[root] < total * 0.08) continue;
      if (intervals.length >= 4 && weights[(root + intervals[3]) % 12] < total * 0.09) continue;
      const outside = total - matched;
      const rootBassBonus = bassPc === root ? total * 0.1 : 0;
      const score = matched - outside * 0.72 + rootBassBonus + total * bias;
      if (score > bestScore) {
        bestScore = score;
        const rootStr = pcName(root, fifths);
        let slash = '';
        if (bassPc >= 0 && bassPc !== root && lowestWeight >= (wEnd - wStart) * 0.25 && intervals.some(iv => (root + iv) % 12 === bassPc)) {
          slash = `/${pcName(bassPc, fifths)}`;
        }
        bestSymbol = `${rootStr}${suffix}${slash}`;
      }
    }
  }
  return bestSymbol || null;
}

const HARMONY_KINDS: [string, string][] = [
  ['maj7', 'major-seventh'],
  ['M7', 'major-seventh'],
  ['mM7', 'major-minor'],
  ['m7b5', 'half-diminished'],
  ['m7-5', 'half-diminished'],
  ['m7', 'minor-seventh'],
  ['m6', 'minor-sixth'],
  ['m9', 'minor-ninth'],
  ['m', 'minor'],
  ['7sus4', 'suspended-fourth'],
  ['9', 'dominant-ninth'],
  ['11', 'dominant-11th'],
  ['13', 'dominant-13th'],
  ['7', 'dominant'],
  ['dim7', 'diminished-seventh'],
  ['dim', 'diminished'],
  ['aug', 'augmented'],
  ['sus4', 'suspended-fourth'],
  ['sus2', 'suspended-second'],
  ['add9', 'major'],
  ['6', 'major-sixth'],
  ['5', 'power'],
  ['', 'major'],
];

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function renderHarmonyXml(symbol: string): string {
  const match = symbol.match(/^([A-G])([#b]?)(.*?)(?:\/([A-G])([#b]?))?$/);
  if (!match) return '';
  const [, rStep, rAcc, qualRaw, bStep, bAcc] = match;
  const kind = HARMONY_KINDS.find(([prefix]) => qualRaw.startsWith(prefix))?.[1] ?? 'major';
  let out = '<harmony placement="above">';
  out += `<root><root-step>${rStep}</root-step>${rAcc ? `<root-alter>${rAcc === '#' ? 1 : -1}</root-alter>` : ''}</root>`;
  out += `<kind text="${escapeXml(qualRaw)}">${kind}</kind>`;
  if (bStep) {
    out += `<bass><bass-step>${bStep}</bass-step>${bAcc ? `<bass-alter>${bAcc === '#' ? 1 : -1}</bass-alter>` : ''}</bass>`;
  }
  out += '<staff>1</staff></harmony>';
  return out;
}

function velocityToDynamic(vel: number): 'pp' | 'p' | 'mp' | 'mf' | 'f' | 'ff' {
  if (vel < 40) return 'pp';
  if (vel < 56) return 'p';
  if (vel < 72) return 'mp';
  if (vel < 90) return 'mf';
  if (vel < 108) return 'f';
  return 'ff';
}

const ACCIDENTAL_XML: Record<number, string> = {
  [-1]: 'flat',
  0: 'natural',
  1: 'sharp',
};

export function midiToMusicXml(song: PianoSong): EngravedScore {
  const measures = buildMeasureTimings(song);
  const fifths = estimateFifths(song.notes, song.fifths);
  const alters = keyAlters(fifths);
  const handNotes = assignHands(song.notes);
  const eventTimes = new Map<string, number>();

  // Index notes into measures
  const measureNotes: HandNote[][] = measures.map(() => []);
  let mHint = 0;
  for (const note of handNotes) {
    while (mHint + 1 < measures.length && measures[mHint].end <= note.start + 0.015) mHint++;
    // Snap slightly early attacks (within 35ms before barline) into the upcoming measure's downbeat
    let targetM = mHint;
    if (
      targetM + 1 < measures.length &&
      measures[targetM + 1].start - note.start > 0 &&
      measures[targetM + 1].start - note.start <= 0.035
    ) {
      targetM = targetM + 1;
    }
    if (targetM < measures.length && note.start < measures[targetM].end) {
      measureNotes[targetM].push(note);
    }
  }

  // Carry-over pitches across barlines per staff: pitch -> remaining end time in seconds
  let carryByStaff: Record<1 | 2, Map<number, number>> = { 1: new Map(), 2: new Map() };

  let prevKey: number | null = null;
  let prevBeats = 0;
  let prevBeatType = 0;
  let lastShownBpm = 0;
  let lastChordSymbol = '';
  let lastDynamic = '';
  let lastDynamicMeasure = -99;

  const xmlParts: string[] = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<score-partwise version="3.1">',
    '<part-list><score-part id="P1"><part-name print-object="no">Piano</part-name></score-part></part-list>',
    '<part id="P1">',
  ];

  for (let mi = 0; mi < measures.length; mi++) {
    const m = measures[mi];
    const totalGrid = Math.max(4, Math.round(m.quarters * DIVISIONS));
    const beatGrid = m.beatType === 8 && m.beats % 3 === 0 ? 6 : 4;
    const notesInBar = measureNotes[mi];
    const nextCarry: Record<1 | 2, Map<number, number>> = { 1: new Map(), 2: new Map() };

    xmlParts.push(`<measure number="${m.measureNumber}">`);

    const needKey = prevKey === null || prevKey !== fifths;
    const needTime = prevBeats !== m.beats || prevBeatType !== m.beatType;
    if (mi === 0 || needKey || needTime) {
      xmlParts.push('<attributes>');
      if (mi === 0) xmlParts.push(`<divisions>${DIVISIONS}</divisions>`);
      if (needKey) xmlParts.push(`<key><fifths>${fifths}</fifths></key>`);
      if (needTime) xmlParts.push(`<time><beats>${m.beats}</beats><beat-type>${m.beatType}</beat-type></time>`);
      if (mi === 0) {
        xmlParts.push('<staves>2</staves>');
        xmlParts.push('<clef number="1"><sign>G</sign><line>2</line></clef>');
        xmlParts.push('<clef number="2"><sign>F</sign><line>4</line></clef>');
      }
      xmlParts.push('</attributes>');
      prevKey = fifths;
      prevBeats = m.beats;
      prevBeatType = m.beatType;
    }

    const roundedBpm = Math.round(m.bpm);
    if (mi === 0 || (Math.abs(roundedBpm - lastShownBpm) >= 10 && mi % 4 === 0)) {
      xmlParts.push(
        `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${roundedBpm}</per-minute></metronome></direction-type><staff>1</staff><sound tempo="${roundedBpm}"/></direction>`,
      );
      lastShownBpm = roundedBpm;
    }

    // Determine chords in this measure (at grid 0 and optionally half-bar)
    const chordMarks: ChordMark[] = [];
    if (song.chords && song.chords.length > 0) {
      for (const ch of song.chords) {
        if (ch.time >= m.start - 0.08 && ch.time < m.end - 0.08) {
          const rawG = timeToGrid(m, Math.max(m.start, ch.time), totalGrid);
          const snappedG = Math.min(totalGrid - beatGrid, Math.round(rawG / beatGrid) * beatGrid);
          if (ch.label && ch.label !== lastChordSymbol && !chordMarks.some(cm => cm.grid === snappedG)) {
            chordMarks.push({ grid: snappedG, symbol: ch.label });
            lastChordSymbol = ch.label;
          }
        }
      }
    } else if (notesInBar.length >= 3) {
      const halfDur = (m.end - m.start) / 2;
      const windows: [number, number, number][] = totalGrid >= 16
        ? [[0, m.start, m.start + halfDur], [8, m.start + halfDur, m.end]]
        : [[0, m.start, m.end]];
      for (const [g, ws, we] of windows) {
        const detected = detectChordInWindow(notesInBar, ws, we, fifths);
        if (detected && detected !== lastChordSymbol) {
          chordMarks.push({ grid: g, symbol: detected });
          lastChordSymbol = detected;
        }
      }
    }

    // Dynamics check (at first active note in measure if dynamic level changed cleanly)
    let measureDynamic: { grid: number; dyn: string } | null = null;
    if (notesInBar.length >= 2 && mi - lastDynamicMeasure >= 3) {
      const avgVel = notesInBar.reduce((s, n) => s + n.velocity, 0) / notesInBar.length;
      const dyn = velocityToDynamic(avgVel);
      if (dyn !== lastDynamic) {
        const firstG = Math.min(totalGrid - 1, Math.max(0, timeToGrid(m, notesInBar[0].start, totalGrid)));
        measureDynamic = { grid: firstG, dyn };
        lastDynamic = dyn;
        lastDynamicMeasure = mi;
      }
    }

    // Build atoms for Staff 1 (Treble, voice 1) and Staff 2 (Bass, voice 5)
    for (const staff of [1, 2] as const) {
      if (staff === 2) {
        xmlParts.push(`<backup><duration>${totalGrid}</duration></backup>`);
      }

      const staffNotes = notesInBar.filter(n => n.staff === staff);
      const carryIn = carryByStaff[staff];

      // Group notes by quantized startGrid
      const chordBuckets = new Map<number, { pitches: Map<number, number>; incomingTie: Set<number> }>();
      const ensureBucket = (g: number) => {
        let b = chordBuckets.get(g);
        if (!b) {
          b = { pitches: new Map(), incomingTie: new Set() };
          chordBuckets.set(g, b);
        }
        return b;
      };

      for (const n of staffNotes) {
        const gStart = Math.min(totalGrid - 1, Math.max(0, timeToGrid(m, n.start, totalGrid)));
        const b = ensureBucket(gStart);
        b.pitches.set(n.pitch, Math.max(b.pitches.get(n.pitch) ?? 0, n.end));
      }

      // Add carried-over notes from previous measure at grid 0 unless re-struck at grid 0
      const b0Existing = chordBuckets.get(0);
      if (carryIn.size > 0) {
        for (const [pitch, remEnd] of carryIn) {
          if (remEnd > m.start + 0.06 && !b0Existing?.pitches.has(pitch)) {
            const b0 = ensureBucket(0);
            b0.pitches.set(pitch, remEnd);
            b0.incomingTie.add(pitch);
          }
        }
      }

      const sortedStarts = [...chordBuckets.keys()].sort((a, b) => a - b);
      const atoms: Atom[] = [];

      if (sortedStarts.length === 0) {
        atoms.push({
          startGrid: 0,
          grid: totalGrid,
          type: 'whole',
          dots: 0,
          pitches: [],
          tieStart: new Set(),
          tieStop: new Set(),
          wholeMeasureRest: true,
        });
      } else {
        let cursor = 0;
        for (let ci = 0; ci < sortedStarts.length; ci++) {
          const gStart = sortedStarts[ci];
          const nextOnset = ci + 1 < sortedStarts.length ? sortedStarts[ci + 1] : totalGrid;
          if (gStart > cursor) {
            const restSegs = splitSpan(cursor, gStart, totalGrid, beatGrid, true);
            for (const seg of restSegs) {
              atoms.push({
                startGrid: seg.startGrid,
                grid: seg.grid,
                type: seg.type,
                dots: seg.dots,
                pitches: [],
                tieStart: new Set(),
                tieStop: new Set(),
              });
            }
          }

          const bucket = chordBuckets.get(gStart)!;
          // Keep up to 5 clearest notes in one hand, always preserving incoming tied notes
          let pitchEntries = [...bucket.pitches.entries()].sort((a, b) => a[0] - b[0]);
          if (pitchEntries.length > 5) {
            const mustKeep = pitchEntries.filter(([p]) => bucket.incomingTie.has(p));
            const others = pitchEntries.filter(([p]) => !bucket.incomingTie.has(p));
            const slots = Math.max(0, 5 - mustKeep.length);
            const picked = slots > 0
              ? [others[0], ...others.slice(Math.max(1, others.length - (slots - 1)))].slice(0, slots)
              : [];
            pitchEntries = [...mustKeep, ...picked].sort((a, b) => a[0] - b[0]);
          }
          const pitches = pitchEntries.map(([p]) => p);
          const maxEndSec = Math.max(...pitchEntries.map(([, e]) => e));
          let rawEndGrid = timeToGrid(m, maxEndSec, totalGrid);
          if (nextOnset - rawEndGrid === 1 && rawEndGrid - gStart >= 2) {
            rawEndGrid = nextOnset;
          }
          const gEnd = Math.max(gStart + 1, Math.min(nextOnset, rawEndGrid));

          // Only tie across the barline if there is a next measure and that pitch is not re-struck on the next downbeat
          const barTieOut = new Set<number>();
          if (gEnd === totalGrid && nextOnset === totalGrid && mi + 1 < measures.length) {
            const nextM = measures[mi + 1];
            const nextTotalGrid = Math.max(4, Math.round(nextM.quarters * DIVISIONS));
            const nextRestruckAtZero = new Set(
              measureNotes[mi + 1]
                .filter(n => n.staff === staff && timeToGrid(nextM, n.start, nextTotalGrid) === 0)
                .map(n => n.pitch),
            );
            for (const [p, endSec] of pitchEntries) {
              if (endSec > m.end + 0.08 && endSec > nextM.start + 0.06 && !nextRestruckAtZero.has(p)) {
                barTieOut.add(p);
                nextCarry[staff].set(p, endSec);
              }
            }
          }

          const noteSegs = splitSpan(gStart, gEnd, totalGrid, beatGrid, false);
          for (let si = 0; si < noteSegs.length; si++) {
            const seg = noteSegs[si];
            const tieStop = new Set<number>();
            const tieStart = new Set<number>();
            if (si === 0) {
              for (const p of bucket.incomingTie) if (pitches.includes(p)) tieStop.add(p);
            } else {
              for (const p of pitches) tieStop.add(p);
            }
            if (si + 1 < noteSegs.length) {
              for (const p of pitches) tieStart.add(p);
            } else {
              for (const p of barTieOut) tieStart.add(p);
            }
            atoms.push({
              startGrid: seg.startGrid,
              grid: seg.grid,
              type: seg.type,
              dots: seg.dots,
              pitches,
              tieStart,
              tieStop,
            });
          }

          cursor = gEnd;
          if (cursor < nextOnset) {
            const restSegs = splitSpan(cursor, nextOnset, totalGrid, beatGrid, true);
            for (const seg of restSegs) {
              atoms.push({
                startGrid: seg.startGrid,
                grid: seg.grid,
                type: seg.type,
                dots: seg.dots,
                pitches: [],
                tieStart: new Set(),
                tieStop: new Set(),
              });
            }
            cursor = nextOnset;
          }
        }
      }

      const beams = computeBeams(atoms, beatGrid);
      const voice = staff === 1 ? 1 : 5;
      const accidentalState = new Map<string, number>();

      let chordMarkIdx = 0;
      let dynEmitted = false;

      for (let ai = 0; ai < atoms.length; ai++) {
        const atom = atoms[ai];
        if (staff === 1) {
          while (chordMarkIdx < chordMarks.length && chordMarks[chordMarkIdx].grid <= atom.startGrid) {
            xmlParts.push(renderHarmonyXml(chordMarks[chordMarkIdx++].symbol));
          }
          if (!dynEmitted && measureDynamic && measureDynamic.grid <= atom.startGrid) {
            xmlParts.push(
              `<direction placement="below"><direction-type><dynamics><${measureDynamic.dyn}/></dynamics></direction-type><staff>1</staff></direction>`,
            );
            dynEmitted = true;
          }
        }

        const elemId = `m${m.measureNumber}_s${staff}_a${ai}`;
        eventTimes.set(elemId, gridToTime(m, atom.startGrid, totalGrid));

        if (atom.pitches.length === 0) {
          if (atom.wholeMeasureRest) {
            xmlParts.push(
              `<note id="${elemId}"><rest measure="yes"/><duration>${atom.grid}</duration><voice>${voice}</voice><staff>${staff}</staff></note>`,
            );
          } else {
            let dotsXml = '';
            for (let d = 0; d < atom.dots; d++) dotsXml += '<dot/>';
            xmlParts.push(
              `<note id="${elemId}"><rest/><duration>${atom.grid}</duration><voice>${voice}</voice><type>${atom.type}</type>${dotsXml}<staff>${staff}</staff></note>`,
            );
          }
          continue;
        }

        const atomBeams = beams.get(ai);
        for (let pi = 0; pi < atom.pitches.length; pi++) {
          const pitch = atom.pitches[pi];
          const { step, alter, octave } = spellPitch(pitch, fifths, alters);
          const stepKey = `${step}${octave}`;
          const activeAlter = accidentalState.get(stepKey) ?? alters[step];
          const isTiedContinuation = atom.tieStop.has(pitch);
          const showAccidental = !isTiedContinuation && alter !== activeAlter;
          accidentalState.set(stepKey, alter);

          const noteIdAttr = pi === 0 ? ` id="${elemId}"` : '';
          let noteXml = `<note${noteIdAttr}>`;
          if (pi > 0) noteXml += '<chord/>';
          noteXml += `<pitch><step>${step}</step>${alter ? `<alter>${alter}</alter>` : ''}<octave>${octave}</octave></pitch>`;
          noteXml += `<duration>${atom.grid}</duration>`;
          if (isTiedContinuation) noteXml += '<tie type="stop"/>';
          if (atom.tieStart.has(pitch)) noteXml += '<tie type="start"/>';
          noteXml += `<voice>${voice}</voice><type>${atom.type}</type>`;
          for (let d = 0; d < atom.dots; d++) noteXml += '<dot/>';
          if (showAccidental && ACCIDENTAL_XML[alter] !== undefined) {
            noteXml += `<accidental>${ACCIDENTAL_XML[alter]}</accidental>`;
          }
          noteXml += `<staff>${staff}</staff>`;
          if (pi === 0 && atomBeams) {
            for (const [lvl, val] of Object.entries(atomBeams)) {
              noteXml += `<beam number="${lvl}">${val}</beam>`;
            }
          }
          if (isTiedContinuation || atom.tieStart.has(pitch)) {
            noteXml += '<notations>';
            if (isTiedContinuation) noteXml += '<tied type="stop"/>';
            if (atom.tieStart.has(pitch)) noteXml += '<tied type="start"/>';
            noteXml += '</notations>';
          }
          noteXml += '</note>';
          xmlParts.push(noteXml);
        }
      }

      if (staff === 1) {
        while (chordMarkIdx < chordMarks.length) {
          xmlParts.push(renderHarmonyXml(chordMarks[chordMarkIdx++].symbol));
        }
      }
    }

    if (mi === measures.length - 1) {
      xmlParts.push('<barline location="right"><bar-style>light-heavy</bar-style></barline>');
    }
    xmlParts.push('</measure>');
    carryByStaff = nextCarry;
  }

  xmlParts.push('</part></score-partwise>');
  return { xml: xmlParts.join(''), measures, eventTimes };
}
