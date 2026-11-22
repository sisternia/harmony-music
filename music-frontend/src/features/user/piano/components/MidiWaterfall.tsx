import React, { useEffect, useMemo, useRef, useState } from 'react';
import { colors } from '@/theme/colors';
import { type PianoSong } from '../utils/transcription';
import { visibleNoteRange } from '../hooks/usePianoPlayback';

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_PCS = new Set([1, 3, 6, 8, 10]);

const PITCH_NAME: string[] = Array.from({ length: 128 }, (_, p) => `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`);
const PITCH_IS_BLACK: boolean[] = Array.from({ length: 128 }, (_, p) => BLACK_PCS.has(p % 12));

const HAND_COLORS = {
  L: { fill: colors.leftHand, glow: colors.leftHandLight, cap: colors.lavender },
  R: { fill: colors.rightHand, glow: colors.rightHandLight, cap: colors.coral },
} as const;

function estimatedChords(song: PianoSong) {
  if (song.chords?.length) return song.chords;
  const result: { time: number; label: string }[] = [];
  const names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
  const patterns: [string, number[]][] = [
    ['maj7', [0, 4, 7, 11]],
    ['m7', [0, 3, 7, 10]],
    ['7', [0, 4, 7, 10]],
    ['m', [0, 3, 7]],
    ['', [0, 4, 7]],
  ];
  for (let time = 0; time < song.duration; time += 1) {
    const [from, to] = visibleNoteRange(song, time, 1);
    const active = song.notes
      .slice(from, to)
      .filter(note => note.start < time + 1 && note.end > time && note.pitch >= 24 && note.pitch <= 96);
    if (active.length < 3) continue;
    const weight = new Float32Array(12);
    for (const note of active) weight[note.pitch % 12] += Math.min(note.end, time + 1) - Math.max(note.start, time);
    let best = '';
    let score = 0;
    for (let root = 0; root < 12; root++) {
      for (const [suffix, intervals] of patterns) {
        const matched = intervals.reduce((sum, offset) => sum + weight[(root + offset) % 12], 0);
        const value = matched - (weight.reduce((sum, n) => sum + n, 0) - matched) * 0.7;
        if (value > score) {
          score = value;
          best = names[root] + suffix;
        }
      }
    }
    if (best && result.at(-1)?.label !== best) result.push({ time, label: best });
  }
  return result;
}

function findChordAt(chords: { time: number; label: string }[], time: number): string {
  if (!chords.length || time < chords[0].time) return '';
  let low = 0;
  let high = chords.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >>> 1;
    if (chords[mid].time <= time) low = mid;
    else high = mid - 1;
  }
  return chords[low].label;
}

const MidiWaterfall = React.memo(function MidiWaterfall({
  song,
  readTime,
  playing,
  position,
}: {
  song: PianoSong;
  readTime: () => number;
  playing: boolean;
  position: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const chordText = useRef<HTMLElement>(null);
  const [direction, setDirection] = useState<'down' | 'up'>('down');
  const chords = useMemo(() => estimatedChords(song), [song]);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d', { alpha: false });
    if (!element || !context) return;

    let frame = 0;
    let width = 0;
    let height = 0;
    let ratio = 1;
    let lastChord = '__init__';

    // Indexed directly by MIDI pitch (21..108) for O(1) lookup with zero string allocations
    const keyX = new Float32Array(128);
    const keyW = new Float32Array(128);

    const syncLayout = () => {
      const rect = element.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.max(1, Math.round(width * ratio));
      const targetH = Math.max(1, Math.round(height * ratio));
      if (element.width !== targetW || element.height !== targetH) {
        element.width = targetW;
        element.height = targetH;
      }
      keyX.fill(0);
      keyW.fill(0);
      const page = element.closest('.piano-page');
      const keys = page?.querySelectorAll<HTMLButtonElement>('.keyboard .top-keys button[data-note]');
      const nameToPitch = new Map<string, number>();
      for (let p = 21; p <= 108; p++) nameToPitch.set(PITCH_NAME[p], p);
      keys?.forEach(key => {
        const pitch = nameToPitch.get(key.dataset.note || '');
        if (pitch === undefined) return;
        const box = key.getBoundingClientRect();
        keyX[pitch] = box.left + box.width * 0.5 - rect.left;
        keyW[pitch] = box.width;
      });
    };

    const drawFrame = () => {
      if (!width || !height) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = colors.surface;
      context.fillRect(0, 0, width, height);

      const now = playing ? readTime() : position;
      const lead = 3.6;
      const hit = height - 2;
      const pixels = hit / lead;

      if (direction === 'down') {
        // Top-down: upcoming notes fall smoothly from top toward the keyboard at bottom
        const [start, end] = visibleNoteRange(song, now, lead);
        for (let index = start; index < end; index++) {
          const note = song.notes[index];
          const pitch = note.pitch;
          if (pitch < 21 || pitch > 108) continue;
          const kw = keyW[pitch];
          if (kw <= 0) continue;
          const keyEnd = note.keyEnd ?? note.end;
          if (keyEnd <= now) continue;

          const rawBottom = hit - (note.start - now) * pixels;
          const rawTop = rawBottom - Math.max(5, (keyEnd - note.start) * pixels);
          const top = Math.max(0, rawTop);
          const bottom = Math.min(hit, rawBottom);
          const barH = bottom - top;
          if (barH <= 0.5) continue;

          const isBlack = PITCH_IS_BLACK[pitch];
          const palette = HAND_COLORS[note.hand === 'L' ? 'L' : 'R'];

          const barW = Math.max(4, kw * (isBlack ? 0.82 : 0.86));
          const x = keyX[pitch] - barW * 0.5;
          const radius = Math.min(4, barW * 0.35, barH * 0.45);

          // Fast outer glow without expensive CPU shadowBlur
          context.fillStyle = palette.glow;
          context.beginPath();
          context.roundRect(x - 2, Math.max(0, top - 2), barW + 4, Math.min(hit - top + 2, barH + 4), radius + 2);
          context.fill();

          // Crisp main note bar
          context.fillStyle = palette.fill;
          context.beginPath();
          context.roundRect(x, top, barW, barH, radius);
          context.fill();

          // Subtle leading-edge highlight when touching or approaching the keys
          if (barH >= 6) {
            context.fillStyle = palette.cap;
            context.fillRect(x + 1.5, bottom - 2.5, Math.max(1, barW - 3), 2);
          }
        }
      } else {
        // Bottom-up: played notes rise smoothly upward from the keyboard at bottom
        const windowStart = Math.max(0, now - lead);
        const [start, end] = visibleNoteRange(song, windowStart, lead + 0.05);
        for (let index = start; index < end; index++) {
          const note = song.notes[index];
          const pitch = note.pitch;
          if (pitch < 21 || pitch > 108 || note.start > now) continue;
          const kw = keyW[pitch];
          if (kw <= 0) continue;
          const keyEnd = Math.min(now, note.keyEnd ?? note.end);
          if (keyEnd <= windowStart) continue;

          const rawTop = hit - (now - note.start) * pixels;
          const rawBottom = hit - Math.max(0, now - keyEnd) * pixels;
          const top = Math.max(0, rawTop);
          const bottom = Math.min(hit, Math.max(top + 5, rawBottom));
          const barH = bottom - top;
          if (barH <= 0.5 || top >= hit) continue;

          const isBlack = PITCH_IS_BLACK[pitch];
          const palette = HAND_COLORS[note.hand === 'L' ? 'L' : 'R'];

          const barW = Math.max(4, kw * (isBlack ? 0.82 : 0.86));
          const x = keyX[pitch] - barW * 0.5;
          const radius = Math.min(4, barW * 0.35, barH * 0.45);

          context.fillStyle = palette.glow;
          context.beginPath();
          context.roundRect(x - 2, Math.max(0, top - 2), barW + 4, barH + 4, radius + 2);
          context.fill();

          context.fillStyle = palette.fill;
          context.beginPath();
          context.roundRect(x, top, barW, barH, radius);
          context.fill();
        }
      }

      const activeChord = findChordAt(chords, now);
      if (activeChord !== lastChord) {
        lastChord = activeChord;
        if (chordText.current) chordText.current.textContent = activeChord;
      }
    };

    const loop = () => {
      drawFrame();
      if (playing) frame = requestAnimationFrame(loop);
    };

    syncLayout();
    if (playing) {
      frame = requestAnimationFrame(loop);
    } else {
      drawFrame();
    }

    const observer = new ResizeObserver(() => {
      syncLayout();
      drawFrame();
    });
    observer.observe(element);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [song, chords, readTime, playing, position, direction]);

  return (
    <div className="piano-midi-waterfall">
      <canvas ref={canvas} aria-label="Các nốt MIDI trên phím đàn" role="img" />
      <div className="piano-midi-hud">
        <div className="piano-midi-chord-lane">
          <span>HỢP ÂM</span>
          <strong ref={chordText} />
        </div>
        <div className="piano-midi-hand-legend" aria-label="Phân biệt tay trái và tay phải">
          <span className="hand-chip hand-left">
            <i aria-hidden="true" /> Tay trái
          </span>
          <span className="hand-chip hand-right">
            <i aria-hidden="true" /> Tay phải
          </span>
          <button
            type="button"
            className="waterfall-dir-btn"
            onClick={() => setDirection(d => (d === 'down' ? 'up' : 'down'))}
            title={direction === 'down' ? 'Đang chạy Top-down (rơi xuống) · Nhấn để đổi sang Bottom-up' : 'Đang chạy Bottom-up (bay lên) · Nhấn để đổi sang Top-down'}
          >
            {direction === 'down' ? '↓ Top-down' : '↑ Bottom-up'}
          </button>
        </div>
      </div>
    </div>
  );
});

export default MidiWaterfall;
