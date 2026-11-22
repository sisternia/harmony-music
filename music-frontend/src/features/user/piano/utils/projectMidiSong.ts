import type { CoverView, SourceView } from '../../../studio/api';
import type { PianoSong, SongNote } from './transcription';
import { normalizeMidiNotes } from './normalizeMidiNotes';
import { assignPianoHands } from './assignPianoHands';

function inferTempoAndMeter(source?: SourceView | null): { bpm: number; meter: string } {
  if (!source?.beats?.length) return { bpm: 120, meter: '4/4' };
  const intervals: number[] = [];
  for (let i = 1; i < Math.min(source.beats.length, 200); i++) {
    const dt = source.beats[i] - source.beats[i - 1];
    if (dt > 0.18 && dt < 2.0) intervals.push(dt);
  }
  intervals.sort((a, b) => a - b);
  const medianDt = intervals.length ? intervals[intervals.length >> 1] : 0.5;
  const bpm = Math.max(30, Math.min(260, Math.round(60 / medianDt)));

  let beatsPerBar = 4;
  if (source.downbeats && source.downbeats.length >= 2) {
    const counts: number[] = [];
    for (let i = 0; i < Math.min(source.downbeats.length - 1, 24); i++) {
      const a = source.downbeats[i] - 0.05;
      const b = source.downbeats[i + 1] - 0.05;
      let count = 0;
      for (const beat of source.beats) {
        if (beat >= a && beat < b) count++;
        else if (beat >= b) break;
      }
      if (count >= 2 && count <= 12) counts.push(count);
    }
    if (counts.length) {
      counts.sort((a, b) => a - b);
      beatsPerBar = counts[counts.length >> 1];
    }
  }
  return { bpm, meter: `${beatsPerBar}/4` };
}

function songFromNotes(
  title: string,
  notes: SongNote[],
  duration: number,
  source?: SourceView | null,
  musicXml?: string,
): PianoSong {
  if (!notes.length) throw new Error('Project này không có nốt MIDI.');
  const pianoNotes = assignPianoHands(normalizeMidiNotes(notes));
  const { bpm, meter } = inferTempoAndMeter(source);
  const chords = source?.chords?.map(([time, label]) => ({ time, label })) ?? [];
  return {
    id: `project-midi-${Date.now()}`,
    midiRoll: true,
    title,
    duration: Math.max(duration, ...pianoNotes.map(note => note.end)),
    bpm,
    meter,
    key: '',
    beats: source?.beats,
    downbeats: source?.downbeats,
    musicXml,
    notes: pianoNotes,
    chords,
    pages: [],
    timeline: [],
    originalAudio: null,
    downloads: {},
    notice: '',
  };
}

export function sourceProjectMidi(title: string, view: SourceView): PianoSong {
  const notes: SongNote[] = view.notes.map(([start, duration, pitch, velocity], index) => ({
    id: `source-${index}`,
    pitch,
    start,
    end: start + duration,
    keyEnd: start + duration,
    velocity,
    hand: pitch < 60 ? 'L' : 'R',
  }));
  return songFromNotes(`${title} · Source MIDI`, notes, view.duration, view);
}

export function takeProjectMidi(
  title: string,
  view: CoverView,
  sourceView?: SourceView | null,
  musicXml?: string,
): PianoSong {
  const notes: SongNote[] = view.notes.map(([start, duration, pitch, velocity, sustainEnd], index) => ({
    id: `take-${index}`,
    pitch,
    start,
    end: Math.max(start + duration, sustainEnd),
    keyEnd: start + duration,
    velocity,
    hand: pitch < 60 ? 'L' : 'R',
  }));
  return songFromNotes(title, notes, view.duration, sourceView, musicXml);
}
