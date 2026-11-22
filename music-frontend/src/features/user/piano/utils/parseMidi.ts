import type { MidiBar, PianoSong, SongNote } from './transcription';
import { normalizeMidiNotes } from './normalizeMidiNotes';
import { assignPianoHands } from './assignPianoHands';

type Event = {
  tick: number;
  order: number;
  kind: 'on' | 'off' | 'tempo' | 'meter' | 'key' | 'pedal' | 'chord';
  track?: number;
  channel?: number;
  pitch?: number;
  value?: number;
  beats?: number;
  beatType?: number;
  fifths?: number;
  minor?: boolean;
  text?: string;
};

function parseChordMarker(raw: string): string | null {
  const text = raw.trim();
  if (!text || text === 'N' || text === 'N.C.') return null;
  if (text.includes(':')) {
    const [root, restRaw] = text.split(':', 2);
    if (!/^[A-G](?:#|b)?$/.test(root)) return null;
    const [qualRaw, bass] = restRaw.split('/', 2);
    const quality = qualRaw === 'maj' ? '' : qualRaw;
    return `${root}${quality}${bass ? `/${bass}` : ''}`;
  }
  if (/^[A-G](?:#|b)?(?:m|maj|min|dim|aug|sus|add|M)?[0-9+-]*(?:sus[24]|add\d+)?(?:\/[A-G](?:#|b)?)?$/.test(text)) {
    return text;
  }
  return null;
}

const MAJOR_KEYS = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
const MINOR_KEYS = ['Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m'];

export async function parseMidi(file: File): Promise<PianoSong> {
  const data = new DataView(await file.arrayBuffer());
  let at = 0;
  const u8 = () => data.getUint8(at++);
  const u16 = () => { const value = data.getUint16(at); at += 2; return value; };
  const u32 = () => { const value = data.getUint32(at); at += 4; return value; };
  const id = () => String.fromCharCode(u8(), u8(), u8(), u8());
  const variable = () => { let value = 0, byte = 0; do { byte = u8(); value = (value << 7) | (byte & 127); } while (byte & 128); return value; };
  if (id() !== 'MThd' || u32() !== 6) throw new Error('Tệp MIDI không hợp lệ.');
  const format = u16(), tracks = u16(), division = u16();
  if (format > 1 || division & 0x8000 || !division) throw new Error('Định dạng MIDI này chưa được hỗ trợ.');
  const events: Event[] = [];
  for (let track = 0; track < tracks; track++) {
    if (id() !== 'MTrk') throw new Error('Thiếu dữ liệu track MIDI.');
    const end = at + u32();
    if (end > data.byteLength) throw new Error('Tệp MIDI bị thiếu dữ liệu.');
    let tick = 0, status = 0;
    while (at < end) {
      tick += variable();
      const first = data.getUint8(at);
      if (first & 0x80) status = u8();
      if (!status) throw new Error('Sự kiện MIDI không hợp lệ.');
      if (status === 0xff) {
        const type = u8(), size = variable(), from = at;
        at += size;
        if (at > end) throw new Error('Sự kiện MIDI bị thiếu dữ liệu.');
        if (type === 0x51 && size === 3) {
          events.push({ tick, order: events.length, kind: 'tempo', value: (data.getUint8(from) << 16) | (data.getUint8(from + 1) << 8) | data.getUint8(from + 2) });
        } else if (type === 0x58 && size >= 2) {
          const beats = Math.max(1, data.getUint8(from));
          const beatType = 1 << Math.min(5, data.getUint8(from + 1));
          events.push({ tick, order: events.length, kind: 'meter', beats, beatType });
        } else if (type === 0x59 && size >= 2) {
          const fifths = Math.max(-7, Math.min(7, data.getInt8(from)));
          const minor = data.getUint8(from + 1) === 1;
          events.push({ tick, order: events.length, kind: 'key', fifths, minor });
        } else if (type === 0x06 || type === 0x01) {
          const raw = new TextDecoder().decode(new Uint8Array(data.buffer, from, size));
          const chord = parseChordMarker(raw);
          if (chord) events.push({ tick, order: events.length, kind: 'chord', text: chord });
        }
      } else if (status === 0xf0 || status === 0xf7) at += variable();
      else {
        const type = status & 0xf0, channel = status & 15, pitch = u8();
        const value = type === 0xc0 || type === 0xd0 ? 0 : u8();
        if (type === 0x90 || type === 0x80) events.push({ tick, order: events.length, kind: type === 0x90 && value ? 'on' : 'off', track, channel, pitch, value });
        if (type === 0xb0 && pitch === 64) events.push({ tick, order: events.length, kind: 'pedal', track, channel, value });
      }
    }
    at = end;
  }
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  let tempo = 500000, lastTick = 0, seconds = 0;
  const tempoPoints: { tick: number; seconds: number; tempo: number }[] = [{ tick: 0, seconds: 0, tempo: 500000 }];
  const meterChanges: { tick: number; beats: number; beatType: number }[] = [];
  let parsedFifths: number | undefined;
  let parsedKeyName = '';

  const active = new Map<string, { start: number; velocity: number; track: number }[]>();
  const pedal = new Set<number>();
  const pending = new Map<number, SongNote[]>();
  const notes: SongNote[] = [], noteTracks: number[] = [], chords: { time: number; label: string }[] = [];
  const releasePending = (channel: number) => { for (const note of pending.get(channel) ?? []) note.end = seconds; pending.delete(channel); };

  for (const event of events) {
    seconds += (event.tick - lastTick) * tempo / division / 1e6;
    lastTick = event.tick;
    if (event.kind === 'tempo') {
      tempo = event.value || tempo;
      if (tempoPoints[tempoPoints.length - 1].tick === event.tick) {
        tempoPoints[tempoPoints.length - 1].tempo = tempo;
      } else {
        tempoPoints.push({ tick: event.tick, seconds, tempo });
      }
      continue;
    }
    if (event.kind === 'meter' && event.beats && event.beatType) {
      if (meterChanges.length && meterChanges[meterChanges.length - 1].tick === event.tick) {
        meterChanges[meterChanges.length - 1] = { tick: event.tick, beats: event.beats, beatType: event.beatType };
      } else {
        meterChanges.push({ tick: event.tick, beats: event.beats, beatType: event.beatType });
      }
      continue;
    }
    if (event.kind === 'key' && event.fifths !== undefined && parsedFifths === undefined) {
      parsedFifths = event.fifths;
      parsedKeyName = (event.minor ? MINOR_KEYS : MAJOR_KEYS)[event.fifths + 7] || '';
      continue;
    }
    if (event.kind === 'chord') { chords.push({ time: seconds, label: event.text || '' }); continue; }
    const channel = event.channel ?? 0;
    if (channel === 9) continue;
    if (event.kind === 'pedal') { if ((event.value ?? 0) >= 64) pedal.add(channel); else { pedal.delete(channel); releasePending(channel); } continue; }
    const key = `${event.track ?? 0}:${channel}:${event.pitch}`;
    if (event.kind === 'on') { const voices = active.get(key) ?? []; voices.push({ start: seconds, velocity: event.value ?? 80, track: event.track ?? 0 }); active.set(key, voices); }
    if (event.kind === 'off') {
      const voice = active.get(key)?.shift();
      if (!voice) continue;
      const note: SongNote = { id: String(notes.length), pitch: event.pitch ?? 60, start: voice.start, end: Math.max(seconds, voice.start + .02), keyEnd: Math.max(seconds, voice.start + .02), velocity: voice.velocity, hand: (event.pitch ?? 60) < 60 ? 'L' : 'R' };
      notes.push(note);
      noteTracks.push(voice.track);
      if (pedal.has(channel)) { const waiting = pending.get(channel) ?? []; waiting.push(note); pending.set(channel, waiting); }
    }
  }
  for (const channel of pending.keys()) releasePending(channel);
  for (const [key, voices] of active) {
    const pitchNum = Number(key.split(':')[2]);
    for (const voice of voices) {
      notes.push({ id: String(notes.length), pitch: pitchNum, start: voice.start, end: Math.max(seconds, voice.start + .02), keyEnd: Math.max(seconds, voice.start + .02), velocity: voice.velocity, hand: pitchNum < 60 ? 'L' : 'R' });
      noteTracks.push(voice.track);
    }
  }
  if (!notes.length) throw new Error('Tệp MIDI không có nốt nhạc.');

  const handedNotes = assignPianoHands(notes, noteTracks);
  const pianoNotes = normalizeMidiNotes(handedNotes);
  const duration = Math.max(seconds, ...pianoNotes.map(note => note.end));

  const tickToSec = (targetTick: number) => {
    let low = 0, high = tempoPoints.length - 1;
    while (low < high) {
      const mid = (low + high + 1) >>> 1;
      if (tempoPoints[mid].tick <= targetTick) low = mid;
      else high = mid - 1;
    }
    const seg = tempoPoints[low];
    return seg.seconds + ((targetTick - seg.tick) * seg.tempo) / division / 1e6;
  };

  if (!meterChanges.length || meterChanges[0].tick > 0) {
    meterChanges.unshift({ tick: 0, beats: 4, beatType: 4 });
  }

  const midiBars: MidiBar[] = [];
  let barTick = 0;
  let meterIdx = 0;
  const maxBars = 1200;
  while (midiBars.length < maxBars) {
    while (meterIdx + 1 < meterChanges.length && meterChanges[meterIdx + 1].tick <= barTick) meterIdx++;
    const { beats, beatType } = meterChanges[meterIdx];
    const fullBarTicks = Math.max(1, Math.round((division * beats * 4) / beatType));
    const nextMeterTick = meterIdx + 1 < meterChanges.length ? meterChanges[meterIdx + 1].tick : Infinity;
    const endTick = nextMeterTick > barTick && nextMeterTick < barTick + fullBarTicks ? nextMeterTick : barTick + fullBarTicks;
    const start = tickToSec(barTick);
    if (start >= duration - 0.005 && midiBars.length > 0) break;
    const end = tickToSec(endTick);
    const quarters = (endTick - barTick) / division;
    const localBpm = quarters > 0 && end > start ? Math.max(24, Math.min(300, (quarters * 60) / (end - start))) : 120;
    midiBars.push({ start, end, beats, beatType, bpm: localBpm });
    barTick = endTick;
    if (end >= duration + 0.5) break;
  }

  const firstNoteTime = pianoNotes[0].start;
  // Merge a tiny silent pickup metadata bar (e.g. SheetSage pre-downbeat offset < 0.35s) into bar 1
  if (
    midiBars.length >= 2 &&
    midiBars[0].end - midiBars[0].start < 0.35 &&
    firstNoteTime >= midiBars[0].end - 0.03
  ) {
    const firstReal = midiBars[1];
    const quarters = (firstReal.beats * 4) / firstReal.beatType;
    midiBars.splice(0, 2, {
      start: 0,
      end: firstReal.end,
      beats: firstReal.beats,
      beatType: firstReal.beatType,
      bpm: Math.max(24, Math.min(300, (quarters * 60) / Math.max(0.2, firstReal.end))),
    });
  }

  // Pick representative BPM and meter from full musical bars rather than transient pickup bars
  const musicalBars = midiBars.filter(b => b.end - b.start >= 0.6);
  const sampleBars = musicalBars.length ? musicalBars.slice(0, 32) : midiBars.slice(0, 32);
  const sortedBpms = sampleBars.map(b => b.bpm).sort((a, b) => a - b);
  const initialBpm = sortedBpms.length
    ? Math.round(sortedBpms[sortedBpms.length >> 1])
    : Math.round(60_000_000 / tempoPoints[0].tempo);
  const refBar = musicalBars[0] ?? midiBars[0];
  const initialMeter = refBar ? `${refBar.beats}/${refBar.beatType}` : '4/4';

  return {
    id: `midi-${Date.now()}`,
    midiRoll: true,
    title: file.name.replace(/\.midi?$/i, ''),
    duration,
    bpm: initialBpm,
    meter: initialMeter,
    key: parsedKeyName,
    fifths: parsedFifths,
    midiBars,
    notes: pianoNotes,
    chords,
    pages: [],
    timeline: [],
    originalAudio: null,
    downloads: {},
    notice: '',
  };
}
