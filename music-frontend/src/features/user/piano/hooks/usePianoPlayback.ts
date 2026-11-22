import { useCallback, useEffect, useRef, useState } from 'react';
import { lowerNoteIndex, midiNoteName, type PianoSong, type SongNote } from '../utils/transcription';

export type PlaybackAudio = {
  isReady: boolean;
  preparePlayback: () => Promise<void>;
  audioTime: () => number;
  scheduleNote: (note: string, duration: number, time: number, velocity: number) => void;
  allOff: () => void;
};

export function usePianoPlayback(song: PianoSong, audio: PlaybackAudio) {
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [seekVersion, setSeekVersion] = useState(0);
  const [error, setError] = useState('');
  const cursor = useRef(0);
  const requested = useRef(0);
  const mounted = useRef(true);
  const running = useRef<{ offset: number; anchor: number; speed: number } | null>(null);
  const notes = song.notes;
  const { audioTime, scheduleNote, allOff, preparePlayback } = audio;
  const readTime = useCallback(() => {
    const state = running.current;
    if (state) {
      cursor.current = Math.min(song.duration, Math.max(state.offset,
        state.offset + (audioTime() - state.anchor) * state.speed));
    }
    return cursor.current;
  }, [audioTime, song.duration]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (!playing) return;
    const offset = readTime();
    const anchor = audioTime() + 0.04;
    running.current = { offset, anchor, speed };
    // Include notes spanning a seek point; audio and visuals share the same clock.
    const upcoming = notes.filter(note => note.end > offset);
    let next = 0;
    const tick = () => {
      const now = audioTime();
      readTime();
      const horizon = offset + (now + 0.12 - anchor) * speed;
      while (next < upcoming.length && upcoming[next].start <= horizon) {
        const note = upcoming[next++];
        const start = Math.max(offset, note.start);
        scheduleNote(midiNoteName(note.pitch), (note.end - start) / speed,
          Math.max(now, anchor + (start - offset) / speed), note.velocity / 127);
      }
      if (cursor.current >= song.duration) {
        setPlaying(false);
        setPosition(song.duration);
      }
    };
    tick();
    const timer = setInterval(tick, 25);
    return () => { clearInterval(timer); readTime(); running.current = null; allOff(); };
  }, [playing, position, speed, seekVersion, song.duration, notes, audioTime, scheduleNote, allOff, readTime]);

  const pause = useCallback(() => {
    requested.current++;
    readTime();
    running.current = null;
    setPlaying(false);
    setPosition(cursor.current);
    allOff();
  }, [allOff, readTime]);

  const stop = useCallback(() => {
    requested.current++;
    running.current = null;
    cursor.current = 0;
    setPlaying(false);
    setPosition(0);
    allOff();
  }, [allOff]);

  const play = useCallback(async () => {
    const request = ++requested.current;
    setError('');
    try {
      await preparePlayback();
      if (!mounted.current || request !== requested.current) return;
      if (cursor.current >= song.duration) {
        cursor.current = 0;
        setPosition(0);
      }
      setPlaying(true);
    } catch (reason) {
      if (mounted.current) setError(reason instanceof Error ? reason.message : 'Không phát được piano.');
    }
  }, [preparePlayback, song.duration]);

  const seek = useCallback((value: number) => {
    allOff();
    running.current = null;
    cursor.current = Math.min(song.duration, Math.max(0, value));
    setPosition(cursor.current);
    setSeekVersion(value => value + 1);
  }, [allOff, song.duration]);

  const changeSpeed = useCallback((value: number) => {
    readTime();
    setPosition(cursor.current);
    setSpeed(value);
  }, [readTime]);

  useEffect(() => {
    const hidden = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', hidden);
    return () => document.removeEventListener('visibilitychange', hidden);
  }, [pause]);

  return { playing, cursor, readTime, position, speed, error, play, pause, stop, seek, changeSpeed };
}

const prefixEndCache = new WeakMap<SongNote[], Float64Array>();

export function visibleNoteRange(song: PianoSong, time: number, lead: number) {
  let prefixEnds = prefixEndCache.get(song.notes);
  if (!prefixEnds) {
    prefixEnds = new Float64Array(song.notes.length);
    let maximum = -Infinity;
    for (let index = 0; index < song.notes.length; index++) {
      maximum = Math.max(maximum, song.notes[index].end);
      prefixEnds[index] = maximum;
    }
    prefixEndCache.set(song.notes, prefixEnds);
  }
  let low = 0;
  let high = prefixEnds.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (prefixEnds[middle] <= time) low = middle + 1;
    else high = middle;
  }
  return [low, lowerNoteIndex(song.notes, time + lead)] as const;
}
