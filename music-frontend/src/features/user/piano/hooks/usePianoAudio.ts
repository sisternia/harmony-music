import { useCallback, useEffect, useRef } from 'react';
import { CacheStorage, SplendidGrandPiano, type Smplr, type StopFn } from 'smplr';

const NOTE_SEMITONES: Record<string, number> = {
  C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4,
  F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11,
};

function noteToFrequency(note: string): number {
  const match = /^([A-Ga-g][#b]?)(-?\d+)$/.exec(note.trim());
  if (!match) return 440;
  const pitch = match[1][0].toUpperCase() + (match[1][1] ?? '');
  const semitone = NOTE_SEMITONES[pitch];
  if (semitone === undefined) return 440;
  const octave = Number(match[2]);
  const midi = (octave + 1) * 12 + semitone;
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * Lightweight Web Audio harmonic piano synth used ONLY during the first 1–2 seconds
 * while SplendidGrandPiano samples load silently in the background.
 */
function playFallbackSynthNote(
  context: AudioContext,
  destination: AudioNode,
  note: string,
  velocity: number,
  time: number,
  duration?: number,
): StopFn {
  const startTime = Math.max(context.currentTime, time);
  const freq = noteToFrequency(note);
  const vel = Math.max(0.05, Math.min(1, velocity));

  const env = context.createGain();
  const filter = context.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.setValueAtTime(Math.min(12000, freq * (4 + vel * 6)), startTime);
  filter.frequency.exponentialRampToValueAtTime(Math.max(400, freq * 2.2), startTime + 1.4);

  const osc1 = context.createOscillator();
  const osc2 = context.createOscillator();
  const osc2Gain = context.createGain();
  osc1.type = 'triangle';
  osc2.type = 'sine';
  osc1.frequency.setValueAtTime(freq, startTime);
  osc2.frequency.setValueAtTime(freq * 2, startTime);
  osc2Gain.gain.setValueAtTime(0.28, startTime);

  const peak = 0.24 * vel;
  env.gain.setValueAtTime(0.0001, startTime);
  env.gain.linearRampToValueAtTime(peak, startTime + 0.008);
  env.gain.exponentialRampToValueAtTime(Math.max(0.0005, peak * 0.18), startTime + 1.6);

  osc1.connect(filter);
  osc2.connect(osc2Gain);
  osc2Gain.connect(filter);
  filter.connect(env);
  env.connect(destination);

  osc1.start(startTime);
  osc2.start(startTime);

  let stopped = false;
  const stop: StopFn = (stopAt?: number) => {
    if (stopped) return;
    stopped = true;
    const t = Math.max(context.currentTime, stopAt ?? context.currentTime);
    try {
      env.gain.cancelScheduledValues(t);
      env.gain.setValueAtTime(Math.max(0.0001, env.gain.value), t);
      env.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
      osc1.stop(t + 0.2);
      osc2.stop(t + 0.2);
    } catch {
      // ignore already-stopped oscillator
    }
  };

  if (duration !== undefined) {
    stop(startTime + Math.max(0.05, duration));
  }
  return stop;
}

type SharedPianoEngine = {
  context: AudioContext;
  output: GainNode;
  piano: Smplr;
  samplesLoaded: boolean;
};

// Module-level singleton: keeps decoded SplendidGrandPiano AudioBuffers warm in RAM
// across route changes so returning to /piano is 0ms instant.
let sharedEngine: SharedPianoEngine | null = null;

function getSharedPianoEngine(): SharedPianoEngine | null {
  if (typeof window === 'undefined' || !window.AudioContext) return null;
  if (sharedEngine && sharedEngine.context.state !== 'closed') {
    return sharedEngine;
  }

  const context = new window.AudioContext({ latencyHint: 'interactive' });
  const output = context.createGain();
  output.connect(context.destination);

  // Load 1 core velocity layer (MF: vel 85..100 -> 5x fewer sample downloads & decodes)
  // with nearest-note fallback so SplendidGrandPiano loads in ~1-2s instead of 15-30s.
  const piano = SplendidGrandPiano(context, {
    destination: output,
    storage: new CacheStorage('cover-studio-piano-mf-v1'),
    notesToLoad: {
      velocityRange: [85, 100],
      fallback: 'nearest',
    },
  });

  const engine: SharedPianoEngine = {
    context,
    output,
    piano,
    samplesLoaded: false,
  };
  sharedEngine = engine;

  void piano.ready.then(() => {
    engine.samplesLoaded = true;
  }).catch(error => {
    console.warn('Could not load SplendidGrandPiano samples; continuing with built-in synth.', error);
  });

  return engine;
}

export function usePianoAudio() {
  const heldNotesRef = useRef(new Map<string, StopFn>());

  useEffect(() => {
    const engine = getSharedPianoEngine();
    const heldNotes = heldNotesRef.current;
    return () => {
      heldNotes.forEach(stop => stop());
      heldNotes.clear();
      engine?.piano.stop();
    };
  }, []);

  const startVoice = useCallback((note: string, velocity: number, time: number, duration?: number) => {
    const engine = getSharedPianoEngine();
    if (!engine) return;
    const clampedVel = Math.max(0.05, Math.min(1, velocity));
    // Map every velocity (0..1) into the loaded MF layer [85, 100] so no note is ever silent
    const mfVelocity = Math.round(85 + clampedVel * 15);
    const stop = engine.samplesLoaded
      ? engine.piano.start({
          note,
          velocity: mfVelocity,
          time,
          ...(duration === undefined ? {} : { duration }),
        })
      : playFallbackSynthNote(engine.context, engine.output, note, clampedVel, time, duration);
    if (duration === undefined) heldNotesRef.current.set(note, stop);
  }, []);

  const releaseNote = useCallback((note: string) => {
    const stop = heldNotesRef.current.get(note);
    heldNotesRef.current.delete(note);
    if (stop) stop(sharedEngine?.context.currentTime);
  }, []);

  const pressNote = useCallback(async (note: string, velocity = 0.5) => {
    const engine = getSharedPianoEngine();
    if (!engine) return;
    if (engine.context.state !== 'running') await engine.context.resume();
    startVoice(note, velocity, engine.context.currentTime);
  }, [startVoice]);

  const allOff = useCallback(() => {
    heldNotesRef.current.forEach(stop => stop());
    heldNotesRef.current.clear();
    sharedEngine?.piano.stop();
  }, []);

  const setGlobalVolume = useCallback((volume: number) => {
    const engine = getSharedPianoEngine();
    if (engine) engine.output.gain.value = Math.max(0, Math.min(1, volume));
  }, []);

  const preparePlayback = useCallback(async () => {
    const engine = getSharedPianoEngine();
    if (!engine) throw new Error('Không thể khởi tạo âm thanh Piano.');
    if (engine.context.state !== 'running') await engine.context.resume();
  }, []);

  const audioTime = useCallback(() => sharedEngine?.context.currentTime ?? 0, []);
  const scheduleNote = useCallback((note: string, duration: number, time: number, velocity: number) => {
    startVoice(note, velocity, time, duration);
  }, [startVoice]);

  // Immediately ready (0ms) thanks to instant Web Audio synth + background SplendidGrandPiano upgrade
  return { isReady: true, pressNote, releaseNote, allOff, setGlobalVolume, preparePlayback, audioTime, scheduleNote };
}
