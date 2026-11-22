import { useCallback, useEffect, useRef, useState } from 'react';
import { CacheStorage, SplendidGrandPiano, type Smplr, type StopFn } from 'smplr';

export function usePianoAudio() {
  const contextRef = useRef<AudioContext | null>(null);
  const outputRef = useRef<GainNode | null>(null);
  const pianoRef = useRef<Smplr | null>(null);
  const heldNotesRef = useRef(new Map<string, StopFn>());
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const AudioContextClass = window.AudioContext;
    if (!AudioContextClass) return;
    const context = new AudioContextClass({ latencyHint: 'interactive' });
    const output = context.createGain();
    output.connect(context.destination);
    const piano = SplendidGrandPiano(context, {
      destination: output,
      storage: new CacheStorage('cover-studio-piano'),
    });
    let mounted = true;
    const heldNotes = heldNotesRef.current;
    contextRef.current = context;
    outputRef.current = output;
    pianoRef.current = piano;

    void piano.ready.then(() => {
      if (mounted) setIsReady(true);
    }).catch(error => {
      console.error('Could not load the piano samples.', error);
    });

    return () => {
      mounted = false;
      heldNotes.clear();
      piano.dispose();
      output.disconnect();
      void context.close();
      contextRef.current = null;
      outputRef.current = null;
      pianoRef.current = null;
    };
  }, []);

  const startVoice = useCallback((note: string, velocity: number, time: number, duration?: number) => {
    const piano = pianoRef.current;
    if (!piano) return;
    const stop = piano.start({
      note,
      velocity: Math.round(Math.max(0, Math.min(1, velocity)) * 127),
      time,
      ...(duration === undefined ? {} : { duration }),
    });
    if (duration === undefined) heldNotesRef.current.set(note, stop);
  }, []);

  const releaseNote = useCallback((note: string) => {
    const stop = heldNotesRef.current.get(note);
    heldNotesRef.current.delete(note);
    if (stop) stop(contextRef.current?.currentTime);
  }, []);

  const pressNote = useCallback(async (note: string, velocity = 0.5) => {
    const context = contextRef.current;
    if (!context || !pianoRef.current || !isReady) return;
    if (context.state !== 'running') await context.resume();
    startVoice(note, velocity, context.currentTime);
  }, [isReady, startVoice]);

  const allOff = useCallback(() => {
    heldNotesRef.current.clear();
    pianoRef.current?.stop();
  }, []);

  const setGlobalVolume = useCallback((volume: number) => {
    if (outputRef.current) outputRef.current.gain.value = Math.max(0, Math.min(1, volume));
  }, []);

  const preparePlayback = useCallback(async () => {
    const context = contextRef.current;
    const piano = pianoRef.current;
    if (!context || !piano) throw new Error('Không thể khởi tạo âm thanh Piano.');
    await piano.ready;
    if (context.state !== 'running') await context.resume();
  }, []);

  const audioTime = useCallback(() => contextRef.current?.currentTime ?? 0, []);
  const scheduleNote = useCallback((note: string, duration: number, time: number, velocity: number) => {
    startVoice(note, velocity, time, duration);
  }, [startVoice]);

  return { isReady, pressNote, releaseNote, allOff, setGlobalVolume, preparePlayback, audioTime, scheduleNote };
}
