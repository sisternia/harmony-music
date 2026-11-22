'use dom';

import React, { useCallback, useMemo, useState, useEffect, useRef } from 'react';
import '../../../../styles/theme.css';
import { useWindowDimensions } from 'react-native';
import Icon from '../components/PianoControlIcon';
import VolumeControl from '../../../../components/ui/VolumeControl';
import { usePianoAudio } from '../hooks/usePianoAudio';
import '../styles/piano.css';
import PianoStaff from '../components/PianoStaff';
import PianoImport from '../components/PianoImport';
import PianoSongPlayer from '../components/PianoSongPlayer';
import RisingWaterfall, { type RisingWaterfallHandle } from '../components/RisingWaterfall';
import { configurePianoApi, type PianoSong } from '../utils/transcription';
import '../styles/transcription.css';
import '../styles/mobile.css';
import { pianoPublicAsset } from '../utils/public-assets';

const allNotes: string[] = [];
allNotes.push("A0", "A#0", "B0");
[1, 2, 3, 4, 5, 6, 7].forEach(octave => {
  "CDEFGAB".split('').forEach(letter => {
    allNotes.push(`${letter}${octave}`);
    if (letter !== 'E' && letter !== 'B') {
      allNotes.push(`${letter}#${octave}`);
    }
  });
});
allNotes.push('C8');

const noteToLabel = (note: string) => {
  if (note.includes('#')) {
    const A = note[0];
    const octave = note[2];
    const aSharp = A + '♯';
    const B = A === 'G' ? 'A' : String.fromCharCode(A.charCodeAt(0) + 1);
    const bFlat = B + '♭';
    return (
      <>
        <span>{aSharp}<sub>{octave}</sub></span>
        <br />
        <span>{bFlat}<sub>{octave}</sub></span>
      </>
    );
  } else {
    const letter = note[0];
    const octave = note[1];
    return <>{letter}<sub>{octave}</sub></>;
  }
};

const KEYBOARD_MAP: Record<string, string> = {
  'KeyA': 'C4', 'KeyW': 'C#4', 'KeyS': 'D4', 'KeyE': 'D#4', 'KeyD': 'E4',
  'KeyF': 'F4', 'KeyT': 'F#4', 'KeyG': 'G4', 'KeyY': 'G#4', 'KeyH': 'A4',
  'KeyU': 'A#4', 'KeyJ': 'B4', 'KeyK': 'C5', 'KeyO': 'C#5', 'KeyL': 'D5',
  'KeyP': 'D#5', 'Semicolon': 'E5', 'Quote': 'F5',
};

const PianoKey = React.memo(function PianoKey({ note, pressed, hand, velocity, onPlay, onStop }: {
  note: string;
  pressed: boolean;
  hand?: 'L' | 'R';
  velocity: number;
  onPlay: (note: string) => void;
  onStop: (note: string) => void;
}) {
  const velocityLevel = Math.max(0, Math.min(1, velocity));
  return (
    <button
      data-note={note}
      data-hand={pressed && hand ? hand : undefined}
      className={pressed ? 'pressed' : ''}
      style={{ '--velocity-stop': `${velocityLevel * 100}%` } as React.CSSProperties}
      onPointerDown={event => {
        if (event.pointerType === 'touch') return;
        event.preventDefault();
        if (event.button === 0) onPlay(note);
      }}
      onPointerEnter={event => { if (event.pointerType !== 'touch' && event.buttons === 1) onPlay(note); }}
      onPointerLeave={event => { if (event.pointerType !== 'touch' && event.buttons === 1) onStop(note); }}
      onPointerUp={event => { if (event.pointerType !== 'touch' && event.button === 0) onStop(note); }}
      onPointerCancel={event => { if (event.pointerType !== 'touch') onStop(note); }}
      aria-label={`Nốt ${note}`}
      tabIndex={-1}
    >
      <span>{noteToLabel(note)}</span>
    </button>
  );
});

export default function PianoKeyboard({ apiUrl, nativeHost = false, initialSong = null, startInEdition = false, workspace = false, onNewTranscription, onProjectUpdate, onEditionChange }: {
  apiUrl?: string;
  nativeHost?: boolean;
  initialSong?: PianoSong | null;
  startInEdition?: boolean;
  workspace?: boolean;
  onNewTranscription?: () => void;
  onProjectUpdate?: (song: PianoSong, previousId?: string) => void;
  onEditionChange?: (edition: boolean) => void;
  dom?: import('expo/dom').DOMProps;
}) {
  const { width, height } = useWindowDimensions();
  const compact = width < 900;
  const [rangeStart, setRangeStart] = useState(39); // C4
  const [showAllKeys, setShowAllKeys] = useState(false);
  const rangeCount = width > height ? 25 : 13;
  useEffect(() => { configurePianoApi(apiUrl); }, [apiUrl]);
  const pianoAudio = usePianoAudio();
  const { isReady, pressNote, releaseNote, setGlobalVolume } = pianoAudio;
  const [importOpen, setImportOpen] = useState(false);
  const [song, setSong] = useState<PianoSong | null>(initialSong);
  const [midiTransportTarget, setMidiTransportTarget] = useState<HTMLDivElement | null>(null);
  const [playbackKeys, setPlaybackKeys] = useState<Set<string>>(new Set());
  const [playbackVelocities, setPlaybackVelocities] = useState<Record<string, number>>({});
  const [playbackHands, setPlaybackHands] = useState<Record<string, 'L' | 'R'>>({});
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const risingRef = useRef<RisingWaterfallHandle | null>(null);

  const [volume, setVolume] = useState(80);
  const volumeRef = useRef(80);
  const [velocityState, setVelocityState] = useState(0);
  const [noteVelocities, setNoteVelocities] = useState<Record<string, number>>({});
  const [showMidiScore, setShowMidiScore] = useState(false);
  const [staffState, setStaffState] = useState(1); // 0: hidden, 1: treble, 2: both
  const [labelStateIndex, setLabelStateIndex] = useState(1); // 0: '', 1: 'letters', 2: 'letters numbers'

  useEffect(() => {
    volumeRef.current = volume;
  }, [volume]);

  const handleActiveNotes = useCallback(
    (notes: Set<string>, velocities: Record<string, number>, hands: Record<string, 'L' | 'R'>) => {
      setPlaybackKeys(notes);
      setPlaybackVelocities(velocities);
      setPlaybackHands(hands);
    },
    [],
  );

  const getButtonStateClass = (val: number | boolean) => {
    if (typeof val === 'boolean') return val ? 'on' : '';
    if (val === 1) return 'on';
    if (val === 2) return 'onon';
    if (val === 3) return 'ononon';
    return '';
  };

  useEffect(() => {
    if (setGlobalVolume) {
      setGlobalVolume(volume / 100);
    }
  }, [volume, setGlobalVolume]);

  const keysByTouchesRef = useRef<Record<number, string>>({});

  const playNote = useCallback((note: string) => {
    if (pressedKeysRef.current.has(note)) return;
    const vel = volumeRef.current / 100;
    void pressNote(note, vel);
    pressedKeysRef.current.add(note);
    setPressedKeys(new Set(pressedKeysRef.current));
    setNoteVelocities(prev => (prev[note] === vel ? prev : { ...prev, [note]: vel }));
    risingRef.current?.pushPress(note);
  }, [pressNote]);

  const stopNote = useCallback((note: string) => {
    if (!pressedKeysRef.current.has(note)) return;
    releaseNote(note);
    pressedKeysRef.current.delete(note);
    setPressedKeys(new Set(pressedKeysRef.current));
    risingRef.current?.pushRelease(note);
  }, [releaseNote]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest('input, textarea, select, button'))) return;
      const note = KEYBOARD_MAP[event.code];
      if (note) {
        event.preventDefault();
        playNote(note);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      const note = KEYBOARD_MAP[event.code];
      if (note) stopNote(note);
    };
    const releasePressedNotes = () => {
      for (const note of pressedKeysRef.current) stopNote(note);
      keysByTouchesRef.current = {};
    };
    const handleVisibilityChange = () => { if (document.hidden) releasePressedNotes(); };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', releasePressedNotes);
    window.addEventListener('pagehide', releasePressedNotes);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', releasePressedNotes);
      window.removeEventListener('pagehide', releasePressedNotes);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [playNote, stopNote]);
  // Touch Handlers
  const getNoteFromTouch = (touch: React.Touch) => {
    const el = document.elementFromPoint(touch.clientX, touch.clientY);
    if (!el) return null;
    const key = el.closest('button[data-note]');
    return key ? key.getAttribute('data-note') : null;
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    // e.preventDefault(); // Might cause issues with passive listener, use css touch-action instead
    Array.from(e.changedTouches).forEach(touch => {
      const note = getNoteFromTouch(touch);
      if (note) {
        playNote(note);
        keysByTouchesRef.current[touch.identifier] = note;
      }
    });
  };

  const releaseTouch = (id: number) => {
    const note = keysByTouchesRef.current[id];
    delete keysByTouchesRef.current[id];
    if (note && !Object.values(keysByTouchesRef.current).includes(note)) stopNote(note);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    Array.from(e.targetTouches).forEach(touch => {
      const note = getNoteFromTouch(touch);
      const id = touch.identifier;
      const currentNote = keysByTouchesRef.current[id];

      if (note && note !== currentNote) {
        if (currentNote) releaseTouch(id);
        playNote(note);
        keysByTouchesRef.current[id] = note;
      } else if (!note && currentNote) {
        releaseTouch(id);
      }
    });
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    Array.from(e.changedTouches).forEach(touch => {
      const id = touch.identifier;
      releaseTouch(id);
    });
  };

  const keysData = useMemo(() => {
    const keys = allNotes.map(note => ({ note, isBlack: note.includes('#') }));
    if (!compact || showAllKeys || song) return keys;
    const start = Math.min(rangeStart, allNotes.length - rangeCount);
    return keys.slice(start, start + rangeCount);
  }, [compact, showAllKeys, song, rangeStart, rangeCount]);

  // Mảng nốt nhạc cho khuông nhạc (Staff notes)
  return (
    <div
      className={`piano-page ${workspace ? 'studio-piano' : ''} ${song ? 'has-song' : ''} ${song?.midiRoll ? 'has-midi-song' : ''} ${nativeHost ? 'native-host' : ''} ${compact && !showAllKeys ? 'zoomed-keys' : ''}`}
      style={{
        '--piano-smoke': `url("${pianoPublicAsset('images/smoke_blur.webp')}")`,
        '--piano-smoke-layer': `url("${pianoPublicAsset('images/smoke_layer1_blur.webp')}")`,
      } as React.CSSProperties}
    >
      {nativeHost && <style>{`@font-face { font-family: Bravura; src: url(${JSON.stringify(pianoPublicAsset('fonts/Bravura.woff2'))}); }`}</style>}
      <PianoImport
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImported={value => {
          pianoAudio.allOff();
          setPlaybackKeys(new Set());
          setPlaybackHands({});
          setSong({ ...value, notes: [...value.notes].sort((a, b) => a.start - b.start) });
        }}
      />

      <div className="content">
        <div className="chat-sidebar"></div>
        <div className="main-content">
          <div className="info-row">
            <RisingWaterfall enabled={!song && velocityState > 0} controllerRef={risingRef} />
            {song ? <PianoSongPlayer
              key={song.id}
              song={song}
              audio={pianoAudio}
              midiTransportTarget={midiTransportTarget}
              showMidiScore={showMidiScore}
              startInEdition={startInEdition}
              onEditionChange={onEditionChange}
              onNewTranscription={onNewTranscription || (() => setImportOpen(true))}
              onSongChange={updated => { onProjectUpdate?.(updated, song.id); setSong(updated); }}
              onActiveNotes={handleActiveNotes}
              onClose={() => { setSong(null); setPlaybackKeys(new Set()); setPlaybackHands({}); }}
            /> : <div className="staff-paper" hidden={staffState === 0}>
              <PianoStaff pressedKeys={pressedKeys} showBass={staffState === 2} />
              <div className="chord-info">
                <b id="chord-name">
                  {!isReady ? 'Loading audio...' : ''}
                </b>
              </div>
            </div>}
          </div>

          {compact && !song && <div className="piano-range-controls" aria-label="Quãng bàn phím">
            <button disabled={showAllKeys || rangeStart === 0} onClick={() => setRangeStart(value => Math.max(0, value - 12))} aria-label="Quãng thấp hơn">‹</button>
            <span>{keysData[0].note} – {keysData[keysData.length - 1].note}</span>
            <button disabled={showAllKeys || rangeStart >= allNotes.length - rangeCount} onClick={() => setRangeStart(value => Math.min(allNotes.length - rangeCount, value + 12))} aria-label="Quãng cao hơn">›</button>
            <button aria-pressed={showAllKeys} onClick={() => setShowAllKeys(value => !value)}>{showAllKeys ? 'Phóng phím' : '88 phím'}</button>
          </div>}
          <div
            className={`keyboard ${['', 'letters', 'letters numbers'][labelStateIndex]} ${velocityState > 0 ? `velocity velocity-${velocityState}` : ''}`}
            style={{ touchAction: 'none' }}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onTouchCancel={handleTouchEnd}
          >
            <div className="top-keys">
              {keysData.map(k => (
                <PianoKey
                  key={k.note}
                  note={k.note}
                  pressed={pressedKeys.has(k.note) || playbackKeys.has(k.note)}
                  hand={song?.midiRoll && !pressedKeys.has(k.note) ? playbackHands[k.note] : undefined}
                  velocity={playbackVelocities[k.note] ?? noteVelocities[k.note] ?? 0}
                  onPlay={playNote}
                  onStop={stopNote}
                />
              ))}
            </div>
            <div className="bottom-keys">
              {keysData.filter(k => !k.isBlack).map(k => (
                <PianoKey
                  key={k.note}
                  note={k.note}
                  pressed={pressedKeys.has(k.note) || playbackKeys.has(k.note)}
                  hand={song?.midiRoll && !pressedKeys.has(k.note) ? playbackHands[k.note] : undefined}
                  velocity={playbackVelocities[k.note] ?? noteVelocities[k.note] ?? 0}
                  onPlay={playNote}
                  onStop={stopNote}
                />
              ))}
            </div>
          </div>

          <div className="bar tool-bar">
            <div id="instrument-app" className="tool-bar">
              <div>
                <button id="instrument-label" title="Chọn file hoặc link YouTube" aria-label="Nhập bản nhạc để chuyển thành piano" onClick={() => setImportOpen(true)}>
                  <span>
                    <Icon icon="mdi-piano" />
                  </span>
                </button>
                <VolumeControl
                  value={volume / 100}
                  onChange={next => setVolume(Math.round(next * 100))}
                  placement="top"
                  label="Điều chỉnh âm lượng Piano"
                  sliderLabel="Âm lượng Piano"
                />
                {song?.midiRoll && (
                  <button
                    id="toggle-midi-score"
                    type="button"
                    title={showMidiScore ? 'Ẩn phổ nhạc (hiện nốt rơi)' : 'Bật phổ nhạc'}
                    aria-label="Bật/tắt phổ nhạc"
                    aria-pressed={showMidiScore}
                    onClick={() => setShowMidiScore(current => !current)}
                    className={getButtonStateClass(showMidiScore)}
                  >
                    <Icon icon="mdi-music-clef-treble" />
                  </button>
                )}
              </div>
            </div>

            {!song?.midiRoll && <div>
              <div>
                <button id="toggle-score" title="show staff" aria-label="toggle show staff" onClick={() => setStaffState(prev => (prev + 1) % 3)} className={getButtonStateClass(staffState)}>
                  <Icon icon="mdi-music-clef-treble" />
                </button>
                <button id="toggle-qwerty" hidden title="show keyboard mapping" aria-label="toggle show keyboard mapping">
                  <Icon icon="mdi-keyboard-outline" />
                </button>
                <button id="toggle-labels" title="show note labels" aria-label="toggle show note labels" onClick={() => setLabelStateIndex(prev => (prev + 1) % 3)} className={getButtonStateClass(labelStateIndex)}>
                  <Icon icon="mdi-label-outline" />
                </button>
                <button
                  id="toggle-velocity"
                  title={['Velocity off', 'Velocity · full key', 'Velocity · notched key'][velocityState]}
                  aria-label="Show velocity level"
                  aria-pressed={velocityState > 0}
                  onClick={() => setVelocityState(current => (current + 1) % 3)}
                  className={getButtonStateClass(velocityState)}
                >
                  <Icon icon="mdi-glass-stange" />
                </button>
              </div>
            </div>}

            {song?.midiRoll ? <div className="piano-midi-transport-slot" ref={setMidiTransportTarget} /> : <div id="recorder-app">
              <div>
                <button>
                  <Icon icon="mdi-record" />
                </button>
                <button>
                  <Icon icon="mdi-play" />
                </button>
              </div>
            </div>}
          </div>
        </div>
      </div>
    </div>
  );
}
