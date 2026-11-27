// カバーの音 (と、薄く原曲の MIDI) をピアノロールで見せる。再生位置に合わせて流れる。
// クリックでその位置へ、ドラッグでループ区間、ホイールで横に動かす (Ctrl / ⌘ + ホイールで拡大縮小)。
import { useCallback, useEffect, useRef, useState } from "react";
import type { CoverView, SourceView } from "../../features/studio/api";
import { formatTime } from "../../features/studio/utils/params";
import type { Player } from "../../features/studio/utils/player";

type Props = {
  player: Player;
  cover: CoverView | null;
  source: SourceView | null;
  showSource: boolean;
  continueAt: number | null;
  height?: number;
};

const CHORD_H = 18;
const PEDAL_H = 10;
const MIN_PITCH = 21;
const MAX_PITCH = 108;
const BLACK_PITCH_CLASSES = [1, 3, 6, 8, 10];
export const CANVAS_FONT = `system-ui, "Segoe UI", "Hiragino Sans", "Yu Gothic UI", "Noto Sans JP", sans-serif`;
const GROUP_COLORS = ["--src-melody", "--src-bass", "--src-keys", "--src-guitar", "--src-other"];

function pitchRange(cover: CoverView | null, source: SourceView | null, showSource: boolean, pitchSpan: number, pitchOffset: number) {
  let lo = MAX_PITCH;
  let hi = MIN_PITCH;
  const scan = (notes: number[][]) => {
    for (const note of notes) {
      const pitch = note[2];
      if (!Number.isFinite(pitch) || pitch < MIN_PITCH || pitch > MAX_PITCH) continue;
      lo = Math.min(lo, pitch);
      hi = Math.max(hi, pitch);
    }
  };
  if (cover) scan(cover.notes);
  if (source && (showSource || !cover)) scan(source.notes);
  if (hi < lo) { lo = 48; hi = 84; }
  lo = Math.max(MIN_PITCH, lo - 2);
  hi = Math.min(MAX_PITCH, hi + 2);
  if (hi - lo < 24) {
    const center = (hi + lo) / 2;
    lo = Math.floor(center - 12);
    hi = Math.ceil(center + 12);
  }
  const span = Math.max(1, Math.min(88, pitchSpan || (hi - lo + 1)));
  const center = Math.max(MIN_PITCH + (span - 1) / 2, Math.min(MAX_PITCH - (span - 1) / 2, (hi + lo) / 2 + pitchOffset));
  lo = Math.max(MIN_PITCH, Math.floor(center - (span - 1) / 2));
  hi = Math.min(MAX_PITCH, lo + span - 1);
  lo = Math.max(MIN_PITCH, hi - span + 1);
  return { lo, hi };
}

function noteName(pitch: number) {
  return ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][pitch % 12] + (Math.floor(pitch / 12) - 1);
}

function clampViewStart(start: number, seconds: number, duration: number) {
  return Math.max(0, Math.min(start, Math.max(0, duration - seconds)));
}

export function PianoRoll({ player, cover, source, showSource, continueAt, height = 390 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const keyboardRef = useRef<HTMLCanvasElement>(null);
  const view = useRef({ start: 0, seconds: 16 });
  const drag = useRef<{ x: number; t: number; moved: boolean } | null>(null);
  const keyboardGeometry = useRef({ lo: MIN_PITCH, hi: MAX_PITCH, top: CHORD_H, rowH: 1 });
  const heldPointers = useRef(new Map<number, number>());
  const manualPitches = useRef(new Map<number, number>());
  const hoverPitch = useRef<number | null>(null);
  const focusedPitch = useRef(60);
  const [zoom, setZoom] = useState(16);
  const [pitchSpan, setPitchSpan] = useState(88);
  const [pitchOffset, setPitchOffset] = useState(0);
  const [fullHeight, setFullHeight] = useState(false);
  const requestedZoom = useRef(zoom);
  const requestedPitchSpan = useRef(pitchSpan);
  const maxZoom = Math.max(4, Math.min(120, Math.ceil(player.duration || 120)));
  const visiblePitchRange = pitchRange(cover, source, showSource, pitchSpan, pitchOffset);
  const dataRef = useRef({ cover, source, showSource, continueAt });

  useEffect(() => {
    dataRef.current = { cover, source, showSource, continueAt };
  }, [cover, source, showSource, continueAt]);

  const timeAt = useCallback((clientX: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(player.duration, view.current.start + ((clientX - rect.left) / rect.width) * view.current.seconds));
  }, [player]);

  useEffect(() => {
    requestedZoom.current = zoom;
  }, [zoom]);

  useEffect(() => {
    requestedPitchSpan.current = pitchSpan;
  }, [pitchSpan]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const keyboard = keyboardRef.current!;
    const keyCtx = keyboard.getContext("2d")!;
    let raf = 0;
    let colors: Record<string, string> = {};
    const readColors = () => {
      const style = getComputedStyle(canvas);
      const names = [
        "--roll-bg",
        "--roll-black-key",
        "--roll-grid",
        "--roll-bar",
        "--roll-text",
        "--roll-key-light",
        "--roll-key-dark",
        "--roll-key-edge",
        "--accent",
        "--accent-2",
        "--accent-soft",
        "--playhead",
        "--loop",
        "--continue",
        ...GROUP_COLORS,
      ];
      colors = Object.fromEntries(names.map((n) => [n, style.getPropertyValue(n).trim()]));
    };
    readColors();
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", readColors);

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { cover, source, showSource, continueAt } = dataRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const kw = keyboard.clientWidth;
      if (keyboard.width !== Math.round(kw * dpr) || keyboard.height !== Math.round(h * dpr)) {
        keyboard.width = Math.round(kw * dpr);
        keyboard.height = Math.round(h * dpr);
      }
      keyCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const pos = player.position();
      const v = view.current;
      const duration = player.duration;
      v.seconds = duration > 0 ? Math.min(requestedZoom.current, duration) : requestedZoom.current;
      // 再生中は再生位置が右端の 85% を越えたら次のページへ送る
      if (player.playing && (pos > v.start + v.seconds * 0.85 || pos < v.start)) {
        v.start = pos - v.seconds * 0.1;
      }
      v.start = clampViewStart(v.start, v.seconds, duration);
      const t0 = v.start;
      const t1 = duration > 0 ? Math.min(v.start + v.seconds, duration) : v.start + v.seconds;
      const xOf = (t: number) => ((t - t0) / v.seconds) * w;

      const { lo, hi } = pitchRange(cover, source, showSource, requestedPitchSpan.current, pitchOffset);
      const top = CHORD_H;
      const bottom = h - PEDAL_H;
      const rowH = (bottom - top) / (hi - lo + 1);
      const yOf = (pitch: number) => top + (hi - pitch) * rowH;
      keyboardGeometry.current = { lo, hi, top, rowH };

      // The keyboard shares this exact pitch range and row geometry with the note grid.
      keyCtx.clearRect(0, 0, kw, h);
      keyCtx.fillStyle = colors["--roll-bg"];
      keyCtx.fillRect(0, 0, kw, h);
      const active = new Set<number>();
      const collectActive = (notes: number[][], endIndex: number, useSustainEnd: boolean) => {
        for (const n of notes) {
          if (n[0] > pos) break;
          if (pos < (useSustainEnd ? n[endIndex] : n[0] + n[1])) active.add(n[2]);
        }
      };
      if (cover) collectActive(cover.notes, 4, true);
      else if (source) collectActive(source.notes, 1, false);
      for (const pitch of manualPitches.current.values()) active.add(pitch);
      for (let p = lo; p <= hi; p++) {
        if (BLACK_PITCH_CLASSES.includes(p % 12)) continue;
        const y = yOf(p);
        keyCtx.fillStyle = active.has(p) || hoverPitch.current === p ? colors["--accent-soft"] : colors["--roll-key-light"];
        keyCtx.fillRect(0, y, kw, rowH);
        keyCtx.strokeStyle = colors["--roll-grid"];
        keyCtx.strokeRect(0.5, y + 0.5, kw - 1, Math.max(0, rowH - 1));
        if (p % 12 === 0 && rowH >= 3) {
          keyCtx.fillStyle = active.has(p) || hoverPitch.current === p ? colors["--accent"] : colors["--roll-text"];
          keyCtx.font = `600 ${rowH >= 8 ? 10 : 9}px ${CANVAS_FONT}`;
          keyCtx.textBaseline = "middle";
          keyCtx.fillText(noteName(p), 5, y + rowH / 2);
        }
      }
      for (let p = lo; p <= hi; p++) {
        if (!BLACK_PITCH_CLASSES.includes(p % 12)) continue;
        const y = yOf(p);
        const keyW = kw * 0.68;
        keyCtx.fillStyle = active.has(p) || hoverPitch.current === p ? colors["--accent"] : colors["--roll-key-dark"];
        keyCtx.beginPath();
        keyCtx.roundRect(0, y, keyW, Math.max(1, rowH), [0, 2, 2, 0]);
        keyCtx.fill();
        keyCtx.strokeStyle = colors["--roll-key-edge"];
        keyCtx.stroke();
      }

      ctx.fillStyle = colors["--roll-bg"];
      ctx.fillRect(0, 0, w, h);
      // 黒鍵の段
      ctx.fillStyle = colors["--roll-black-key"];
      for (let p = lo; p <= hi; p++) {
        if ([1, 3, 6, 8, 10].includes(p % 12)) ctx.fillRect(0, yOf(p), w, rowH);
      }
      // 拍と小節
      if (source) {
        ctx.fillStyle = colors["--roll-grid"];
        for (const b of source.beats) if (b >= t0 && b <= t1) ctx.fillRect(Math.round(xOf(b)), top, 1, bottom - top);
        ctx.fillStyle = colors["--roll-bar"];
        for (const b of source.downbeats)
          if (b >= t0 && b <= t1) ctx.fillRect(Math.round(xOf(b)), top, 1, bottom - top);
      } else {
        ctx.fillStyle = colors["--roll-grid"];
        for (let s = Math.ceil(t0); s <= t1; s++) ctx.fillRect(Math.round(xOf(s)), top, 1, bottom - top);
      }
      // ループ区間
      if (player.loop) {
        const [a, b] = player.loop;
        ctx.globalAlpha = player.loopOn ? 1 : 0.4;
        ctx.fillStyle = colors["--loop"];
        ctx.fillRect(xOf(a), 0, xOf(b) - xOf(a), h);
        ctx.globalAlpha = 1;
      }
      // 原曲の MIDI (薄く)
      if (source && (showSource || !cover)) {
        ctx.globalAlpha = cover ? 0.45 : 0.85;
        for (const n of source.notes) {
          if (n[0] > t1) break;
          if (n[0] + n[1] < t0) continue;
          ctx.fillStyle = colors[GROUP_COLORS[n[4]] ?? "--src-other"];
          ctx.fillRect(xOf(n[0]), yOf(n[2]) + 1, Math.max(2, (n[1] / v.seconds) * w), Math.max(1, rowH - 2));
        }
        ctx.globalAlpha = 1;
      }
      // カバー: 鳴り終わり (ペダル) を薄く、押している長さを濃く
      if (cover) {
        for (const n of cover.notes) {
          if (n[0] > t1) break;
          if (n[4] < t0) continue;
          const x = xOf(n[0]);
          const y = yOf(n[2]);
          ctx.globalAlpha = 0.18;
          ctx.fillStyle = colors["--accent"];
          ctx.fillRect(x, y + 1, Math.max(2, ((n[4] - n[0]) / v.seconds) * w), Math.max(1, rowH - 2));
          ctx.globalAlpha = 0.35 + 0.65 * (n[3] / 127);
          const grad = ctx.createLinearGradient(x, 0, x + 40, 0);
          grad.addColorStop(0, colors["--accent"]);
          grad.addColorStop(1, colors["--accent-2"]);
          ctx.fillStyle = grad;
          ctx.fillRect(x, y, Math.max(2.5, (n[1] / v.seconds) * w), Math.max(2, rowH));
        }
        ctx.globalAlpha = 1;
        // ペダル
        ctx.fillStyle = colors["--accent"];
        ctx.globalAlpha = 0.5;
        for (const [a, b] of cover.pedals) {
          if (a > t1 || b < t0) continue;
          ctx.fillRect(xOf(a), h - PEDAL_H + 3, Math.max(1, xOf(b) - xOf(a) - 1), PEDAL_H - 5);
        }
        ctx.globalAlpha = 1;
      }
      // コード名
      ctx.fillStyle = colors["--roll-bg"];
      ctx.fillRect(0, 0, w, CHORD_H);
      if (source) {
        ctx.font = `11px ${CANVAS_FONT}`;
        ctx.fillStyle = colors["--roll-text"];
        ctx.textBaseline = "middle";
        let lastX = -Infinity;
        for (const [t, name] of source.chords) {
          if (t > t1) break;
          const x = xOf(t);
          if (x < -40 || x - lastX < 28) continue;
          ctx.fillText(name, x + 2, CHORD_H / 2);
          lastX = x;
        }
      }
      // 続きを作り直す位置
      if (continueAt != null && continueAt >= t0 && continueAt <= t1) {
        ctx.fillStyle = colors["--continue"];
        ctx.fillRect(xOf(continueAt) - 1, 0, 2, h);
      }
      // 再生位置
      const px = xOf(pos);
      if (px >= 0 && px <= w) {
        ctx.fillStyle = colors["--playhead"];
        ctx.fillRect(px - 1, 0, 2, h);
      }
      // 時刻の目盛り
      ctx.font = `10px ${CANVAS_FONT}`;
      ctx.fillStyle = colors["--roll-text"];
      ctx.textBaseline = "bottom";
      const every = v.seconds > 40 ? 10 : v.seconds > 16 ? 5 : 2;
      for (let s = Math.ceil(t0 / every) * every; s <= t1; s += every) {
        ctx.fillText(formatTime(s), xOf(s) + 3, bottom - 2);
      }

    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      media.removeEventListener("change", readColors);
    };
  }, [player, pitchOffset]);

  const pitchAtPointer = useCallback((clientY: number) => {
    const rect = keyboardRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const { lo, hi, top, rowH } = keyboardGeometry.current;
    const row = Math.floor((clientY - rect.top - top) / rowH);
    if (row < 0 || row > hi - lo) return null;
    return hi - row;
  }, []);

  const updateKeyboardLabel = useCallback((pitch: number) => {
    const keyboard = keyboardRef.current;
    if (!keyboard) return;
    const label = `${noteName(pitch)} · MIDI ${pitch}`;
    keyboard.title = label;
    keyboard.setAttribute("aria-label", `${label}. Playable piano key. Use up and down arrows to choose a note, then Space or Enter to play.`);
  }, []);

  const pressPointerNote = useCallback((pointerId: number, pitch: number) => {
    if (player.pianoState === "loading") return;
    if (heldPointers.current.get(pointerId) === pitch) return;
    if (heldPointers.current.has(pointerId)) {
      const previousPitch = heldPointers.current.get(pointerId)!;
      manualPitches.current.delete(pointerId);
      heldPointers.current.delete(pointerId);
      if (![...heldPointers.current.values()].includes(previousPitch)) player.stopManualNote(-previousPitch);
    }
    const alreadyHeld = [...heldPointers.current.values()].includes(pitch);
    if (alreadyHeld || player.startManualNote(-pitch, pitch)) {
      heldPointers.current.set(pointerId, pitch);
      manualPitches.current.set(pointerId, pitch);
    }
  }, [player]);

  const releasePointerNote = useCallback((pointerId: number) => {
    const pitch = heldPointers.current.get(pointerId);
    if (pitch == null) return;
    heldPointers.current.delete(pointerId);
    manualPitches.current.delete(pointerId);
    if (![...heldPointers.current.values()].includes(pitch)) player.stopManualNote(-pitch);
  }, [player]);

  useEffect(() => {
    const releaseAll = () => {
      for (const pitch of new Set(heldPointers.current.values())) player.stopManualNote(-pitch);
      heldPointers.current.clear();
      manualPitches.current.clear();
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") releaseAll();
    };
    window.addEventListener("blur", releaseAll);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("blur", releaseAll);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      releaseAll();
    };
  }, [player]);

  // Ctrl + ホイールでページごと拡大しないよう、passive でない listener で受ける
  useEffect(() => {
    const canvas = canvasRef.current!;
    const rollContent = canvas.parentElement!;
    const onWheel = (e: WheelEvent) => {
      const v = view.current;
      if (e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setPitchOffset((offset) => Math.max(-44, Math.min(44, offset + Math.sign(e.deltaY) * Math.max(1, Math.round(Math.abs(e.deltaY) / 80)) * 3)));
      } else if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const t = timeAt(e.clientX);
        const next = Math.min(Math.max(4, player.duration || 120), Math.max(4, v.seconds * Math.exp(e.deltaY * 0.002)));
        v.start = clampViewStart(t - ((t - v.start) * next) / v.seconds, next, player.duration);
        v.seconds = next;
        setZoom(next);
      } else {
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        if (delta) e.preventDefault();
        v.start = clampViewStart(v.start + (delta / 800) * v.seconds, v.seconds, player.duration);
      }
    };
    rollContent.addEventListener("wheel", onWheel, { passive: false });
    return () => rollContent.removeEventListener("wheel", onWheel);
  }, [player, player.duration, timeAt]);

  return (
    <div className={`roll${fullHeight ? " roll-expanded" : ""}`}>
      <div className="roll-content">
        <canvas
          ref={keyboardRef}
          className="piano-keyboard"
          style={{ height: fullHeight ? "min(68vh, 720px)" : `var(--roll-h, ${height}px)` }}
          role="application"
          tabIndex={0}
          aria-disabled={player.pianoState === "loading"}
          aria-label={`Playable 88-key piano, MIDI pitches ${MIN_PITCH} to ${MAX_PITCH}; visible ${noteName(visiblePitchRange.hi)} to ${noteName(visiblePitchRange.lo)}.`}
          title={player.pianoState === "loading" ? "Piano sound is loading" : "88-key piano · A0–C8"}
          onPointerDown={(event) => {
            if (player.pianoState === "loading") return;
            event.preventDefault();
            event.currentTarget.focus({ preventScroll: true });
            event.currentTarget.setPointerCapture(event.pointerId);
            const pitch = pitchAtPointer(event.clientY);
            if (pitch != null) pressPointerNote(event.pointerId, pitch);
          }}
          onPointerMove={(event) => {
            const pitch = pitchAtPointer(event.clientY);
            if (pitch == null) return;
            hoverPitch.current = pitch;
            focusedPitch.current = pitch;
            updateKeyboardLabel(pitch);
            if (heldPointers.current.has(event.pointerId)) pressPointerNote(event.pointerId, pitch);
          }}
          onPointerUp={(event) => releasePointerNote(event.pointerId)}
          onPointerCancel={(event) => releasePointerNote(event.pointerId)}
          onLostPointerCapture={(event) => releasePointerNote(event.pointerId)}
          onPointerLeave={(event) => {
            hoverPitch.current = null;
            if (!heldPointers.current.has(event.pointerId)) {
              event.currentTarget.title = player.pianoState === "loading" ? "Piano sound is loading" : "88-key piano · A0–C8";
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowUp" || event.key === "ArrowDown") {
              event.preventDefault();
              focusedPitch.current = Math.max(MIN_PITCH, Math.min(MAX_PITCH, focusedPitch.current + (event.key === "ArrowUp" ? 1 : -1)));
              hoverPitch.current = focusedPitch.current;
              updateKeyboardLabel(focusedPitch.current);
            } else if (event.key === " " || event.key === "Enter") {
              event.preventDefault();
              pressPointerNote(-1000, focusedPitch.current);
            }
          }}
          onKeyUp={(event) => {
            if (event.key === " " || event.key === "Enter") releasePointerNote(-1000);
          }}
          onBlur={() => releasePointerNote(-1000)}
        />
        <canvas
        ref={canvasRef}
        // 高さは CSS 変数 --roll-h で上書きできる (携帯では少し低くする)
        style={{ height: fullHeight ? "min(68vh, 720px)" : `var(--roll-h, ${height}px)` }}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, t: timeAt(e.clientX), moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          if (Math.abs(e.clientX - d.x) > 4) d.moved = true;
          if (d.moved) {
            const t = timeAt(e.clientX);
            player.setLoop([Math.max(0, Math.min(d.t, t)), Math.max(d.t, t)]);
          }
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          if (!d.moved) {
            player.seek(timeAt(e.clientX));
          } else if (player.loop && player.loop[1] - player.loop[0] < 0.5) {
            player.setLoop(null);
          } else if (player.loop) {
            player.seek(player.loop[0]);
          }
        }}
        // 指で頁を送り始めると pointercancel になる。押しっぱなしの状態を捨て、
        // 途中まで引いてしまった短いループも消す
        onPointerCancel={() => {
          drag.current = null;
          if (player.loop && player.loop[1] - player.loop[0] < 0.5) player.setLoop(null);
        }}
        />
      </div>
      <div className="roll-zoom roll-zoom-bar">
        <button
          type="button"
          className="roll-fit"
          onClick={() => setZoom((z) => Math.max(4, Math.round(z * 0.72)))}
          title="Phóng to"
        >+</button>
        <span className="mono">{Math.round(Math.min(zoom, player.duration || zoom))}s</span>
        <button
          type="button"
          className="roll-fit"
          onClick={() => setZoom((z) => Math.min(maxZoom, Math.round(z * 1.35)))}
          title="Thu nhỏ"
        >−</button>
        <button
          type="button"
          className={pitchSpan === 88 ? "roll-fit on" : "roll-fit"}
          aria-pressed={pitchSpan === 88}
          onClick={() => {
            setPitchSpan((s) => (s === 88 ? 36 : 88));
            setPitchOffset(0);
          }}
          title="Chuyển giữa toàn bộ 88 phím (A0–C8) và phóng to vùng nốt"
        >88 phím</button>
        <button
          type="button"
          className={fullHeight ? "roll-fit on" : "roll-fit"}
          onClick={() => setFullHeight((value) => !value)}
          aria-pressed={fullHeight}
          title="Mở rộng chiều cao Piano Roll"
        >{fullHeight ? "Thu gọn" : "Mở rộng"}</button>
      </div>
    </div>
  );
}
