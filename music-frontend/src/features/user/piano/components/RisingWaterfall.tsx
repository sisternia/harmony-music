import { useEffect, useImperativeHandle, useRef } from 'react';
import { colors } from '@/theme/colors';

export type RisingWaterfallHandle = {
  pushPress: (note: string) => void;
  pushRelease: (note: string) => void;
  clear: () => void;
};

type RisingBar = {
  note: string;
  startMs: number;
  endMs: number | null;
};

export default function RisingWaterfall({
  enabled,
  controllerRef,
}: {
  enabled: boolean;
  controllerRef: React.RefObject<RisingWaterfallHandle | null>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const barsRef = useRef<RisingBar[]>([]);
  const wakeRef = useRef<(() => void) | null>(null);

  useImperativeHandle(
    controllerRef,
    () => ({
      pushPress(note: string) {
        if (!enabled) return;
        barsRef.current.push({ note, startMs: performance.now(), endMs: null });
        wakeRef.current?.();
      },
      pushRelease(note: string) {
        const list = barsRef.current;
        for (let i = list.length - 1; i >= 0; i--) {
          if (list[i].note === note && list[i].endMs === null) {
            list[i].endMs = performance.now();
            break;
          }
        }
        wakeRef.current?.();
      },
      clear() {
        barsRef.current.length = 0;
        wakeRef.current?.();
      },
    }),
    [enabled],
  );

  useEffect(() => {
    if (!enabled) {
      barsRef.current.length = 0;
      return;
    }
    const element = canvasRef.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;

    let frame = 0;
    let running = false;
    let width = 0;
    let height = 0;
    let ratio = 1;
    const positions = new Map<string, { x: number; w: number; isBlack: boolean }>();

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
      positions.clear();
      const page = element.closest('.piano-page');
      const keys = page?.querySelectorAll<HTMLButtonElement>('.keyboard .top-keys button[data-note]');
      keys?.forEach(key => {
        const note = key.dataset.note || '';
        if (!note) return;
        const box = key.getBoundingClientRect();
        positions.set(note, {
          x: box.left + box.width * 0.5 - rect.left,
          w: box.width,
          isBlack: note.includes('#'),
        });
      });
    };

    const draw = () => {
      running = false;
      if (!width || !height) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const bars = barsRef.current;
      if (!bars.length) return;

      const now = performance.now();
      const speedPxPerMs = Math.max(0.18, height / 3600); // ~3.6s to cross viewport
      let writeIdx = 0;

      for (let i = 0; i < bars.length; i++) {
        const bar = bars[i];
        const pos = positions.get(bar.note);
        const top = height - (now - bar.startMs) * speedPxPerMs;
        const bottom = bar.endMs !== null ? height - (now - bar.endMs) * speedPxPerMs : height;

        if (bottom <= -4) continue; // floated completely off screen
        bars[writeIdx++] = bar;

        if (!pos) continue;
        const clampedTop = Math.max(0, top);
        const clampedBottom = Math.min(height, Math.max(clampedTop + 4, bottom));
        const barH = clampedBottom - clampedTop;
        if (barH <= 0.5) continue;

        const barW = Math.max(4, pos.w * (pos.isBlack ? 0.84 : 0.88));
        const x = pos.x - barW * 0.5;
        const radius = Math.min(4, barW * 0.35, barH * 0.45);

        // Fast GPU-friendly outer aura without CSS box-shadow reflow
        context.globalAlpha = 0.2;
        context.fillStyle = colors.rightHand;
        context.beginPath();
        context.roundRect(x - 2.5, Math.max(0, clampedTop - 2), barW + 5, barH + 4, radius + 2);
        context.fill();

        context.globalAlpha = 1;
        context.fillStyle = colors.rightHand;
        context.beginPath();
        context.roundRect(x, clampedTop, barW, barH, radius);
        context.fill();
      }

      bars.length = writeIdx;
      if (bars.length > 0) {
        running = true;
        frame = requestAnimationFrame(draw);
      }
    };

    const wake = () => {
      if (!running) {
        running = true;
        frame = requestAnimationFrame(draw);
      }
    };
    wakeRef.current = wake;

    syncLayout();
    if (barsRef.current.length > 0) wake();

    const observer = new ResizeObserver(() => {
      syncLayout();
      if (barsRef.current.length > 0) wake();
    });
    observer.observe(element);

    return () => {
      wakeRef.current = null;
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [enabled]);

  if (!enabled) return null;
  return <canvas ref={canvasRef} className="piano-rising-canvas" aria-hidden="true" />;
}

