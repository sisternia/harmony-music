import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadVerovioToolkit } from '../../../studio/utils/verovio';
import type { PianoSong } from '../utils/transcription';
import { buildMeasureTimings, midiToMusicXml, type EngravedMeasureTiming } from '../utils/midiToMusicXml';

type Props = {
  song: PianoSong;
  readTime: () => number;
  playing: boolean;
  position: number;
  onSeek: (time: number) => void;
};

type Anchor = { time: number; x: number };

type MeasureBox = {
  measureNumber: number;
  start: number;
  end: number;
  leftX: number;
  rightX: number;
  topY: number;
  bottomY: number;
  anchors: Anchor[];
};

const SCALES = [30, 35, 40, 45, 50, 56];
const DEFAULT_SCALE_IDX = 2; // scale 40 -> 100%
const EMPTY_PAGES: string[] = [];

const pedalMarks = (xml: string) => xml.replace(/(<pedal type="[a-z]+") line="yes"/g, '$1 line="no"');

function buildTimemapEventTimes(
  timemap: import('verovio/esm').VerovioTimemapEntry[],
  measures: EngravedMeasureTiming[],
): Map<string, number> {
  const out = new Map<string, number>();
  let measureIdx = -1;
  let measureStartQ = 0;
  for (const entry of timemap) {
    if (entry.measureOn) {
      measureIdx = Math.min(measures.length - 1, measureIdx + 1);
      measureStartQ = entry.qstamp;
    }
    if (measureIdx < 0 || measureIdx >= measures.length) continue;
    const m = measures[measureIdx];
    const relQ = Math.max(0, entry.qstamp - measureStartQ);
    const frac = m.quarters > 0 ? Math.min(1, relQ / m.quarters) : 0;
    const time = m.start + frac * (m.end - m.start);
    for (const id of entry.on ?? []) out.set(id, time);
    for (const id of entry.restsOn ?? []) out.set(id, time);
  }
  return out;
}

function computeMeasureBoxes(
  canvasEl: HTMLDivElement,
  measures: EngravedMeasureTiming[],
  eventTimes: Map<string, number>,
): MeasureBox[] {
  const canvasRect = canvasEl.getBoundingClientRect();
  if (!canvasRect.width || !canvasRect.height) return [];

  const measureByNum = new Map<number, EngravedMeasureTiming>();
  for (const m of measures) measureByNum.set(m.measureNumber, m);

  const elements = canvasEl.querySelectorAll<SVGGElement>('g.measure');
  const boxes: MeasureBox[] = [];

  elements.forEach((measureEl, seqIdx) => {
    const attrN = Number(measureEl.getAttribute('data-n'));
    const timing = (Number.isFinite(attrN) && measureByNum.get(attrN)) || measures[seqIdx];
    if (!timing) return;

    const staff1Lines = measureEl.querySelectorAll<SVGPathElement>('g.staff[data-n="1"] > path');
    const staff2Lines = measureEl.querySelectorAll<SVGPathElement>('g.staff[data-n="2"] > path');
    const measureRect = measureEl.getBoundingClientRect();
    if (!measureRect.width) return;

    let topY = measureRect.top - canvasRect.top;
    let bottomY = measureRect.bottom - canvasRect.top;
    let leftX = measureRect.left - canvasRect.left;
    let rightX = measureRect.right - canvasRect.left;

    if (staff1Lines.length >= 5) {
      const firstLine = staff1Lines[0].getBoundingClientRect();
      const lastLine1 = staff1Lines[staff1Lines.length - 1].getBoundingClientRect();
      topY = firstLine.top - canvasRect.top - 3;
      leftX = firstLine.left - canvasRect.left;
      rightX = firstLine.right - canvasRect.left;
      bottomY = lastLine1.bottom - canvasRect.top + 3;
    }
    if (staff2Lines.length >= 5) {
      const lastLine2 = staff2Lines[staff2Lines.length - 1].getBoundingClientRect();
      bottomY = lastLine2.bottom - canvasRect.top + 3;
    }

    const rawAnchors: Anchor[] = [];
    measureEl.querySelectorAll<SVGGElement>('g.note[id], g.rest[id]').forEach(el => {
      const t = eventTimes.get(el.id);
      if (t === undefined || t < timing.start - 0.05 || t > timing.end + 0.05) return;
      const head = el.querySelector<SVGGraphicsElement>('.notehead') ?? el;
      const r = head.getBoundingClientRect();
      if (!r.width && !r.height) return;
      const x = r.left + Math.min(7, r.width * 0.45) - canvasRect.left;
      rawAnchors.push({ time: Math.max(timing.start, Math.min(timing.end, t)), x });
    });

    rawAnchors.sort((a, b) => a.time - b.time || a.x - b.x);

    // Merge anchors at the same timestamp (e.g. treble + bass notes on the same beat)
    const merged: Anchor[] = [];
    for (const a of rawAnchors) {
      const prev = merged[merged.length - 1];
      if (prev && Math.abs(a.time - prev.time) <= 0.012) {
        prev.x = Math.min(prev.x, a.x);
      } else {
        merged.push({ ...a });
      }
    }

    // Ensure non-decreasing X across time
    for (let i = 1; i < merged.length; i++) {
      if (merged[i].x < merged[i - 1].x) merged[i].x = merged[i - 1].x;
    }

    const musicStartX = merged.length ? merged[0].x : Math.min(rightX - 6, leftX + 18);
    if (!merged.length || merged[0].time > timing.start + 0.015) {
      merged.unshift({ time: timing.start, x: Math.max(leftX + 4, musicStartX - 8) });
    } else {
      merged[0].time = timing.start;
    }

    const endX = Math.max(merged[merged.length - 1].x + 4, rightX - 2);
    merged.push({ time: timing.end, x: endX });

    boxes.push({
      measureNumber: timing.measureNumber,
      start: timing.start,
      end: timing.end,
      leftX,
      rightX: Math.max(rightX, endX),
      topY,
      bottomY,
      anchors: merged,
    });
  });

  boxes.sort((a, b) => a.start - b.start || a.measureNumber - b.measureNumber);
  return boxes;
}

function locateCursor(boxes: MeasureBox[], time: number): { x: number; topY: number; bottomY: number } | null {
  if (!boxes.length) return null;
  let low = 0;
  let high = boxes.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >>> 1;
    if (boxes[mid].start <= time) low = mid;
    else high = mid - 1;
  }
  const box = boxes[low];
  const clamped = Math.max(box.start, Math.min(box.end, time));
  const anchors = box.anchors;
  let x = anchors[0].x;
  for (let i = 0; i < anchors.length - 1; i++) {
    const a0 = anchors[i];
    const a1 = anchors[i + 1];
    if (clamped <= a1.time || i === anchors.length - 2) {
      const dt = a1.time - a0.time;
      const frac = dt > 1e-4 ? Math.max(0, Math.min(1, (clamped - a0.time) / dt)) : 0;
      x = a0.x + frac * (a1.x - a0.x);
      break;
    }
  }
  return { x, topY: box.topY, bottomY: box.bottomY };
}

type RenderState = {
  key: string;
  pages: string[];
  complete: boolean;
  error: string | null;
  timemapEvents: Map<string, number> | null;
};

export default function MidiSheetScore({ song, readTime, playing, position, onSeek }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const layoutRef = useRef<MeasureBox[]>([]);
  const lastScrollTargetRef = useRef<number>(-1);

  const [scaleIdx, setScaleIdx] = useState(DEFAULT_SCALE_IDX);
  const [follow, setFollow] = useState(true);
  const [containerWidth, setContainerWidth] = useState(0);
  const [renderState, setRenderState] = useState<RenderState>({
    key: '',
    pages: [],
    complete: false,
    error: null,
    timemapEvents: null,
  });

  const engraved = useMemo(() => {
    if (song.musicXml) {
      return {
        xml: song.musicXml,
        measures: buildMeasureTimings(song),
        eventTimes: new Map<string, number>(),
        fromBackendXml: true,
      };
    }
    return { ...midiToMusicXml(song), fromBackendXml: false };
  }, [song]);

  const scale = SCALES[scaleIdx];
  const zoomPercent = Math.round((scale / SCALES[DEFAULT_SCALE_IDX]) * 100);
  const renderKey = `${song.id}:${scale}:${containerWidth}:${engraved.xml.length}`;

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const updateWidth = () => {
      const quantized = Math.max(320, Math.round(el.clientWidth / 32) * 32);
      setContainerWidth(prev => (prev === quantized ? prev : quantized));
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!containerWidth) return;
    let cancelled = false;
    let frameId = 0;

    loadVerovioToolkit()
      .then(tk => {
        if (cancelled) return;
        const pageWidth = Math.max(1150, Math.round(((containerWidth - 40) * 100) / scale));
        tk.setOptions({
          svgViewBox: true,
          header: 'none',
          footer: 'none',
          breaks: 'auto',
          font: 'Leipzig',
          scale,
          pageWidth,
          pageHeight: 2800,
          adjustPageHeight: true,
          pageMarginTop: 28,
          pageMarginBottom: 28,
          pageMarginLeft: 36,
          pageMarginRight: 36,
          svgAdditionalAttribute: ['measure@n', 'staff@n'],
        });
        if (!tk.loadData(pedalMarks(engraved.xml))) {
          throw new Error(tk.getLog() || 'Không dựng được phổ nhạc.');
        }
        const totalPages = tk.getPageCount();
        const timemapEvents = engraved.fromBackendXml
          ? buildTimemapEventTimes(tk.renderToTimemap({ includeMeasures: true, includeRests: true }), engraved.measures)
          : null;
        const firstBatchCount = Math.min(totalPages, 2);
        const collected: string[] = [];
        for (let p = 1; p <= firstBatchCount; p++) collected.push(tk.renderToSVG(p));

        if (cancelled) return;
        setRenderState({
          key: renderKey,
          pages: [...collected],
          complete: collected.length >= totalPages,
          error: null,
          timemapEvents,
        });

        let nextPage = firstBatchCount + 1;
        const pumpRemaining = () => {
          if (cancelled || nextPage > totalPages) return;
          const endPage = Math.min(totalPages, nextPage + 1);
          for (; nextPage <= endPage; nextPage++) {
            collected.push(tk.renderToSVG(nextPage));
          }
          if (cancelled) return;
          setRenderState({
            key: renderKey,
            pages: [...collected],
            complete: nextPage > totalPages,
            error: null,
            timemapEvents,
          });
          if (nextPage <= totalPages) {
            frameId = requestAnimationFrame(pumpRemaining);
          }
        };
        if (nextPage <= totalPages) {
          frameId = requestAnimationFrame(pumpRemaining);
        }
      })
      .catch(reason => {
        if (!cancelled) {
          setRenderState({
            key: renderKey,
            pages: [],
            complete: true,
            error: reason instanceof Error ? reason.message : String(reason),
            timemapEvents: null,
          });
        }
      });

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
    };
  }, [containerWidth, scale, engraved, renderKey]);

  const activeEventTimes = renderState.timemapEvents ?? engraved.eventTimes;
  const pages = renderState.key === renderKey ? renderState.pages : EMPTY_PAGES;
  const error = renderState.key === renderKey ? renderState.error : null;

  const paintCursor = useCallback(
    (now: number, allowAutoScroll: boolean) => {
      const cursorEl = cursorRef.current;
      const scrollEl = scrollRef.current;
      if (!cursorEl) return;
      const hit = locateCursor(layoutRef.current, now);
      if (!hit) {
        cursorEl.style.opacity = '0';
        return;
      }
      cursorEl.style.opacity = '1';
      cursorEl.style.transform = `translate3d(${hit.x.toFixed(1)}px, ${hit.topY.toFixed(1)}px, 0)`;
      cursorEl.style.height = `${Math.max(24, hit.bottomY - hit.topY).toFixed(1)}px`;

      if (allowAutoScroll && follow && scrollEl) {
        const vh = scrollEl.clientHeight;
        const relTop = hit.topY - scrollEl.scrollTop;
        const relBottom = hit.bottomY - scrollEl.scrollTop;
        if (relTop < 16 || relBottom > vh - 20) {
          const desiredTop = Math.max(0, Math.round(hit.topY - Math.max(20, vh * 0.16)));
          if (Math.abs(desiredTop - lastScrollTargetRef.current) > 24) {
            lastScrollTargetRef.current = desiredTop;
            scrollEl.scrollTo({ top: desiredTop, behavior: playing ? 'smooth' : 'auto' });
          }
        }
      }
    },
    [follow, playing],
  );

  // Recompute measure layout whenever rendered pages or dimensions change
  useEffect(() => {
    const canvasEl = canvasRef.current;
    if (!canvasEl || !pages.length) {
      layoutRef.current = [];
      return;
    }
    const rebuild = () => {
      layoutRef.current = computeMeasureBoxes(canvasEl, engraved.measures, activeEventTimes);
      lastScrollTargetRef.current = -1;
      paintCursor(playing ? readTime() : position, false);
    };
    rebuild();
    const observer = new ResizeObserver(rebuild);
    observer.observe(canvasEl);
    return () => observer.disconnect();
  }, [pages, engraved.measures, activeEventTimes, paintCursor, playing, position, readTime]);

  // Animate playback cursor smoothly at 60fps
  useEffect(() => {
    if (!playing) {
      lastScrollTargetRef.current = -1;
      paintCursor(position, follow);
      return;
    }
    let frame = 0;
    const tick = () => {
      paintCursor(readTime(), true);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, position, follow, readTime, paintCursor]);

  const downloadMusicXml = () => {
    const blob = new Blob([engraved.xml], { type: 'application/vnd.recordare.musicxml+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const safeTitle = (song.title || 'piano-score').replace(/[\\/:*?"<>|]+/g, '_');
    link.href = url;
    link.download = `${safeTitle}.musicxml`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="midi-sheet-score">
      <div className="midi-sheet-toolbar">
        <label className="midi-sheet-follow">
          <input
            type="checkbox"
            checked={follow}
            onChange={event => {
              lastScrollTargetRef.current = -1;
              setFollow(event.target.checked);
            }}
          />
          <span>Theo bản nhạc</span>
        </label>
        <div className="midi-sheet-zoom" role="group" aria-label="Thu phóng phổ nhạc">
          <button
            type="button"
            disabled={scaleIdx <= 0}
            onClick={() => setScaleIdx(idx => Math.max(0, idx - 1))}
            aria-label="Thu nhỏ phổ nhạc"
            title="Thu nhỏ"
          >
            −
          </button>
          <span>{zoomPercent}%</span>
          <button
            type="button"
            disabled={scaleIdx >= SCALES.length - 1}
            onClick={() => setScaleIdx(idx => Math.min(SCALES.length - 1, idx + 1))}
            aria-label="Phóng to phổ nhạc"
            title="Phóng to"
          >
            ＋
          </button>
        </div>
        <button type="button" className="midi-sheet-download" onClick={downloadMusicXml} title="Tải về tệp MusicXML">
          ↓ MusicXML
        </button>
      </div>

      <div
        className="midi-sheet-scroll"
        ref={scrollRef}
        onWheel={() => {
          if (playing && follow) setFollow(false);
        }}
      >
        {error ? (
          <div className="midi-sheet-status midi-sheet-error" role="alert">
            Không thể hiển thị phổ nhạc: {error}
          </div>
        ) : !pages.length ? (
          <div className="midi-sheet-status" role="status">
            Đang dựng phổ nhạc…
          </div>
        ) : (
          <div
            className="midi-sheet-canvas"
            ref={canvasRef}
            onClick={event => {
              const canvasEl = canvasRef.current;
              if (!canvasEl || !layoutRef.current.length) return;
              const rect = canvasEl.getBoundingClientRect();
              const clickX = event.clientX - rect.left;
              const clickY = event.clientY - rect.top;
              const hitBox = layoutRef.current.find(
                b =>
                  clickY >= b.topY - 32 &&
                  clickY <= b.bottomY + 32 &&
                  clickX >= b.leftX - 8 &&
                  clickX <= b.rightX + 8,
              );
              if (!hitBox) return;
              const anchors = hitBox.anchors;
              let targetTime = hitBox.start;
              if (clickX <= anchors[0].x) {
                targetTime = anchors[0].time;
              } else if (clickX >= anchors[anchors.length - 1].x) {
                targetTime = Math.max(hitBox.start, hitBox.end - 0.02);
              } else {
                for (let i = 0; i < anchors.length - 1; i++) {
                  const a0 = anchors[i];
                  const a1 = anchors[i + 1];
                  if (clickX <= a1.x || i === anchors.length - 2) {
                    const dx = a1.x - a0.x;
                    const frac = dx > 0.5 ? Math.max(0, Math.min(1, (clickX - a0.x) / dx)) : 0;
                    targetTime = a0.time + frac * (a1.time - a0.time);
                    break;
                  }
                }
              }
              lastScrollTargetRef.current = -1;
              onSeek(targetTime);
            }}
          >
            {pages.map((svg, idx) => (
              <div key={idx} className="midi-sheet-page" dangerouslySetInnerHTML={{ __html: svg }} />
            ))}
            <div ref={cursorRef} className="midi-sheet-cursor" aria-hidden="true" />
          </div>
        )}
      </div>
    </div>
  );
}
