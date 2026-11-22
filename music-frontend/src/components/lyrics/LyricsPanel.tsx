import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type ProjectDetail, type SongscriptionSegment } from "../../features/studio/api";
import type { Player } from "../../features/studio/utils/player";
import {
  LANGUAGE_OPTIONS,
  PIPELINE_STAGES,
  SECTION_OPTIONS,
  exportTranscriptFiles,
  formatClock,
  formatPreciseClock,
  resolveStageIndex,
  stripImportedLyricsFormat,
} from "../../features/studio/utils/lyricsUtils";
import { Icon } from "../ui/StudioIcon";
import { usePlayerState } from "../piano/PlayerBar";

const STAGE_LABELS: Record<string, string> = {
  preparing_audio: "Bước 1/5 · Đang chuẩn bị dữ liệu âm thanh…",
  separating_vocals: "Bước 2/5 · Đang tách riêng giọng hát AI (BS-RoFormer)…",
  recognizing_lyrics: "Bước 3/5 · Đang nhận diện lời hát đa ngôn ngữ…",
  aligning_timestamps: "Bước 4/5 · Đang căn chỉnh dấu thời gian từng từ/câu…",
  saving_transcript: "Bước 5/5 · Đang lưu bản lời bài hát…",
  completed: "Hoàn tất!",
  stopping: "Đang dừng tác vụ…",
};

export function LyricsPanel({
  pid,
  project,
  player,
}: {
  pid: string;
  project: ProjectDetail;
  player: Player;
}) {
  usePlayerState(player);
  const qc = useQueryClient();

  const lyricsQuery = useQuery({
    queryKey: ["project-lyrics", pid],
    queryFn: () => api.projectLyrics(pid),
    refetchInterval: q => {
      const st = q.state.data?.state;
      return st === "queued" || st === "running" ? 1000 : false;
    },
  });

  const data = lyricsQuery.data;
  const models = data?.models ?? [
    { id: "turbo", label: "Whisper Large-v3 Turbo (Nhanh & Chuẩn · Khuyên dùng 4GB VRAM)" },
    { id: "large-v3", label: "Whisper Large-v3 (Độ chính xác tối đa)" },
    { id: "medium", label: "Whisper Medium (Nhẹ VRAM)" },
    { id: "small", label: "Whisper Small (Siêu nhanh / Tương thích CPU)" },
  ];
  const vocalAvailable = data?.vocal_isolation_available ?? true;

  // Settings & Mode states
  const [mode, setMode] = useState<"extract" | "align">("extract");
  const [language, setLanguage] = useState("auto");
  const [modelId, setModelId] = useState("turbo");
  const [isolateVocals, setIsolateVocals] = useState(true);
  const [userLyricsText, setUserLyricsText] = useState("");
  const [showAlignForm, setShowAlignForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Editor & View states
  const [viewMode, setViewMode] = useState<"synced" | "editor" | "plain">("synced");
  const [searchQuery, setSearchQuery] = useState("");
  const [onlyUncertain, setOnlyUncertain] = useState(false);
  const [autoFollow, setAutoFollow] = useState(true);
  const [copied, setCopied] = useState(false);

  const transcriptVersion = `${pid}:${data?.updated ?? ""}:${data?.segment_count ?? 0}`;
  const [editSession, setEditSession] = useState<{
    version: string;
    segments: SongscriptionSegment[];
    undo: SongscriptionSegment[][];
    redo: SongscriptionSegment[][];
    dirty: boolean;
  } | null>(null);

  const currentSession = useMemo(() => {
    if (editSession && editSession.version === transcriptVersion) {
      return editSession;
    }
    const base = (data?.transcript?.segments ?? []).map(s => ({ ...s }));
    return {
      version: transcriptVersion,
      segments: base,
      undo: [] as SongscriptionSegment[][],
      redo: [] as SongscriptionSegment[][],
      dirty: false,
    };
  }, [editSession, transcriptVersion, data?.transcript?.segments]);

  const draftSegments = currentSession.segments;
  const undoStack = currentSession.undo;
  const redoStack = currentSession.redo;
  const dirty = currentSession.dirty;

  const [saving, setSaving] = useState(false);
  const [editingSegId, setEditingSegId] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");

  const listScrollRef = useRef<HTMLDivElement>(null);
  const lyricsFileRef = useRef<HTMLInputElement>(null);

  // Sync live playback clock from the shared Studio Player (reusing existing clock, no independent timer)
  const [currentTime, setCurrentTime] = useState(() => player.position());
  useEffect(() => {
    let raf = 0;
    const syncClock = () => {
      setCurrentTime(player.position());
      if (player.playing) {
        raf = requestAnimationFrame(syncClock);
      }
    };
    syncClock();
    return () => cancelAnimationFrame(raf);
  }, [player, player.playing, player.version]);

  const applyDraftChange = (next: SongscriptionSegment[]) => {
    setEditSession({
      version: transcriptVersion,
      segments: next.map((s, idx) => ({ ...s, id: idx + 1 })),
      undo: [...currentSession.undo.slice(-29), currentSession.segments],
      redo: [],
      dirty: true,
    });
  };

  const undo = () => {
    if (!currentSession.undo.length) return;
    const prev = currentSession.undo[currentSession.undo.length - 1];
    setEditSession({
      version: transcriptVersion,
      segments: prev,
      undo: currentSession.undo.slice(0, -1),
      redo: [...currentSession.redo, currentSession.segments],
      dirty: true,
    });
  };

  const redo = () => {
    if (!currentSession.redo.length) return;
    const next = currentSession.redo[currentSession.redo.length - 1];
    setEditSession({
      version: transcriptVersion,
      segments: next,
      undo: [...currentSession.undo, currentSession.segments],
      redo: currentSession.redo.slice(0, -1),
      dirty: true,
    });
  };

  const handleImportFile = async (file?: File | null) => {
    if (!file) return;
    try {
      const raw = await file.text();
      setUserLyricsText(stripImportedLyricsFormat(raw));
      setMode("align");
      setShowAlignForm(true);
      setError(null);
    } catch {
      setError("Không thể đọc tệp lời bài hát.");
    }
  };

  const runExtraction = async (forceSeparate = false) => {
    if (submitting) return;
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "align") {
        if (!userLyricsText.trim()) {
          setError("Vui lòng nhập hoặc tải lên lời bài hát (.txt/.lrc) để căn chỉnh dấu thời gian.");
          setSubmitting(false);
          return;
        }
        const res = await api.alignProjectLyrics(pid, {
          lyrics_text: userLyricsText.trim(),
          language,
          model: modelId,
          isolate_vocals: isolateVocals && vocalAvailable,
        });
        qc.setQueryData(["project-lyrics", pid], res);
        setShowAlignForm(false);
      } else {
        const res = await api.extractProjectLyrics(pid, {
          language,
          model: modelId,
          isolate_vocals: isolateVocals && vocalAvailable,
          force_separate: forceSeparate,
        });
        qc.setQueryData(["project-lyrics", pid], res);
      }
      void qc.invalidateQueries({ queryKey: ["project", pid] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const reAlignEditedScript = async () => {
    const lines: string[] = [];
    for (const s of draftSegments) {
      if (s.section) lines.push(`[${s.section}]`);
      if (s.text.trim()) lines.push(s.text.trim());
    }
    const payload = lines.join("\n").trim();
    if (!payload) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await api.alignProjectLyrics(pid, {
        lyrics_text: payload,
        language: data?.language || language || "auto",
        model: data?.model || modelId || "turbo",
        isolate_vocals: isolateVocals && vocalAvailable,
      });
      qc.setQueryData(["project-lyrics", pid], res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const saveEdits = async (segsToSave = draftSegments) => {
    if (saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.saveProjectLyrics(pid, segsToSave);
      qc.setQueryData(["project-lyrics", pid], res);
      setEditSession(null);
      setEditingSegId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const activeSegmentId = useMemo(() => {
    for (let i = draftSegments.length - 1; i >= 0; i--) {
      const seg = draftSegments[i];
      if (currentTime >= seg.start - 0.08 && currentTime <= seg.end + 0.35) {
        return seg.id;
      }
    }
    return null;
  }, [draftSegments, currentTime]);

  useEffect(() => {
    if (!player.playing || !autoFollow || activeSegmentId === null) return;
    const container = listScrollRef.current;
    const row = container?.querySelector<HTMLElement>(`[data-seg-id="${activeSegmentId}"]`);
    if (!container || !row) return;
    const rowRect = row.getBoundingClientRect();
    const boxRect = container.getBoundingClientRect();
    if (rowRect.top < boxRect.top + 32 || rowRect.bottom > boxRect.bottom - 42) {
      container.scrollTo({
        top: Math.max(0, container.scrollTop + (rowRect.top - boxRect.top) - boxRect.height * 0.3),
        behavior: "smooth",
      });
    }
  }, [activeSegmentId, player.playing, autoFollow]);

  const seekAndPlay = (sec: number) => {
    player.seek(Math.max(0, sec));
    setCurrentTime(player.position());
    if (!player.playing) player.play();
  };

  const uncertainCount = useMemo(
    () => draftSegments.filter(s => s.low_confidence || s.words?.some(w => w.uncertain)).length,
    [draftSegments],
  );

  const filteredSegments = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return draftSegments.filter(seg => {
      if (onlyUncertain && !seg.low_confidence && !seg.words?.some(w => w.uncertain)) return false;
      if (!q) return true;
      return (
        seg.text.toLowerCase().includes(q) ||
        Boolean(seg.chord && seg.chord.toLowerCase().includes(q)) ||
        Boolean(seg.section && seg.section.toLowerCase().includes(q))
      );
    });
  }, [draftSegments, searchQuery, onlyUncertain]);

  const updateLineField = (id: number, patch: Partial<SongscriptionSegment>) => {
    applyDraftChange(draftSegments.map(s => (s.id === id ? { ...s, ...patch, low_confidence: false } : s)));
  };

  const insertLineAfter = (idx: number) => {
    const prev = draftSegments[idx];
    const nextSeg = draftSegments[idx + 1];
    const start = prev ? Number((prev.end + 0.1).toFixed(2)) : 0;
    const end = nextSeg ? Number(Math.max(start + 1.2, nextSeg.start - 0.1).toFixed(2)) : Number((start + 2.5).toFixed(2));
    const inserted: SongscriptionSegment = {
      id: 0,
      start,
      end,
      text: "Lời mới…",
      lang: prev?.lang || data?.language || "en",
      section: null,
      words: [],
    };
    applyDraftChange([...draftSegments.slice(0, idx + 1), inserted, ...draftSegments.slice(idx + 1)]);
  };

  const deleteLineAt = (idx: number) => {
    if (draftSegments.length <= 1) return;
    applyDraftChange(draftSegments.filter((_, i) => i !== idx));
  };

  const splitLineAt = (idx: number) => {
    const seg = draftSegments[idx];
    if (!seg) return;
    const raw = seg.text.trim();
    if (raw.length < 2) return;
    const midChar = Math.floor(raw.length / 2);
    const spaceIdx = raw.indexOf(" ", Math.max(1, midChar - 8));
    const splitPos = spaceIdx > 0 && spaceIdx < raw.length - 1 ? spaceIdx : midChar;
    const firstText = raw.slice(0, splitPos).trim();
    const secondText = raw.slice(splitPos).trim();
    if (!firstText || !secondText) return;
    const midTime = Number(((seg.start + seg.end) / 2).toFixed(2));
    applyDraftChange([
      ...draftSegments.slice(0, idx),
      { ...seg, end: Math.max(Number((seg.start + 0.1).toFixed(2)), midTime), text: firstText, words: [] },
      { ...seg, start: midTime, text: secondText, section: null, words: [] },
      ...draftSegments.slice(idx + 1),
    ]);
  };

  const mergeWithNext = (idx: number) => {
    if (idx < 0 || idx >= draftSegments.length - 1) return;
    const cur = draftSegments[idx];
    const nxt = draftSegments[idx + 1];
    applyDraftChange([
      ...draftSegments.slice(0, idx),
      {
        ...cur,
        end: Math.max(cur.end, nxt.end),
        text: `${cur.text.trim()} ${nxt.text.trim()}`.trim(),
        words: [...(cur.words ?? []), ...(nxt.words ?? [])],
      },
      ...draftSegments.slice(idx + 2),
    ]);
  };

  const exporters = useMemo(
    () =>
      exportTranscriptFiles(
        project.title,
        data?.transcript?.language || data?.language || "en",
        data?.transcript?.language_name || data?.language_name || "Auto",
        draftSegments,
      ),
    [project.title, data?.transcript?.language, data?.transcript?.language_name, data?.language, data?.language_name, draftSegments],
  );

  const copyScript = async () => {
    const text = draftSegments.map(s => (s.section ? `[${s.section}]\n${s.text}` : s.text)).join("\n");
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Ignored
    }
  };

  if (!project.audio) {
    return (
      <div className="project-lyrics-card empty" role="status">
        <Icon name="music" size={24} />
        <h3>Project này chưa có tệp âm thanh gốc</h3>
        <p>
          Tính năng <strong>AI Lyrics Transcription</strong> yêu cầu bản thu âm gốc chứa giọng hát (MP3, WAV, FLAC, M4A)
          và không tạo lời giả từ tệp Piano MIDI. Hãy tạo hoặc mở một Project có tệp âm thanh để trích xuất lời bài hát.
        </p>
      </div>
    );
  }

  const isRunning = data?.state === "queued" || data?.state === "running";
  const hasTranscript = Boolean(data?.transcript && draftSegments.length > 0);
  const stageIdx = resolveStageIndex(data?.message, data?.progress);
  const isCoverSolo = player.mix >= 0.95;

  return (
    <div className="project-lyrics-card">
      {/* Control Header */}
      <div className="project-lyrics-controls">
        <div className="project-lyrics-controls-top">
          <div className="songscription-mode-tabs" role="tablist" aria-label="Chế độ lời bài hát">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "extract" && !showAlignForm}
              onClick={() => {
                setMode("extract");
                setShowAlignForm(false);
              }}
            >
              ✦ Trích xuất tự động (Mode 1)
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "align" || showAlignForm}
              onClick={() => {
                setMode("align");
                setShowAlignForm(true);
              }}
            >
              ⇄ Căn chỉnh lời có sẵn (Mode 2)
            </button>
          </div>

          <select
            className="project-lyrics-select"
            value={language}
            onChange={e => setLanguage(e.target.value)}
            aria-label="Ngôn ngữ bài hát"
          >
            {LANGUAGE_OPTIONS.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>

          <select
            className="project-lyrics-select"
            value={modelId}
            onChange={e => setModelId(e.target.value)}
            aria-label="Mô hình AI"
          >
            {models.map(m => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>

          <label className="project-lyrics-vocal-toggle">
            <input
              type="checkbox"
              checked={isolateVocals && vocalAvailable}
              disabled={!vocalAvailable}
              onChange={e => setIsolateVocals(e.target.checked)}
            />
            <span>Tách giọng hát AI {data?.vocal_cached ? "(Đã cache)" : ""}</span>
          </label>

          <button
            type="button"
            className="btn small"
            onClick={() => lyricsFileRef.current?.click()}
            title="Nhập lời bài hát từ tệp .TXT, .LRC hoặc .SRT"
          >
            Nhập .TXT/.LRC
          </button>
          <input
            ref={lyricsFileRef}
            type="file"
            accept=".txt,.lrc,.srt"
            hidden
            onChange={e => {
              void handleImportFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />

          <button
            type="button"
            className="btn primary small"
            disabled={submitting || isRunning}
            onClick={() => void runExtraction(false)}
          >
            {mode === "align" || showAlignForm
              ? "Đồng bộ lời với nhạc"
              : hasTranscript
                ? "↻ Trích xuất lại"
                : "✦ Trích xuất lời bài hát"}
          </button>
        </div>

        {(mode === "align" || showAlignForm) && !isRunning && (
          <div className="project-lyrics-align-drawer">
            <textarea
              rows={4}
              value={userLyricsText}
              onChange={e => setUserLyricsText(e.target.value)}
              placeholder="Dán lời bài hát chuẩn vào đây (mỗi câu một dòng, có thể kèm [Verse], [Chorus]…). Hệ thống sẽ giữ nguyên văn bản của bạn và căn chỉnh dấu thời gian khớp với giọng hát."
              aria-label="Lời bài hát để đồng bộ"
            />
          </div>
        )}
      </div>

      {error && (
        <div className="alert" role="alert" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      {/* Running State */}
      {isRunning ? (
        <div className="songscription-running-card" aria-busy="true">
          <div className="songscription-running-head">
            <span className="songscription-pulse" aria-hidden="true" />
            <div>
              <h2>Đang xử lý AI Lyrics cho {project.title}</h2>
              <p>
                {data?.message
                  ? STAGE_LABELS[data.message] ?? data.message
                  : "Đang chờ trong hàng đợi GPU…"}
              </p>
            </div>
            {data?.job && (
              <button
                type="button"
                className="btn small"
                onClick={async () => {
                  await api.cancelJob(data.job!);
                  void qc.invalidateQueries({ queryKey: ["project-lyrics", pid] });
                }}
              >
                <Icon name="stop" size={14} /> Dừng
              </button>
            )}
          </div>
          <div className="songscription-progress-track">
            <div
              className={`songscription-progress-fill ${data?.progress == null ? "indeterminate" : ""}`}
              style={
                data?.progress != null
                  ? { width: `${Math.max(6, Math.round(data.progress * 100))}%` }
                  : undefined
              }
            />
          </div>
          <div className="songscription-stage-steps">
            {PIPELINE_STAGES.slice(0, 5).map((st, idx) => (
              <span
                key={st.key}
                className={`songscription-step-pill ${idx < stageIdx ? "done" : idx === stageIdx ? "current" : ""}`}
              >
                {idx < stageIdx ? "✓ " : ""}
                {st.short}
              </span>
            ))}
          </div>
          {data?.log && data.log.length > 0 && (
            <details className="songscription-log" open>
              <summary>Nhật ký xử lý ({data.log.length})</summary>
              <pre>{data.log.join("\n")}</pre>
            </details>
          )}
        </div>
      ) : data?.state === "error" ? (
        <div className="songscription-empty songscription-error-box" role="alert">
          <h2>Không thể hoàn tất trích xuất lời</h2>
          <p>{data.error}</p>
          <div className="songscription-error-actions">
            <button type="button" className="btn primary" onClick={() => void runExtraction(false)}>
              ↻ Thử lại ngay
            </button>
          </div>
        </div>
      ) : !data?.transcript ? (
        <div className="songscription-empty">
          <div className="songscription-empty-badge">✦</div>
          <h2>Sẵn sàng trích xuất lời bài hát từ tệp âm thanh gốc</h2>
          <p>
            Bấm <strong>✦ Trích xuất lời bài hát</strong> ở thanh công cụ phía trên để AI tách riêng giọng hát từ{" "}
            <code>{project.audio}</code>, tự động nhận diện ngôn ngữ và đồng bộ từng câu/từ với thanh phát nhạc bên dưới.
          </p>
        </div>
      ) : (
        <div className="songscription-viewer">
          <div className="songscription-viewer-head">
            <div className="songscription-viewer-title-row">
              <div className="songscription-badges">
                {data.transcript.language_name && (
                  <span className="song-badge primary">
                    ✦ {data.transcript.language_name}
                    {data.transcript.language_confidence != null &&
                      ` · ${Math.round(data.transcript.language_confidence * 100)}%`}
                  </span>
                )}
                <span className="song-badge subtle">
                  {data.transcript.mode === "align" ? "Căn chỉnh lời gốc" : "AI Tự trích xuất"}
                </span>
                <span className="song-badge subtle">
                  {data.transcript.alignment_method === "word" ? "Word-Sync" : "Line-Sync"}
                </span>
                {data.vocal_isolated && (
                  <span className="song-badge subtle">
                    Đã tách giọng hát AI {data.vocal_cached ? "(Cache)" : ""}
                  </span>
                )}
                <span className="song-badge subtle">
                  ⏱ {formatClock(currentTime)} / {formatClock(player.duration || data.duration || 0)}
                </span>
                {uncertainCount > 0 && (
                  <button
                    type="button"
                    className={`song-badge warning-btn ${onlyUncertain ? "active" : ""}`}
                    onClick={() => setOnlyUncertain(v => !v)}
                  >
                    ⚠ {uncertainCount} câu cần xem lại
                  </button>
                )}
              </div>

              <div className="songscription-head-actions">
                {isCoverSolo && (
                  <button
                    type="button"
                    className="song-badge accent"
                    onClick={() => player.setMix(0.35)}
                    title="Đang phát 100% Piano Cover. Nhấp để bật thêm âm thanh bài gốc (Original) nghe giọng hát."
                  >
                    🎹 Đang nghe Piano Cover (Tham chiếu lời) · Bật tiếng hát gốc
                  </button>
                )}
                <label className="songscription-autofollow">
                  <input
                    type="checkbox"
                    checked={autoFollow}
                    onChange={e => setAutoFollow(e.target.checked)}
                  />
                  <span>Tự cuộn theo nhạc</span>
                </label>
                {dirty && (
                  <button
                    type="button"
                    className="songscription-save-btn"
                    disabled={saving}
                    onClick={() => void saveEdits()}
                  >
                    {saving ? "Đang lưu…" : "✓ Lưu chỉnh sửa"}
                  </button>
                )}
                <button
                  type="button"
                  className="songscription-realign-btn"
                  disabled={submitting}
                  onClick={() => void reAlignEditedScript()}
                  title="Đồng bộ lại dấu thời gian cho lời đã chỉnh sửa (không cần tách giọng lại)"
                >
                  ⇄ Đồng bộ lại
                </button>
              </div>
            </div>
          </div>

          <div className="songscription-toolbar">
            <div className="songscription-mode-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "synced"}
                onClick={() => setViewMode("synced")}
              >
                Lời đồng bộ (Live)
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "editor"}
                onClick={() => setViewMode("editor")}
              >
                Trình sửa lời &amp; Thời gian
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "plain"}
                onClick={() => setViewMode("plain")}
              >
                Văn bản đầy đủ
              </button>
            </div>

            {viewMode === "editor" ? (
              <div className="songscription-editor-tools">
                <button type="button" disabled={!undoStack.length} onClick={undo}>
                  ↶ Hoàn tác
                </button>
                <button type="button" disabled={!redoStack.length} onClick={redo}>
                  ↷ Làm lại
                </button>
                <button type="button" onClick={() => insertLineAfter(draftSegments.length - 1)}>
                  ＋ Thêm dòng
                </button>
              </div>
            ) : (
              viewMode === "synced" && (
                <input
                  className="songscription-search"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Tìm câu hát, hợp âm…"
                  aria-label="Tìm câu hát"
                />
              )
            )}

            <div className="songscription-exports">
              <button type="button" onClick={() => void copyScript()}>
                {copied ? "✓ Đã chép" : "Sao chép"}
              </button>
              <button type="button" onClick={() => exporters.exportTxt(true)}>
                TXT ↓
              </button>
              <button type="button" onClick={exporters.exportLrc}>
                LRC ↓
              </button>
              {exporters.hasWordTimings && (
                <button type="button" onClick={exporters.exportEnhancedLrc}>
                  ELRC ↓
                </button>
              )}
              <button type="button" onClick={exporters.exportSrt}>
                SRT ↓
              </button>
              <button type="button" onClick={() => exporters.exportJson({ project_id: pid })}>
                JSON ↓
              </button>
            </div>
          </div>

          <div
            className="songscription-body"
            ref={listScrollRef}
            onWheel={() => {
              if (player.playing && autoFollow) setAutoFollow(false);
            }}
          >
            {!draftSegments.length ? (
              <div className="songscription-no-lyrics">
                Không phát hiện thấy câu hát có lời trong bản thu này (bản nhạc hòa tấu không lời).
              </div>
            ) : viewMode === "plain" ? (
              <div className="songscription-plain-text">
                {draftSegments.map(s => (s.section ? `[${s.section}]\n${s.text}` : s.text)).join("\n")}
              </div>
            ) : viewMode === "editor" ? (
              <div className="songscription-editor-table">
                {draftSegments.map((seg, idx) => (
                  <div
                    key={seg.id}
                    data-seg-id={seg.id}
                    className={`songscription-editor-row ${seg.id === activeSegmentId ? "active" : ""} ${
                      seg.low_confidence ? "uncertain" : ""
                    }`}
                  >
                    <div className="songscription-editor-times">
                      <button
                        type="button"
                        className="songscription-time-pill"
                        onClick={() => seekAndPlay(seg.start)}
                      >
                        ▶ {formatClock(seg.start)}
                      </button>
                      <label>
                        <span>Bắt đầu</span>
                        <input
                          type="number"
                          step={0.1}
                          min={0}
                          value={seg.start}
                          onChange={e => updateLineField(seg.id, { start: Math.max(0, Number(e.target.value)) })}
                        />
                      </label>
                      <label>
                        <span>Kết thúc</span>
                        <input
                          type="number"
                          step={0.1}
                          min={seg.start}
                          value={seg.end}
                          onChange={e =>
                            updateLineField(seg.id, { end: Math.max(seg.start + 0.05, Number(e.target.value)) })
                          }
                        />
                      </label>
                      <select
                        value={seg.section ?? ""}
                        aria-label="Phân đoạn bài hát"
                        onChange={e => updateLineField(seg.id, { section: e.target.value || null })}
                      >
                        {SECTION_OPTIONS.map(sec => (
                          <option key={sec} value={sec}>
                            {sec ? `[${sec}]` : "— Đoạn —"}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="songscription-editor-main">
                      <input
                        className="songscription-editor-input"
                        value={seg.text}
                        onChange={e => updateLineField(seg.id, { text: e.target.value })}
                        aria-label={`Dòng lời ${idx + 1}`}
                      />
                      <div className="songscription-editor-row-actions">
                        <button type="button" onClick={() => splitLineAt(idx)}>
                          ✂ Tách
                        </button>
                        <button
                          type="button"
                          disabled={idx >= draftSegments.length - 1}
                          onClick={() => mergeWithNext(idx)}
                        >
                          ⊕ Gộp
                        </button>
                        <button type="button" onClick={() => insertLineAfter(idx)}>
                          ＋ Chèn
                        </button>
                        <button
                          type="button"
                          className="danger"
                          disabled={draftSegments.length <= 1}
                          onClick={() => deleteLineAt(idx)}
                        >
                          × Xóa
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="songscription-lines">
                {filteredSegments.map(seg => {
                  const isActive = seg.id === activeSegmentId;
                  const isEditing = editingSegId === seg.id;
                  const isUncertain = Boolean(seg.low_confidence || seg.words?.some(w => w.uncertain));
                  return (
                    <div key={seg.id}>
                      {seg.section && (
                        <div className="songscription-section-divider">
                          <span>[{seg.section}]</span>
                        </div>
                      )}
                      <div
                        data-seg-id={seg.id}
                        className={`songscription-line ${isActive ? "active" : ""} ${
                          isUncertain ? "uncertain" : ""
                        }`}
                      >
                        <button
                          type="button"
                          className="songscription-time-pill"
                          onClick={() => seekAndPlay(seg.start)}
                          title={`Phát từ ${formatPreciseClock(seg.start)}`}
                        >
                          {formatClock(seg.start)}
                        </button>
                        <div className="songscription-line-content">
                          {(seg.chord || isUncertain) && (
                            <div className="songscription-line-tags">
                              {seg.chord && <span className="line-chord-tag">{seg.chord}</span>}
                              {isUncertain && <span className="line-uncertain-tag">Cần kiểm tra</span>}
                            </div>
                          )}
                          {isEditing ? (
                            <div className="songscription-edit-row">
                              <input
                                value={editingText}
                                onChange={e => setEditingText(e.target.value)}
                                onKeyDown={e => {
                                  if (e.key === "Enter") {
                                    const next = draftSegments.map(s =>
                                      s.id === seg.id
                                        ? { ...s, text: editingText.trim() || s.text, low_confidence: false }
                                        : s,
                                    );
                                    applyDraftChange(next);
                                    void saveEdits(next);
                                  }
                                  if (e.key === "Escape") setEditingSegId(null);
                                }}
                                autoFocus
                              />
                              <button
                                type="button"
                                disabled={saving}
                                onClick={() => {
                                  const next = draftSegments.map(s =>
                                    s.id === seg.id
                                      ? { ...s, text: editingText.trim() || s.text, low_confidence: false }
                                      : s,
                                  );
                                  applyDraftChange(next);
                                  void saveEdits(next);
                                }}
                              >
                                Lưu
                              </button>
                              <button type="button" onClick={() => setEditingSegId(null)}>
                                Hủy
                              </button>
                            </div>
                          ) : (
                            <p className="songscription-line-text" onClick={() => seekAndPlay(seg.start)}>
                              {seg.words && seg.words.length > 0
                                ? seg.words.map((w, wi) => {
                                    const sung =
                                      isActive && currentTime >= w.start - 0.04 && currentTime <= w.end + 0.15;
                                    return (
                                      <span
                                        key={wi}
                                        className={`${sung ? "word-active" : ""} ${
                                          w.uncertain ? "word-uncertain" : ""
                                        }`}
                                      >
                                        {w.word}
                                      </span>
                                    );
                                  })
                                : seg.text}
                            </p>
                          )}
                        </div>
                        {!isEditing && (
                          <button
                            type="button"
                            className="songscription-edit-trigger"
                            aria-label="Chỉnh sửa câu hát"
                            onClick={() => {
                              setEditingSegId(seg.id);
                              setEditingText(seg.text);
                            }}
                          >
                            ✎
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
