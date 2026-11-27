import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type ProjectDetail, type SongscriptionSegment } from "../../features/studio/api";
import type { Player } from "../../features/studio/utils/player";
import {
  DEFAULT_SONGSCRIPTION_MODEL_ID,
  getSongscriptionModels,
  normalizeSongscriptionModelId,
} from "../../features/studio/utils/songscriptionModels";
import {
  LANGUAGE_OPTIONS,
  PIPELINE_STAGES,
  STAGE_LABELS,
  exportTranscriptFiles,
  formatClock,
  formatPreciseClock,
  resolveStageIndex,
  stripImportedLyricsFormat,
} from "../../features/studio/utils/lyricsUtils";
import { Icon } from "../ui/StudioIcon";
import { usePlayerState } from "../piano/PlayerBar";
import {
  KARAOKE_SYNC_LEAD_SEC,
  getWordPlaybackState,
  prepareWordsForKaraoke,
} from "../../features/studio/utils/lyricsPlayback";

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
  const models = getSongscriptionModels(data?.models);
  const vocalAvailable = data?.vocal_isolation_available ?? true;

  // Setup drawer (shown on empty state or when clicking "Trích xuất / Căn chỉnh lại" in ⋯ menu)
  const [setupOpen, setSetupOpen] = useState(false);
  const [mode, setMode] = useState<"extract" | "align">("extract");
  const [language, setLanguage] = useState("auto");
  const [modelId, setModelId] = useState(DEFAULT_SONGSCRIPTION_MODEL_ID);
  const selectedModelId = normalizeSongscriptionModelId(modelId);
  const [isolateVocals, setIsolateVocals] = useState(true);
  const [userLyricsText, setUserLyricsText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // View & menus
  const [viewMode, setViewMode] = useState<"synced" | "plain">("synced");
  const [searchQuery, setSearchQuery] = useState("");
  const [onlyUncertain, setOnlyUncertain] = useState(false);
  const [autoFollow, setAutoFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);

  const exportRef = useRef<HTMLDivElement>(null);
  const moreRef = useRef<HTMLDivElement>(null);
  const listScrollRef = useRef<HTMLDivElement>(null);
  const lyricsFileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!exportOpen && !moreOpen) return;
    const onDown = (e: MouseEvent) => {
      const node = e.target as Node;
      if (exportOpen && exportRef.current && !exportRef.current.contains(node)) setExportOpen(false);
      if (moreOpen && moreRef.current && !moreRef.current.contains(node)) setMoreOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [exportOpen, moreOpen]);

  const transcriptVersion = `${pid}:${data?.updated ?? ""}:${data?.segment_count ?? 0}`;
  const [editSession, setEditSession] = useState<{
    version: string;
    segments: SongscriptionSegment[];
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
      dirty: false,
    };
  }, [editSession, transcriptVersion, data?.transcript?.segments]);

  const draftSegments = currentSession.segments;
  const dirty = currentSession.dirty;

  const [saving, setSaving] = useState(false);
  const [editingSegId, setEditingSegId] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");

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
      dirty: true,
    });
  };

  const handleImportFile = async (file?: File | null) => {
    if (!file) return;
    try {
      const raw = await file.text();
      setUserLyricsText(stripImportedLyricsFormat(raw));
      setMode("align");
      setSetupOpen(true);
      setError(null);
    } catch {
      setError("Không thể đọc tệp lời bài hát.");
    }
  };

  const runExtraction = async () => {
    if (submitting) return;
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "align") {
        if (!userLyricsText.trim()) {
          setError("Vui lòng dán hoặc nhập tệp lời bài hát để căn chỉnh.");
          setSubmitting(false);
          return;
        }
        const res = await api.alignProjectLyrics(pid, {
          lyrics_text: userLyricsText.trim(),
          language,
          model: selectedModelId,
          isolate_vocals: isolateVocals && vocalAvailable,
        });
        qc.setQueryData(["project-lyrics", pid], res);
      } else {
        const res = await api.extractProjectLyrics(pid, {
          language,
          model: selectedModelId,
          isolate_vocals: isolateVocals && vocalAvailable,
          force_separate: false,
        });
        qc.setQueryData(["project-lyrics", pid], res);
      }
      setSetupOpen(false);
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
    setMoreOpen(false);
    setSubmitting(true);
    setError(null);
    try {
      const res = await api.alignProjectLyrics(pid, {
        lyrics_text: payload,
        language: data?.language || language || "auto",
        model: normalizeSongscriptionModelId(data?.model || selectedModelId),
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

  const highlightTime = player.playing ? currentTime + KARAOKE_SYNC_LEAD_SEC : currentTime;

  const karaokeWordsBySegId = useMemo(() => {
    const map = new Map<number, SongscriptionSegment["words"]>();
    for (let i = 0; i < draftSegments.length; i++) {
      const seg = draftSegments[i];
      const nextStart = i + 1 < draftSegments.length ? draftSegments[i + 1].start : null;
      map.set(seg.id, prepareWordsForKaraoke(seg.words, seg.start, seg.end, nextStart));
    }
    return map;
  }, [draftSegments]);

  const activeSegmentId = useMemo(() => {
    for (let i = draftSegments.length - 1; i >= 0; i--) {
      const seg = draftSegments[i];
      const words = karaokeWordsBySegId.get(seg.id) ?? [];
      const effStart = words.length > 0 ? Math.min(seg.start, words[0].start) : seg.start;
      const effEnd = words.length > 0 ? Math.max(seg.end, words[words.length - 1].end) : seg.end;
      if (highlightTime >= effStart && highlightTime < effEnd) {
        return seg.id;
      }
    }
    return null;
  }, [draftSegments, karaokeWordsBySegId, highlightTime]);

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

  const deleteLineById = (id: number) => {
    if (draftSegments.length <= 1) return;
    const next = draftSegments.filter(s => s.id !== id);
    applyDraftChange(next);
    setEditingSegId(null);
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
      setExportOpen(false);
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
        <p>AI Lyrics cần tệp âm thanh gốc có giọng hát để trích xuất và đồng bộ lời bài hát.</p>
      </div>
    );
  }

  const isRunning = data?.state === "queued" || data?.state === "running";
  const stageIdx = resolveStageIndex(data?.message, data?.progress);
  const isCoverSolo = player.mix >= 0.95;

  return (
    <div className="project-lyrics-card pl-clean">
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
              <h2>Đang trích xuất lời cho {project.title}</h2>
              <p>
                {data?.message
                  ? STAGE_LABELS[data.message] ?? data.message
                  : "Đang chờ xử lý…"}
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
        </div>
      ) : data?.state === "error" ? (
        <div className="songscription-empty songscription-error-box" role="alert">
          <h2>Không thể trích xuất lời</h2>
          <p>{data.error}</p>
          <button type="button" className="btn primary" onClick={() => void runExtraction()}>
            ↻ Thử lại
          </button>
        </div>
      ) : !data?.transcript ? (
        <div className="songscription-empty">
          <div className="songscription-empty-badge">✦</div>
          <h2>Trích xuất lời bài hát</h2>
          <p>AI sẽ tách giọng hát từ <code>{project.audio}</code> và đồng bộ từng câu theo nhạc.</p>
          <div className="pl-empty-actions">
            <button
              type="button"
              className="btn primary"
              disabled={submitting}
              onClick={() => void runExtraction()}
            >
              ✦ Trích xuất tự động
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setMode("align");
                setSetupOpen(v => !v);
              }}
            >
              Dán lời có sẵn…
            </button>
          </div>
          {setupOpen && (
            <div className="pl-setup-drawer pl-setup-centered">
              <textarea
                rows={5}
                value={userLyricsText}
                onChange={e => setUserLyricsText(e.target.value)}
                placeholder="Dán lời bài hát chuẩn vào đây (mỗi câu 1 dòng)…"
                aria-label="Lời bài hát để đồng bộ"
              />
              <div className="pl-setup-foot">
                <button type="button" className="btn small" onClick={() => lyricsFileRef.current?.click()}>
                  Chọn tệp .TXT / .LRC
                </button>
                <button type="button" className="btn primary small" disabled={submitting} onClick={() => void runExtraction()}>
                  Đồng bộ lời với nhạc
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="songscription-viewer pl-viewer">
          {/* ONE Single Clean Toolbar */}
          <div className="pl-bar" role="toolbar" aria-label="Công cụ lời bài hát">
            <div className="pl-bar-group">
              <div className="pl-tabs" role="tablist">
                <button
                  type="button"
                  role="tab"
                  aria-selected={viewMode === "synced"}
                  className={viewMode === "synced" ? "on" : ""}
                  onClick={() => setViewMode("synced")}
                >
                  Lời hát
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={viewMode === "plain"}
                  className={viewMode === "plain" ? "on" : ""}
                  onClick={() => setViewMode("plain")}
                >
                  Văn bản
                </button>
              </div>

              {viewMode === "synced" && (
                <input
                  className="songscription-search pl-search"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Tìm câu hát…"
                  aria-label="Tìm câu hát"
                />
              )}

              {uncertainCount > 0 && viewMode === "synced" && (
                <button
                  type="button"
                  className={`pl-flag-pill ${onlyUncertain ? "on" : ""}`}
                  onClick={() => setOnlyUncertain(v => !v)}
                >
                  Cần xem ({uncertainCount})
                </button>
              )}
            </div>

            <div className="pl-bar-group">
              {isCoverSolo && (
                <button
                  type="button"
                  className="btn small"
                  onClick={() => player.setMix(0.35)}
                  title="Bật thêm âm thanh bản gốc để nghe giọng hát"
                >
                  Bật tiếng hát gốc
                </button>
              )}

              {dirty && (
                <button
                  type="button"
                  className="btn primary small"
                  disabled={saving}
                  onClick={() => void saveEdits()}
                >
                  {saving ? "Đang lưu…" : "Lưu chỉnh sửa"}
                </button>
              )}

              {/* Single Export Dropdown */}
              <div className="studio-menu-wrap" ref={exportRef}>
                <button
                  type="button"
                  className="btn small"
                  aria-expanded={exportOpen}
                  onClick={() => {
                    setExportOpen(v => !v);
                    setMoreOpen(false);
                  }}
                >
                  {copied ? "✓ Đã chép" : "Xuất lời ▾"}
                </button>
                {exportOpen && (
                  <div className="studio-menu-dropdown" role="menu">
                    <button type="button" role="menuitem" onClick={() => void copyScript()}>
                      Sao chép toàn bộ lời
                    </button>
                    <button type="button" role="menuitem" onClick={() => { exporters.exportTxt(true); setExportOpen(false); }}>
                      Tải TXT (Văn bản)
                    </button>
                    <button type="button" role="menuitem" onClick={() => { exporters.exportLrc(); setExportOpen(false); }}>
                      Tải LRC (Đồng bộ câu)
                    </button>
                    {exporters.hasWordTimings && (
                      <button type="button" role="menuitem" onClick={() => { exporters.exportEnhancedLrc(); setExportOpen(false); }}>
                        Tải Enhanced LRC (Từng từ)
                      </button>
                    )}
                    <button type="button" role="menuitem" onClick={() => { exporters.exportSrt(); setExportOpen(false); }}>
                      Tải SRT (Phụ đề)
                    </button>
                    <button type="button" role="menuitem" onClick={() => { exporters.exportJson({ project_id: pid }); setExportOpen(false); }}>
                      Tải JSON
                    </button>
                  </div>
                )}
              </div>

              {/* Single Overflow Menu for Rare Actions */}
              <div className="studio-menu-wrap" ref={moreRef}>
                <button
                  type="button"
                  className="studio-menu-btn"
                  aria-label="Tác vụ khác"
                  aria-expanded={moreOpen}
                  onClick={() => {
                    setMoreOpen(v => !v);
                    setExportOpen(false);
                  }}
                >
                  ⋯
                </button>
                {moreOpen && (
                  <div className="studio-menu-dropdown" role="menu">
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setAutoFollow(v => !v);
                        setMoreOpen(false);
                      }}
                    >
                      {autoFollow ? "✓ Tự cuộn theo nhạc" : "Tự cuộn theo nhạc"}
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      disabled={submitting}
                      onClick={() => void reAlignEditedScript()}
                    >
                      ⇄ Đồng bộ lại dấu thời gian
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setMoreOpen(false);
                        setSetupOpen(v => !v);
                      }}
                    >
                      ↻ Trích xuất / Căn chỉnh lại…
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Optional Collapsible Re-Extract / Align Drawer (hidden by default!) */}
          {setupOpen && (
            <div className="pl-setup-drawer">
              <div className="pl-setup-row">
                <div className="pl-tabs" role="tablist">
                  <button
                    type="button"
                    className={mode === "extract" ? "on" : ""}
                    onClick={() => setMode("extract")}
                  >
                    AI trích xuất lại
                  </button>
                  <button
                    type="button"
                    className={mode === "align" ? "on" : ""}
                    onClick={() => setMode("align")}
                  >
                    Căn chỉnh lời có sẵn
                  </button>
                </div>

                <select
                  className="project-lyrics-select"
                  value={language}
                  onChange={e => setLanguage(e.target.value)}
                  aria-label="Ngôn ngữ"
                >
                  {LANGUAGE_OPTIONS.map(([code, label]) => (
                    <option key={code} value={code}>
                      {label}
                    </option>
                  ))}
                </select>

                <select
                  className="project-lyrics-select"
                  value={selectedModelId}
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
                  <span>Tách vocal</span>
                </label>
              </div>

              {mode === "align" && (
                <textarea
                  rows={4}
                  value={userLyricsText}
                  onChange={e => setUserLyricsText(e.target.value)}
                  placeholder="Dán lời bài hát chuẩn vào đây (mỗi câu một dòng)…"
                  aria-label="Lời bài hát để đồng bộ"
                />
              )}

              <div className="pl-setup-foot">
                {mode === "align" && (
                  <button type="button" className="btn small" onClick={() => lyricsFileRef.current?.click()}>
                    Nhập .TXT / .LRC
                  </button>
                )}
                <div className="grow" />
                <button type="button" className="btn small" onClick={() => setSetupOpen(false)}>
                  Đóng
                </button>
                <button
                  type="button"
                  className="btn primary small"
                  disabled={submitting}
                  onClick={() => void runExtraction()}
                >
                  {mode === "align" ? "Đồng bộ lời" : "Chạy trích xuất"}
                </button>
              </div>
            </div>
          )}

          {/* Clean Scrollable Lyrics Area */}
          <div
            className="songscription-body"
            ref={listScrollRef}
            onWheel={() => {
              if (player.playing && autoFollow) setAutoFollow(false);
            }}
          >
            {!draftSegments.length ? (
              <div className="songscription-no-lyrics">Không phát hiện thấy câu hát trong bản thu này.</div>
            ) : viewMode === "plain" ? (
              <div className="songscription-plain-text">
                {draftSegments.map(s => (s.section ? `[${s.section}]\n${s.text}` : s.text)).join("\n")}
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
                          {seg.chord && (
                            <div className="songscription-line-tags">
                              <span className="line-chord-tag">{seg.chord}</span>
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
                              <button
                                type="button"
                                className="danger"
                                disabled={draftSegments.length <= 1}
                                onClick={() => deleteLineById(seg.id)}
                              >
                                Xóa
                              </button>
                              <button type="button" onClick={() => setEditingSegId(null)}>
                                Hủy
                              </button>
                            </div>
                          ) : (
                            <p className="songscription-line-text" onClick={() => seekAndPlay(seg.start)}>
                              {(() => {
                                const words = karaokeWordsBySegId.get(seg.id) ?? seg.words ?? [];
                                if (words.length === 0) return seg.text;
                                return words.map((w, wi) => {
                                  const nextStart = wi + 1 < words.length ? words[wi + 1].start : null;
                                  const st = getWordPlaybackState(w, highlightTime, nextStart);
                                  return (
                                    <span
                                      key={wi}
                                      className={`${st === "active" ? "word-active" : st === "sung" ? "word-sung" : ""} ${
                                        w.uncertain ? "word-uncertain" : ""
                                      }`}
                                    >
                                      {w.word}
                                    </span>
                                  );
                                });
                              })()}
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
