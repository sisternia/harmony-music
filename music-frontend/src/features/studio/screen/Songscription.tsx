import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import Icon from "../../../components/ui/Icon";
import VolumeControl from "../../../components/ui/VolumeControl";
import { colors } from "../../../theme/colors";
import {
  api,
  urls,
  type ProjectSummary,
  type RetranscribeLineResult,
  type ReviewStatus,
  type SongscriptionItem,
  type SongscriptionSegment,
} from "../api";
import { SongscriptionCreateModal } from "../components/SongscriptionCreateModal";
import { SongscriptionLibraryView } from "../components/SongscriptionLibraryView";
import { getPlaybackPercent, getPlaybackPosition, getWordPlaybackState } from "../utils/lyricsPlayback";
import {
  PIPELINE_STAGES,
  SECTION_OPTIONS,
  STAGE_LABELS,
  computeAvgConfidence,
  computeReviewStats,
  exportTranscriptFiles,
  formatClock,
  formatPreciseClock,
  getEffectiveReviewStatus,
  getReviewReason,
  isLineFlaggedForReview,
  resolveStageIndex,
} from "../utils/lyricsUtils";

const EMPTY_ITEMS: SongscriptionItem[] = [];

type ReviewFilter = "all" | "needs_review" | "verified" | "edited";

function parseSongscriptionRoute(path: string): { mode: "library" | "new" | "detail"; sessionId: string | null } {
  const clean = path.replace(/\/+$/, "") || "/songscription";
  if (clean === "/songscription") return { mode: "library", sessionId: null };
  if (clean === "/songscription/new") return { mode: "new", sessionId: null };
  const m = clean.match(/^\/songscription\/([^/]+)$/);
  if (m) return { mode: "detail", sessionId: decodeURIComponent(m[1]) };
  return { mode: "library", sessionId: null };
}

export function Songscription({
  path,
  navigate,
  projects,
  youtubeSupported,
}: {
  path: string;
  navigate: (to: string) => void;
  projects: ProjectSummary[];
  youtubeSupported: boolean;
}) {
  const qc = useQueryClient();
  const route = useMemo(() => parseSongscriptionRoute(path), [path]);
  const activeId = route.mode === "detail" ? route.sessionId : null;

  const listQuery = useQuery({
    queryKey: ["songscriptions"],
    queryFn: api.songscriptions,
    refetchInterval: q =>
      q.state.data?.items.some(item => item.state === "queued" || item.state === "running") ? 1200 : false,
  });

  const items = listQuery.data?.items ?? EMPTY_ITEMS;
  const models = listQuery.data?.models ?? [
    { id: "turbo", label: "Whisper Large-v3 Turbo (Khuyên dùng · Nhanh & 4GB VRAM)" },
    { id: "large-v3", label: "Whisper Large-v3 (Độ chính xác tối đa)" },
    { id: "medium", label: "Whisper Medium (Nhẹ VRAM)" },
    { id: "small", label: "Whisper Small (Siêu nhanh / CPU)" },
  ];
  const vocalAvailable = listQuery.data?.vocal_isolation_available ?? true;

  const detailQuery = useQuery({
    queryKey: ["songscription", activeId],
    queryFn: () => api.songscription(activeId!),
    enabled: Boolean(activeId),
    retry: false,
    refetchInterval: q => {
      const st = q.state.data?.state;
      return st === "queued" || st === "running" ? 1000 : false;
    },
  });

  const activeItem: SongscriptionItem | null = activeId
    ? (detailQuery.data ?? items.find(i => i.id === activeId) ?? null)
    : null;
  const isRunning = activeItem?.state === "queued" || activeItem?.state === "running";
  const isDone = activeItem?.state === "done";

  // Create modal and workspace menus
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const isCreateOpen = createModalOpen || route.mode === "new";
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const [formError, setFormError] = useState("");

  // Review Queue & Display states (Stage C)
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");
  const [viewMode, setViewMode] = useState<"review" | "plain">("review");
  const [searchQuery, setSearchQuery] = useState("");
  const [showChords, setShowChords] = useState(true);
  const [showWordHighlight, setShowWordHighlight] = useState(true);
  const [autoFollow, setAutoFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  // Selected / Expanded line for inline editing
  const [selectedSegId, setSelectedSegId] = useState<number | null>(null);
  const [loopSelectedLine, setLoopSelectedLine] = useState(false);
  const [showWordEditor, setShowWordEditor] = useState(false);

  // Precision AI Re-transcribe state for flagged ("Cần kiểm tra") lines
  const [retranscribingTarget, setRetranscribingTarget] = useState<"all" | number | null>(null);
  const [aiRefineResults, setAiRefineResults] = useState<RetranscribeLineResult[]>([]);

  // Versioned edit session (zero setState-in-effect)
  const transcriptVersion = `${activeItem?.id ?? ""}:${activeItem?.completed ?? ""}:${activeItem?.segment_count ?? 0}`;
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
    const base = (activeItem?.transcript?.segments ?? []).map(s => ({ ...s }));
    return {
      version: transcriptVersion,
      segments: base,
      undo: [] as SongscriptionSegment[][],
      redo: [] as SongscriptionSegment[][],
      dirty: false,
    };
  }, [editSession, transcriptVersion, activeItem?.transcript?.segments]);

  const draftSegments = currentSession.segments;
  const undoStack = currentSession.undo;
  const redoStack = currentSession.redo;
  const dirty = currentSession.dirty;

  const [savingAll, setSavingAll] = useState(false);
  const [realigning, setRealigning] = useState(false);

  // Authoritative audio element & refs
  const audioRef = useRef<HTMLAudioElement>(null);
  const listScrollRef = useRef<HTMLDivElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const moreMenuRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const duration = Number.isFinite(audioDuration) && audioDuration > 0
    ? audioDuration
    : activeItem?.duration ?? 0;
  const position = getPlaybackPosition(currentTime, duration);
  const progressPercent = getPlaybackPercent(position, duration);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = volume;
    audio.muted = muted;
    audio.playbackRate = playbackRate;
  }, [activeId, isDone, volume, muted, playbackRate]);

  const flashToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => {
      setToastMsg(cur => (cur === msg ? null : cur));
    }, 3200);
  };

  const openSession = (sid: string) => {
    if (dirty && activeId && sid !== activeId) {
      if (!confirm("Bạn có thay đổi lời bài hát chưa lưu. Rời khỏi phiên này mà không lưu?")) return;
    }
    setSelectedSegId(null);
    setAiRefineResults([]);
    setCurrentTime(0);
    setAudioDuration(0);
    setPlaying(false);
    setLoopSelectedLine(false);
    setCreateModalOpen(false);
    navigate(`/songscription/${encodeURIComponent(sid)}`);
  };

  const backToLibrary = () => {
    if (dirty && activeId) {
      if (!confirm("Bạn có thay đổi lời bài hát chưa lưu. Quay lại thư viện Songscription mà không lưu?")) return;
    }
    setSelectedSegId(null);
    setPlaying(false);
    setLoopSelectedLine(false);
    setCreateModalOpen(false);
    navigate("/songscription");
  };

  const closeCreateModal = () => {
    setCreateModalOpen(false);
    if (route.mode === "new") {
      navigate("/songscription");
    }
  };

  // Close dropdown menus on outside click or Escape
  useEffect(() => {
    if (!exportMenuOpen && !moreMenuOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (exportMenuOpen && exportMenuRef.current && !exportMenuRef.current.contains(target)) {
        setExportMenuOpen(false);
      }
      if (moreMenuOpen && moreMenuRef.current && !moreMenuRef.current.contains(target)) {
        setMoreMenuOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setExportMenuOpen(false);
        setMoreMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [exportMenuOpen, moreMenuOpen]);

  // Warn browser tab close when there are unsaved lyric edits
  useEffect(() => {
    if (!dirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

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

  const reviewStats = useMemo(() => computeReviewStats(draftSegments), [draftSegments]);
  const avgConfidence = useMemo(() => computeAvgConfidence(draftSegments), [draftSegments]);
  const hasAnyChords = useMemo(() => draftSegments.some(s => Boolean(s.chord)), [draftSegments]);
  const hasWordTimings = useMemo(
    () => draftSegments.some(s => Boolean(s.words && s.words.length > 0)),
    [draftSegments],
  );
  const hasUserCorrections = useMemo(
    () =>
      dirty ||
      draftSegments.some(s => s.review_status === "edited" || s.review_status === "verified"),
    [dirty, draftSegments],
  );

  const filteredSegments = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return draftSegments.filter(seg => {
      const st = getEffectiveReviewStatus(seg);
      if (reviewFilter === "needs_review" && !isLineFlaggedForReview(seg)) return false;
      if (reviewFilter === "verified" && st !== "verified") return false;
      if (reviewFilter === "edited" && st !== "edited") return false;
      if (!q) return true;
      return (
        seg.text.toLowerCase().includes(q) ||
        Boolean(seg.chord && seg.chord.toLowerCase().includes(q)) ||
        Boolean(seg.section && seg.section.toLowerCase().includes(q))
      );
    });
  }, [draftSegments, reviewFilter, searchQuery]);

  const selectedSegment = useMemo(
    () => draftSegments.find(s => s.id === selectedSegId) ?? null,
    [draftSegments, selectedSegId],
  );

  const playingSegmentId = useMemo(() => {
    for (let i = draftSegments.length - 1; i >= 0; i--) {
      const seg = draftSegments[i];
      if (currentTime >= seg.start && currentTime < seg.end) {
        return seg.id;
      }
    }
    return null;
  }, [draftSegments, currentTime]);

  // Auto-loop selected segment on the single authoritative <audio> timeline
  const handleAudioTimeUpdate = useCallback((el: HTMLAudioElement) => {
    const t = el.currentTime;
    if (loopSelectedLine && selectedSegment && !el.paused) {
      const endBound = Math.max(selectedSegment.start + 0.2, selectedSegment.end + 0.08);
      if (t >= endBound || t < selectedSegment.start - 0.25) {
        el.currentTime = Math.max(0, selectedSegment.start);
        setCurrentTime(el.currentTime);
        return;
      }
    }
    setCurrentTime(t);
  }, [loopSelectedLine, selectedSegment]);

  // timeupdate is too infrequent for short sung words. Use the audio clock
  // at most 30 times per second while playing; seeks still update immediately.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let lastPaint = -Infinity;
    const syncTime = (now: number) => {
      const audio = audioRef.current;
      if (audio && now - lastPaint >= 1000 / 30) {
        handleAudioTimeUpdate(audio);
        lastPaint = now;
      }
      frame = requestAnimationFrame(syncTime);
    };
    frame = requestAnimationFrame(syncTime);
    return () => cancelAnimationFrame(frame);
  }, [playing, handleAudioTimeUpdate]);

  // Auto-follow active playing line
  useEffect(() => {
    if (!playing || !autoFollow || playingSegmentId === null) return;
    const container = listScrollRef.current;
    const row = container?.querySelector<HTMLElement>(`[data-seg-id="${playingSegmentId}"]`);
    if (!container || !row) return;
    const rowRect = row.getBoundingClientRect();
    const boxRect = container.getBoundingClientRect();
    if (rowRect.top < boxRect.top + 36 || rowRect.bottom > boxRect.bottom - 48) {
      container.scrollTo({
        top: Math.max(0, container.scrollTop + (rowRect.top - boxRect.top) - boxRect.height * 0.3),
        behavior: "smooth",
      });
    }
  }, [playingSegmentId, playing, autoFollow]);

  const scrollRowIntoView = (segId: number) => {
    requestAnimationFrame(() => {
      const container = listScrollRef.current;
      const row = container?.querySelector<HTMLElement>(`[data-seg-id="${segId}"]`);
      if (!container || !row) return;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  const seekAudio = (sec: number, autoPlay = false) => {
    const el = audioRef.current;
    if (!el) return;
    const end = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : duration;
    el.currentTime = getPlaybackPosition(sec, end);
    setCurrentTime(el.currentTime);
    if (autoPlay && el.paused) {
      void el.play().catch(() => {});
    }
  };

  const changeVolume = (value: number) => {
    const next = Math.min(1, Math.max(0, value));
    setVolume(next);
    setMuted(next === 0);
    const audio = audioRef.current;
    if (audio) {
      audio.volume = next;
      audio.muted = next === 0;
    }
  };

  const selectAndSeekLine = (seg: SongscriptionSegment, autoPlay = false) => {
    setSelectedSegId(seg.id);
    setShowWordEditor(false);
    seekAudio(seg.start, autoPlay);
    scrollRowIntoView(seg.id);
  };

  const replaySelectedLine = () => {
    if (!selectedSegment) return;
    seekAudio(selectedSegment.start, true);
  };

  const navigateRelativeLine = (delta: -1 | 1, onlyFlagged = false) => {
    const pool = onlyFlagged
      ? draftSegments.filter(isLineFlaggedForReview)
      : filteredSegments.length
        ? filteredSegments
        : draftSegments;
    if (!pool.length) {
      if (onlyFlagged) flashToast("Tất cả các câu hát đều đã được kiểm tra!");
      return;
    }
    const curIdx = pool.findIndex(s => s.id === selectedSegId);
    let target: SongscriptionSegment;
    if (curIdx === -1) {
      target = delta > 0 ? pool[0] : pool[pool.length - 1];
    } else {
      const nextIdx = (curIdx + delta + pool.length) % pool.length;
      target = pool[nextIdx];
    }
    selectAndSeekLine(target, true);
  };

  const saveAllDraftSegments = async (segmentsToSave = draftSegments, toastText = "Đã lưu thay đổi lời bài hát") => {
    if (!activeItem || savingAll) return;
    setSavingAll(true);
    setFormError("");
    try {
      const saved = await api.updateSongscription(activeItem.id, { segments: segmentsToSave });
      qc.setQueryData(["songscription", activeItem.id], saved);
      void qc.invalidateQueries({ queryKey: ["songscriptions"] });
      setEditSession(null);
      flashToast(toastText);
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSavingAll(false);
    }
  };

  const markLineReviewStatus = async (segId: number, status: ReviewStatus, advanceToNextFlagged = false) => {
    const next = draftSegments.map(s =>
      s.id === segId
        ? {
            ...s,
            review_status: status,
            low_confidence: status === "needs_review" || status === "skipped",
            words:
              status === "verified" || status === "edited"
                ? (s.words ?? []).map(w => ({ ...w, uncertain: false }))
                : s.words,
          }
        : s,
    );
    applyDraftChange(next);
    await saveAllDraftSegments(
      next,
      status === "verified"
        ? "Đã xác nhận câu hát"
        : status === "skipped"
          ? "Đã đánh dấu xem lại sau"
          : "Đã cập nhật trạng thái",
    );
    if (advanceToNextFlagged) {
      const remaining = next.filter(s => s.id !== segId && isLineFlaggedForReview(s));
      const afterCurrent = remaining.find(s => s.id > segId) ?? remaining[0];
      if (afterCurrent) {
        selectAndSeekLine(afterCurrent, true);
      } else {
        flashToast("Hoàn tất! Không còn câu nào cần kiểm tra.");
      }
    }
  };

  // Keyboard shortcuts for efficient review workflow (when not typing inside form fields)
  useEffect(() => {
    if (!isDone || viewMode !== "review") return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable);

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (dirty) void saveAllDraftSegments();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !isTyping) {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (isTyping) return;

      if (e.code === "Space") {
        // Let focused buttons handle Space themselves, including volume.
        if (target?.closest('button, a, [role="button"]')) return;
        e.preventDefault();
        const el = audioRef.current;
        if (!el) return;
        if (el.paused) void el.play().catch(() => {});
        else el.pause();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        replaySelectedLine();
      } else if (e.key === "l" || e.key === "L") {
        e.preventDefault();
        if (selectedSegment) setLoopSelectedLine(v => !v);
      } else if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        navigateRelativeLine(1, true);
      } else if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        navigateRelativeLine(1, false);
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        navigateRelativeLine(-1, false);
      } else if ((e.key === "v" || e.key === "V") && selectedSegment) {
        e.preventDefault();
        void markLineReviewStatus(selectedSegment.id, "verified", true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDone, viewMode, dirty, selectedSegment, draftSegments, filteredSegments, loopSelectedLine]);

  const handleCreated = async (created: SongscriptionItem) => {
    setCreateModalOpen(false);
    setSelectedSegId(null);
    setEditSession(null);
    qc.setQueryData(["songscription", created.id], created);
    await qc.invalidateQueries({ queryKey: ["songscriptions"] });
    navigate(`/songscription/${encodeURIComponent(created.id)}`);
    flashToast(`Đã tạo phiên "${created.title}"`);
  };

  const handleRenameSession = async (item: SongscriptionItem, newTitle: string) => {
    const updated = await api.updateSongscription(item.id, { title: newTitle });
    qc.setQueryData(["songscription", item.id], updated);
    await qc.invalidateQueries({ queryKey: ["songscriptions"] });
    flashToast("Đã đổi tên phiên Songscription");
  };

  const handleRetry = async (sid: string, confirmOverwrite = false) => {
    if (
      confirmOverwrite &&
      hasUserCorrections &&
      !confirm(
        "Bạn đã chỉnh sửa hoặc xác nhận lời bài hát này. Chạy lại AI trích xuất sẽ thay thế bản lời hiện tại — bạn có chắc chắn muốn tiếp tục không?",
      )
    ) {
      return;
    }
    setMoreMenuOpen(false);
    setFormError("");
    try {
      const updated = await api.retrySongscription(sid);
      qc.setQueryData(["songscription", sid], updated);
      setEditSession(null);
      setSelectedSegId(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const handleRetranscribeFlagged = async (segmentIds?: number[]) => {
    if (!activeItem || retranscribingTarget !== null) return;
    setFormError("");
    const targetKey: "all" | number = segmentIds && segmentIds.length === 1 ? segmentIds[0] : "all";
    setRetranscribingTarget(targetKey);
    try {
      // Save any pending manual edits first so backend operates on latest timestamps/text
      if (dirty) {
        await api.updateSongscription(activeItem.id, { segments: draftSegments });
      }
      const res = await api.retranscribeSongscriptionFlagged(activeItem.id, {
        segment_ids: segmentIds && segmentIds.length > 0 ? segmentIds : undefined,
      });
      qc.setQueryData(["songscription", activeItem.id], res.item);
      setEditSession(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      setAiRefineResults(res.results);
      if (res.retranscribed_count === 0) {
        flashToast("Không còn câu nào cần kiểm tra để đọc lại.");
      } else if (res.improved_count > 0) {
        flashToast(
          `AI đã đọc lại ${res.retranscribed_count} câu · Cải thiện ${res.improved_count} câu (${res.auto_resolved_count} câu đạt độ tin cậy cao)`,
        );
      } else {
        flashToast(`AI đã đọc lại ${res.retranscribed_count} câu — bản hiện tại đã tối ưu.`);
      }
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRetranscribingTarget(null);
    }
  };

  const undoAiRefineLine = async (res: RetranscribeLineResult) => {
    if (!activeItem) return;
    const next = draftSegments.map(s =>
      s.id === res.id
        ? {
            ...s,
            text: res.old_text,
            confidence: res.old_confidence,
            low_confidence: true,
            review_status: "needs_review" as ReviewStatus,
          }
        : s,
    );
    setEditSession({
      version: transcriptVersion,
      segments: next,
      undo: [...undoStack, draftSegments],
      redo: [],
      dirty: true,
    });
    setAiRefineResults(prev => prev.filter(r => r.id !== res.id));
    try {
      const updated = await api.updateSongscription(activeItem.id, { segments: next });
      qc.setQueryData(["songscription", activeItem.id], updated);
      setEditSession(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      flashToast(`Đã hoàn tác câu #${res.id} về lời cũ`);
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const handleRealignFromEditor = async () => {
    if (!activeItem || realigning) return;
    if (
      dirty &&
      !confirm(
        "Đồng bộ lại sẽ giữ nguyên câu chữ hiện tại trong trình sửa và tính lại dấu thời gian theo giọng hát. Tiếp tục?",
      )
    ) {
      return;
    }
    const textLines: string[] = [];
    for (const seg of draftSegments) {
      if (seg.section) textLines.push(`[${seg.section}]`);
      if (seg.text.trim()) textLines.push(seg.text.trim());
    }
    const textPayload = textLines.join("\n").trim();
    if (!textPayload) {
      setFormError("Không có dòng lời nào để căn chỉnh lại.");
      return;
    }
    setMoreMenuOpen(false);
    setRealigning(true);
    setFormError("");
    try {
      const updated = await api.alignSongscription(activeItem.id, {
        lyrics_text: textPayload,
        language: activeItem.language || "auto",
        model: activeItem.model || "turbo",
        isolate_vocals: activeItem.isolate_vocals,
      });
      qc.setQueryData(["songscription", activeItem.id], updated);
      setEditSession(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRealigning(false);
    }
  };

  const handleDelete = async (item: SongscriptionItem) => {
    if (!confirm(`Xóa bản Songscription "${item.title}"? Hành động này không thể hoàn tác.`)) return;
    try {
      await api.deleteSongscription(item.id);
      setSelectedSegId(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      flashToast(`Đã xóa "${item.title}"`);
      if (activeId === item.id) {
        navigate("/songscription");
      }
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  // Inline line edit helpers
  const updateLineField = (id: number, patch: Partial<SongscriptionSegment>, markEdited = true) => {
    const next = draftSegments.map(s =>
      s.id === id
        ? {
            ...s,
            ...patch,
            review_status: markEdited ? ("edited" as ReviewStatus) : s.review_status,
            low_confidence: markEdited ? false : s.low_confidence,
          }
        : s,
    );
    applyDraftChange(next);
  };

  const updateWordTimestamp = (segId: number, wordIdx: number, field: "start" | "end", value: number) => {
    const next = draftSegments.map(s => {
      if (s.id !== segId || !s.words) return s;
      const updatedWords = s.words.map((w, wi) =>
        wi === wordIdx ? { ...w, [field]: Math.max(0, Number(value.toFixed(2))) } : w,
      );
      return { ...s, words: updatedWords, review_status: "edited" as ReviewStatus, low_confidence: false };
    });
    applyDraftChange(next);
  };

  const insertLineAfter = (idx: number) => {
    const prev = draftSegments[idx];
    const nextSeg = draftSegments[idx + 1];
    const start = prev ? Number((prev.end + 0.1).toFixed(2)) : 0;
    const end = nextSeg
      ? Number(Math.max(start + 1.2, nextSeg.start - 0.1).toFixed(2))
      : Number((start + 2.5).toFixed(2));
    const inserted: SongscriptionSegment = {
      id: 0,
      start,
      end,
      text: "",
      lang: prev?.lang || activeItem?.language || "en",
      section: null,
      review_status: "edited",
      low_confidence: false,
      words: [],
    };
    const next = [...draftSegments.slice(0, idx + 1), inserted, ...draftSegments.slice(idx + 1)];
    applyDraftChange(next);
    setSelectedSegId(idx + 2);
  };

  const deleteLineAt = (idx: number) => {
    if (draftSegments.length <= 1) return;
    if (!confirm("Xóa dòng lời bài hát này?")) return;
    const next = draftSegments.filter((_, i) => i !== idx);
    applyDraftChange(next);
    setSelectedSegId(null);
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
    const firstSeg: SongscriptionSegment = {
      ...seg,
      end: Math.max(Number((seg.start + 0.1).toFixed(2)), midTime),
      text: firstText,
      review_status: "edited",
      low_confidence: false,
      words: [],
    };
    const secondSeg: SongscriptionSegment = {
      ...seg,
      start: midTime,
      text: secondText,
      section: null,
      review_status: "edited",
      low_confidence: false,
      words: [],
    };
    const next = [...draftSegments.slice(0, idx), firstSeg, secondSeg, ...draftSegments.slice(idx + 1)];
    applyDraftChange(next);
  };

  const mergeWithNext = (idx: number) => {
    if (idx < 0 || idx >= draftSegments.length - 1) return;
    const cur = draftSegments[idx];
    const nxt = draftSegments[idx + 1];
    const merged: SongscriptionSegment = {
      ...cur,
      end: Math.max(cur.end, nxt.end),
      text: `${cur.text.trim()} ${nxt.text.trim()}`.trim(),
      review_status: "edited",
      low_confidence: false,
      words: [...(cur.words ?? []), ...(nxt.words ?? [])],
    };
    const next = [...draftSegments.slice(0, idx), merged, ...draftSegments.slice(idx + 2)];
    applyDraftChange(next);
  };

  const copyScript = async () => {
    const text = draftSegments.map(s => (s.section ? `[${s.section}]\n${s.text}` : s.text)).join("\n");
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setMoreMenuOpen(false);
      flashToast("Đã sao chép toàn bộ lời bài hát");
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard blocked
    }
  };

  const exporters = useMemo(
    () =>
      exportTranscriptFiles(
        activeItem?.title || "songscription",
        activeItem?.transcript?.language || activeItem?.language || "en",
        activeItem?.transcript?.language_name || activeItem?.language_name || "Auto",
        draftSegments,
      ),
    [
      activeItem?.title,
      activeItem?.transcript?.language,
      activeItem?.transcript?.language_name,
      activeItem?.language,
      activeItem?.language_name,
      draftSegments,
    ],
  );

  const stageIdx = resolveStageIndex(activeItem?.message, activeItem?.progress);
  const stageLabel = activeItem?.message
    ? STAGE_LABELS[activeItem.message] ?? activeItem.message
    : activeItem?.state === "queued"
      ? "Đang chờ trong hàng đợi GPU…"
      : "Đang xử lý…";

  // State A: Library view (default at /songscription, and also behind modal at /songscription/new)
  if (route.mode !== "detail") {
    return (
      <>
        {toastMsg && (
          <div className="ss-toast" role="status" aria-live="polite">
            {toastMsg}
          </div>
        )}
        <SongscriptionLibraryView
          items={items}
          projects={projects}
          isLoading={listQuery.isPending}
          isError={listQuery.isError}
          errorMessage={listQuery.error instanceof Error ? listQuery.error.message : undefined}
          onRetryFetch={() => void listQuery.refetch()}
          onCreateNew={() => setCreateModalOpen(true)}
          onOpenSession={openSession}
          onRenameSession={handleRenameSession}
          onRetrySession={item => handleRetry(item.id, false)}
          onDeleteSession={handleDelete}
        />
        <SongscriptionCreateModal
          open={isCreateOpen}
          projects={projects}
          models={models}
          vocalAvailable={vocalAvailable}
          youtubeSupported={youtubeSupported}
          onClose={closeCreateModal}
          onCreated={created => void handleCreated(created)}
        />
      </>
    );
  }

  return (
    <div className="songscription-studio ss-workspace">
      {/* Compact Workspace Top Bar with Back-to-Library Navigation */}
      <header className="ss-topbar">
        <div className="ss-topbar-left">
          <button
            type="button"
            className="ss-back-btn"
            onClick={backToLibrary}
            aria-label="Quay lại thư viện Songscription"
          >
            ← Quay lại Songscription
          </button>
          <div className="ss-topbar-title">
            <span className="studio-eyebrow">HARMONY AURALIS · LYRICS WORKSPACE</span>
            <h1>
              Songscription — <em>Trích xuất &amp; rà soát lời bài hát</em>
            </h1>
          </div>
        </div>

        <div className="ss-topbar-right">
          <button
            type="button"
            className="ss-secondary-btn"
            aria-haspopup="dialog"
            onClick={() => setCreateModalOpen(true)}
          >
            ＋ Tạo Songscription
          </button>
        </div>
      </header>

      {toastMsg && (
        <div className="ss-toast" role="status" aria-live="polite">
          {toastMsg}
        </div>
      )}

      {formError && (
        <div className="upload-error ss-workspace-error" role="alert">
          {formError}
        </div>
      )}

      <SongscriptionCreateModal
        open={isCreateOpen}
        projects={projects}
        models={models}
        vocalAvailable={vocalAvailable}
        youtubeSupported={youtubeSupported}
        onClose={closeCreateModal}
        onCreated={created => void handleCreated(created)}
      />

      {/* Full-width review workspace; sessions are managed in the library. */}
      <div className="ss-shell">
        <section className="songscription-stage ss-main-stage" aria-live="polite">
          {detailQuery.isPending && !activeItem ? (
            <div className="songscription-empty">
              <span className="songscription-pulse" aria-hidden="true" />
              <h2>Đang tải phiên Songscription…</h2>
              <p>Đang nạp dữ liệu lời bài hát và trục thời gian đã lưu.</p>
            </div>
          ) : !activeItem ? (
            <div className="songscription-error-card" role="alert">
              <h2>Không tìm thấy phiên Songscription</h2>
              <p>
                {detailQuery.error instanceof Error
                  ? detailQuery.error.message
                  : "Phiên này có thể đã bị xóa hoặc đường dẫn không tồn tại."}
              </p>
              <button type="button" className="ss-secondary-btn" onClick={backToLibrary}>
                ← Quay lại thư viện Songscription
              </button>
            </div>
          ) : isRunning ? (
            /* Stage B — Process */
            <div className="songscription-running-card" aria-busy="true">
              <div className="songscription-running-head">
                <span className="songscription-pulse" aria-hidden="true" />
                <div>
                  <h2>{activeItem.title}</h2>
                  <p>{stageLabel}</p>
                </div>
                {activeItem.job && (
                  <button
                    type="button"
                    className="songscription-stop-btn"
                    onClick={async () => {
                      await api.cancelJob(activeItem.job!);
                      void qc.invalidateQueries({ queryKey: ["songscriptions"] });
                      void qc.invalidateQueries({ queryKey: ["songscription", activeItem.id] });
                    }}
                  >
                    ■ Dừng
                  </button>
                )}
              </div>

              <div className="songscription-progress-track">
                <div
                  className={`songscription-progress-fill ${activeItem.progress == null ? "indeterminate" : ""}`}
                  style={
                    activeItem.progress != null
                      ? { width: `${Math.max(4, Math.round(activeItem.progress * 100))}%` }
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

              {activeItem.log && activeItem.log.length > 0 && (
                <details className="songscription-log" open>
                  <summary>Nhật ký xử lý trực tiếp ({activeItem.log.length})</summary>
                  <pre>{activeItem.log.join("\n")}</pre>
                </details>
              )}
            </div>
          ) : activeItem.state === "error" || activeItem.state === "cancelled" ? (
            <div className="songscription-empty songscription-error-box" role="alert">
              <h2>{activeItem.state === "cancelled" ? "Tác vụ đã dừng" : "Không thể hoàn tất trích xuất lời"}</h2>
              <p>{activeItem.error || "Đã xảy ra lỗi trong quá trình xử lý âm thanh."}</p>
              <div className="songscription-error-actions">
                <button
                  type="button"
                  className="upload-submit"
                  onClick={() => void handleRetry(activeItem.id, false)}
                >
                  ↻ Thử lại ngay
                </button>
              </div>
            </div>
          ) : (
            /* Stage C — Review & Export Workspace */
            <div className="songscription-viewer ss-review-workspace">
              {/* 1. Compact Header with Explicit Confidence Breakdown & Clear Action Hierarchy */}
              <div className="ss-review-header">
                <div className="ss-review-header-info">
                  <h2>{activeItem.title}</h2>
                  <div className="ss-meta-strip" aria-label="Thông tin bản lời">
                    <span className="ss-meta-main">
                      {activeItem.transcript?.alignment_method === "word"
                        ? "Đồng bộ từng từ"
                        : "Đồng bộ theo câu"}
                      {activeItem.transcript?.language_name && ` · ${activeItem.transcript.language_name}`}
                    </span>
                    <span className="ss-meta-detail">
                      {formatClock(audioDuration || activeItem.duration || 0)}
                    </span>
                    {avgConfidence !== null && (
                      <span
                        className="ss-meta-detail"
                        title="Độ tin cậy nhận diện lời hát trung bình trên tất cả các câu"
                      >
                        Tin cậy trung bình {Math.round(avgConfidence * 100)}%
                      </span>
                    )}
                    {activeItem.transcript?.is_multilingual && (
                      <span className="ss-meta-note">Đa ngôn ngữ</span>
                    )}
                  </div>
                </div>

                <div className="ss-header-actions">
                  {/* Primary Action: Save if dirty, else Next Flagged Line */}
                  {dirty ? (
                    <button
                      type="button"
                      className="ss-primary-btn"
                      disabled={savingAll}
                      onClick={() => void saveAllDraftSegments()}
                    >
                      {savingAll ? "Đang lưu…" : "✓ Lưu chỉnh sửa"}
                    </button>
                  ) : reviewStats.needsReview > 0 ? (
                    <button
                      type="button"
                      className="ss-primary-btn"
                      onClick={() => {
                        setViewMode("review");
                        navigateRelativeLine(1, true);
                      }}
                    >
                      Kiểm tra câu tiếp theo → ({reviewStats.needsReview})
                    </button>
                  ) : (
                    <span className="ss-all-reviewed-pill">✓ Đã duyệt xong tất cả</span>
                  )}

                  {reviewStats.needsReview > 0 && (
                    <button
                      type="button"
                      className="ss-secondary-btn ss-ai-reread-btn"
                      disabled={retranscribingTarget !== null || savingAll}
                      onClick={() => void handleRetranscribeFlagged()}
                      title="Dùng AI đa lượt (Beam x8 + giọng tách thuần + ngữ cảnh câu lân cận) để đọc lại chính xác tất cả các câu đang bị đánh dấu Cần kiểm tra"
                    >
                      {retranscribingTarget === "all"
                        ? "⏳ AI đang đọc lại…"
                        : `✦ AI đọc lại câu cần kiểm tra (${reviewStats.needsReview})`}
                    </button>
                  )}

                  {/* Single Compact Export Dropdown Menu */}
                  <div className="ss-menu-wrap" ref={exportMenuRef}>
                    <button
                      type="button"
                      className="ss-secondary-btn"
                      aria-haspopup="menu"
                      aria-expanded={exportMenuOpen}
                      onClick={() => {
                        setExportMenuOpen(v => !v);
                        setMoreMenuOpen(false);
                      }}
                    >
                      Xuất lời bài hát ▾
                    </button>
                    {exportMenuOpen && (
                      <div className="ss-dropdown-menu" role="menu" aria-label="Định dạng xuất lời bài hát">
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportTxt(true);
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>TXT — Văn bản lời bài hát</strong>
                          <small>Văn bản thuần UTF-8 kèm nhãn phân đoạn</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportLrc();
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>LRC — Lời đồng bộ theo câu</strong>
                          <small>Dấu thời gian [mm:ss.xx] cho trình phát nhạc</small>
                        </button>
                        {exporters.hasWordTimings && (
                          <button
                            type="button"
                            role="menuitem"
                            onClick={() => {
                              exporters.exportEnhancedLrc();
                              setExportMenuOpen(false);
                            }}
                          >
                            <strong>Enhanced LRC — Đồng bộ từng chữ</strong>
                            <small>Kèm thẻ &lt;mm:ss.xx&gt; theo từng từ</small>
                          </button>
                        )}
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportSrt();
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>SRT — Phụ đề video</strong>
                          <small>Chuẩn SubRip với thời gian bắt đầu &amp; kết thúc</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportJson({
                              mode: activeItem.transcript?.mode,
                              alignment_method: activeItem.transcript?.alignment_method,
                            });
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>JSON — Dữ liệu có cấu trúc</strong>
                          <small>Toàn bộ metadata, trạng thái duyệt &amp; dấu thời gian</small>
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Secondary Actions Menu */}
                  <div className="ss-menu-wrap" ref={moreMenuRef}>
                    <button
                      type="button"
                      className="ss-icon-menu-btn"
                      aria-haspopup="menu"
                      aria-expanded={moreMenuOpen}
                      aria-label="Tác vụ khác"
                      title="Tác vụ khác (Sao chép, Đồng bộ lại, Trích xuất lại)"
                      onClick={() => {
                        setMoreMenuOpen(v => !v);
                        setExportMenuOpen(false);
                      }}
                    >
                      ⋯
                    </button>
                    {moreMenuOpen && (
                      <div className="ss-dropdown-menu" role="menu" aria-label="Tác vụ bổ sung">
                        <button type="button" role="menuitem" onClick={() => void copyScript()}>
                          <strong>{copied ? "✓ Đã sao chép" : "Sao chép toàn bộ lời"}</strong>
                          <small>Chép nhanh vào bộ nhớ tạm</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          disabled={realigning}
                          onClick={() => void handleRealignFromEditor()}
                        >
                          <strong>{realigning ? "Đang đồng bộ lại…" : "⇄ Đồng bộ lại dấu thời gian"}</strong>
                          <small>Giữ nguyên câu chữ đã sửa, căn lại thời gian (dùng cache giọng hát)</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => void handleRetry(activeItem.id, true)}
                        >
                          <strong>↻ Chạy lại AI trích xuất</strong>
                          <small>Tạo lại bản lời mới từ âm thanh gốc (sẽ hỏi xác nhận nếu đã sửa)</small>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {formError && (
                <div className="ss-inline-alert" role="alert">
                  <span>{formError}</span>
                  <button type="button" onClick={() => setFormError("")} aria-label="Đóng thông báo lỗi">
                    ×
                  </button>
                </div>
              )}

              {/* 2. Sticky Audio Transport & Selected-Line Assist Bar */}
              <div className="ss-transport-sticky">
                <div className="songscription-player-bar ss-player-row">
                  <audio
                    key={activeItem.id}
                    ref={audioRef}
                    src={urls.songscriptionAudio(activeItem.id)}
                    preload="metadata"
                    onPlay={() => setPlaying(true)}
                    onPause={e => {
                      setPlaying(false);
                      handleAudioTimeUpdate(e.currentTarget);
                    }}
                    onEnded={e => {
                      setPlaying(false);
                      handleAudioTimeUpdate(e.currentTarget);
                    }}
                    onTimeUpdate={e => handleAudioTimeUpdate(e.currentTarget)}
                    onSeeking={e => handleAudioTimeUpdate(e.currentTarget)}
                    onSeeked={e => handleAudioTimeUpdate(e.currentTarget)}
                    onLoadedMetadata={e => {
                      const loaded = e.currentTarget.duration;
                      setAudioDuration(Number.isFinite(loaded) && loaded > 0 ? loaded : activeItem.duration || 0);
                      handleAudioTimeUpdate(e.currentTarget);
                    }}
                    onVolumeChange={e => {
                      const audio = e.currentTarget;
                      setVolume(audio.volume);
                      setMuted(audio.muted);
                    }}
                  />
                  <div className="ss-transport-btns">
                    <button
                      type="button"
                      className="ss-mini-btn ss-transport-icon"
                      onClick={() => seekAudio(currentTime - 5)}
                      title="Lùi 5 giây"
                      aria-label="Lùi 5 giây"
                    >
                      <Icon name="rewind5" color={colors.text} size={21} />
                    </button>
                    <button
                      type="button"
                      className="songscription-play-btn"
                      aria-label={playing ? "Tạm dừng (Space)" : "Phát bài hát (Space)"}
                      title={playing ? "Tạm dừng (Space)" : "Phát bài hát (Space)"}
                      onClick={() => {
                        const el = audioRef.current;
                        if (!el) return;
                        if (el.paused) void el.play().catch(() => {});
                        else el.pause();
                      }}
                    >
                      <Icon name={playing ? "pause" : "play"} color={colors.onPrimary} size={20} />
                    </button>
                    <button
                      type="button"
                      className="ss-mini-btn ss-transport-icon"
                      onClick={() => seekAudio(currentTime + 5)}
                      title="Tua tới 5 giây"
                      aria-label="Tua tới 5 giây"
                    >
                      <Icon name="forward5" color={colors.text} size={21} />
                    </button>
                    <VolumeControl
                      value={muted ? 0 : volume}
                      onChange={changeVolume}
                      className="ss-volume-control"
                      buttonClassName="ss-mini-btn ss-transport-icon"
                      onOpen={() => {
                        setExportMenuOpen(false);
                        setMoreMenuOpen(false);
                      }}
                    />
                  </div>

                  <span className="songscription-clock">
                    {formatClock(position)} / {formatClock(duration)}
                  </span>

                  <input
                    className="ss-seek-slider"
                    type="range"
                    min={0}
                    max={duration || 1}
                    step={0.01}
                    value={position}
                    style={{ "--fill": `${progressPercent}%` } as CSSProperties}
                    disabled={duration <= 0}
                    onChange={e => seekAudio(Number(e.target.value))}
                    aria-label="Vị trí phát nhạc"
                    aria-valuetext={`${formatClock(position)} trên ${formatClock(duration)}`}
                  />

                  <select
                    value={playbackRate}
                    aria-label="Tốc độ phát"
                    onChange={e => {
                      const rate = Number(e.target.value);
                      setPlaybackRate(rate);
                      if (audioRef.current) audioRef.current.playbackRate = rate;
                    }}
                  >
                    {[0.5, 0.75, 1, 1.25, 1.5].map(r => (
                      <option key={r} value={r}>
                        {r}×
                      </option>
                    ))}
                  </select>

                  <div className="ss-display-toggles">
                    <button
                      type="button"
                      className={`ss-toggle-chip ${autoFollow ? "on" : ""}`}
                      aria-pressed={autoFollow}
                      onClick={() => setAutoFollow(v => !v)}
                      title="Tự động cuộn theo câu đang phát"
                    >
                      Tự cuộn
                    </button>
                    {hasAnyChords && (
                      <button
                        type="button"
                        className={`ss-toggle-chip ${showChords ? "on" : ""}`}
                        aria-pressed={showChords}
                        onClick={() => setShowChords(v => !v)}
                        title="Hiện hoặc ẩn hợp âm trên từng câu hát"
                      >
                        Hợp âm
                      </button>
                    )}
                    {hasWordTimings && (
                      <button
                        type="button"
                        className={`ss-toggle-chip ${showWordHighlight ? "on" : ""}`}
                        aria-pressed={showWordHighlight}
                        onClick={() => setShowWordHighlight(v => !v)}
                        title="Tô sáng từng từ khi đang phát"
                      >
                        Tô sáng từ
                      </button>
                    )}
                  </div>
                </div>

                {/* Selected-Line Audio Assist Strip (shown when a line is active/selected) */}
                {selectedSegment && (
                  <div className="ss-selected-assist" role="region" aria-label="Điều khiển nghe cho câu đang chọn">
                    <span className="ss-assist-label">
                      Câu #{selectedSegment.id} ({formatPreciseClock(selectedSegment.start)} –{" "}
                      {formatPreciseClock(selectedSegment.end)})
                    </span>
                    <div className="ss-assist-actions">
                      <button
                        type="button"
                        onClick={() => navigateRelativeLine(-1, false)}
                        title="Câu trước (K / ↑)"
                      >
                        ← Câu trước
                      </button>
                      <button
                        type="button"
                        onClick={replaySelectedLine}
                        title="Nghe lại câu đang chọn (R)"
                      >
                        ↺ Nghe lại
                      </button>
                      <button
                        type="button"
                        className={loopSelectedLine ? "on" : ""}
                        aria-pressed={loopSelectedLine}
                        onClick={() => {
                          const next = !loopSelectedLine;
                          setLoopSelectedLine(next);
                          if (next) seekAudio(selectedSegment.start, true);
                        }}
                        title="Lặp lại đoạn âm thanh của câu này (L)"
                      >
                        ⟳ {loopSelectedLine ? "Đang lặp câu" : "Lặp câu"}
                      </button>
                      <button
                        type="button"
                        onClick={() => navigateRelativeLine(1, false)}
                        title="Câu tiếp theo (J / ↓)"
                      >
                        Câu sau →
                      </button>
                    </div>
                  </div>
                )}

                {/* 3. Review Queue Bar: Filters, Progress, Search, Undo/Redo */}
                <div className="ss-queue-bar">
                  <div className="ss-filter-tabs" role="tablist" aria-label="Bộ lọc hàng đợi kiểm tra lời">
                    {(
                      [
                        ["all", "Tất cả", reviewStats.total],
                        ["needs_review", "Cần kiểm tra", reviewStats.needsReview],
                        ["verified", "Đã xác nhận", reviewStats.verified],
                        ["edited", "Đã chỉnh sửa", reviewStats.edited],
                      ] as [ReviewFilter, string, number][]
                    ).map(([key, label, count]) => (
                      <button
                        key={key}
                        type="button"
                        role="tab"
                        aria-selected={reviewFilter === key}
                        className={key === "needs_review" && count > 0 ? "has-flagged" : ""}
                        onClick={() => {
                          setReviewFilter(key);
                          setViewMode("review");
                        }}
                      >
                        <span>{label}</span>
                        <em>{count}</em>
                      </button>
                    ))}
                  </div>

                  <div className="ss-queue-right">
                    <div className="ss-review-progress" title={`Đã rà soát ${reviewStats.reviewedCount}/${reviewStats.total} câu`}>
                      <div className="ss-review-progress-bar">
                        <div style={{ width: `${reviewStats.progressPercent}%` }} />
                      </div>
                      <span>
                        {reviewStats.reviewedCount}/{reviewStats.total} ({reviewStats.progressPercent}%)
                      </span>
                    </div>

                    <input
                      className="songscription-search ss-search-input"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Tìm lời, đoạn, hợp âm…"
                      aria-label="Tìm câu hát"
                    />

                    <div className="ss-undo-group">
                      <button
                        type="button"
                        disabled={!undoStack.length}
                        onClick={undo}
                        title="Hoàn tác (Ctrl+Z)"
                        aria-label="Hoàn tác"
                      >
                        ↶
                      </button>
                      <button
                        type="button"
                        disabled={!redoStack.length}
                        onClick={redo}
                        title="Làm lại (Ctrl+Shift+Z)"
                        aria-label="Làm lại"
                      >
                        ↷
                      </button>
                    </div>

                    <div className="songscription-mode-tabs ss-view-switch" role="tablist" aria-label="Chế độ xem">
                      <button
                        type="button"
                        role="tab"
                        aria-selected={viewMode === "review"}
                        onClick={() => setViewMode("review")}
                      >
                        Rà soát
                      </button>
                      <button
                        type="button"
                        role="tab"
                        aria-selected={viewMode === "plain"}
                        onClick={() => setViewMode("plain")}
                      >
                        Văn bản
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3.5. AI Precision Re-read Results Banner */}
              {aiRefineResults.length > 0 && (
                <div className="ss-ai-refine-summary" role="status" aria-live="polite">
                  <div className="ss-ai-refine-summary-head">
                    <strong>
                      ✦ Kết quả AI đọc lại chính xác ({aiRefineResults.length} câu ·{" "}
                      {aiRefineResults.filter(r => r.improved).length} câu cải thiện)
                    </strong>
                    <button
                      type="button"
                      className="ss-ai-refine-close"
                      onClick={() => setAiRefineResults([])}
                      aria-label="Đóng bảng kết quả đọc lại"
                    >
                      Đóng
                    </button>
                  </div>
                  <div className="ss-ai-refine-list">
                    {aiRefineResults.map(r => (
                      <div key={r.id} className={`ss-ai-refine-item ${r.improved ? "improved" : ""}`}>
                        <div className="ss-ai-refine-item-meta">
                          <span className="ss-ai-refine-id">Câu #{r.id}</span>
                          <span className="ss-ai-refine-conf">
                            {Math.round(r.old_confidence * 100)}% →{" "}
                            <strong>{Math.round(r.new_confidence * 100)}%</strong>
                          </span>
                          <span className="ss-ai-refine-variant">{r.variant}</span>
                          {r.auto_resolved && (
                            <span className="ss-status-chip verified">✓ Đã đạt chuẩn</span>
                          )}
                        </div>
                        <div className="ss-ai-refine-diff">
                          {r.changed ? (
                            <>
                              <span className="ss-ai-refine-old">{r.old_text}</span>
                              <span className="ss-ai-refine-arrow">→</span>
                              <span className="ss-ai-refine-new">{r.new_text}</span>
                            </>
                          ) : (
                            <span className="ss-ai-refine-same">{r.new_text}</span>
                          )}
                        </div>
                        {r.changed && (
                          <button
                            type="button"
                            className="ss-ai-refine-undo"
                            onClick={() => void undoAiRefineLine(r)}
                          >
                            ↶ Hoàn tác
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 4. Compact Transcript Rows / Plain Text Body */}
              <div
                className="songscription-body ss-transcript-body"
                ref={listScrollRef}
                onWheel={() => {
                  if (playing && autoFollow) setAutoFollow(false);
                }}
              >
                {!draftSegments.length ? (
                  <div className="songscription-no-lyrics">
                    <p>
                      Không phát hiện thấy câu hát có lời trong bản thu này (bản nhạc hòa tấu không lời).
                    </p>
                    <button
                      type="button"
                      className="ss-secondary-btn"
                      onClick={() => insertLineAfter(-1)}
                    >
                      ＋ Thêm dòng lời thủ công
                    </button>
                  </div>
                ) : viewMode === "plain" ? (
                  <div className="songscription-plain-text">
                    {draftSegments
                      .map(s => (s.section ? `[${s.section}]\n${s.text}` : s.text))
                      .join("\n")}
                  </div>
                ) : !filteredSegments.length ? (
                  <div className="songscription-no-lyrics">
                    <p>Không có câu hát nào khớp với bộ lọc hiện tại.</p>
                    <button
                      type="button"
                      className="ss-secondary-btn"
                      onClick={() => {
                        setReviewFilter("all");
                        setSearchQuery("");
                      }}
                    >
                      Xem tất cả ({draftSegments.length} câu)
                    </button>
                  </div>
                ) : (
                  <div className="ss-rows">
                    {filteredSegments.map(seg => {
                      const realIdx = draftSegments.findIndex(s => s.id === seg.id);
                      const isPlayingRow = seg.id === playingSegmentId;
                      const isSelected = seg.id === selectedSegId;
                      const status = getEffectiveReviewStatus(seg);
                      const flagged = isLineFlaggedForReview(seg);
                      const reason = flagged ? getReviewReason(seg) : null;
                      const showSegLang =
                        activeItem.transcript?.is_multilingual &&
                        seg.lang &&
                        seg.lang !== activeItem.transcript.language;

                      return (
                        <div key={seg.id} data-seg-id={seg.id}>
                          {seg.section && (
                            <div className="songscription-section-divider">
                              <span>[{seg.section}]</span>
                            </div>
                          )}

                          <div
                            className={`ss-row ${isPlayingRow ? "playing" : ""} ${isSelected ? "selected" : ""} status-${status}`}
                            aria-current={isPlayingRow ? "true" : undefined}
                          >
                            {/* Compact Primary Row */}
                            <div className="ss-row-main">
                              <button
                                type="button"
                                className="songscription-time-pill ss-time-btn"
                                onClick={() => selectAndSeekLine(seg, true)}
                                title={`Phát từ ${formatPreciseClock(seg.start)}`}
                              >
                                {formatClock(seg.start)}
                              </button>

                              <button
                                type="button"
                                className="ss-row-text-btn"
                                onClick={() => selectAndSeekLine(seg, true)}
                                aria-expanded={isSelected}
                              >
                                <span className="ss-row-lyrics">
                                  {showWordHighlight && seg.words && seg.words.length > 0
                                    ? seg.words.map((w, wi) => {
                                        const wordState = getWordPlaybackState(w, currentTime);
                                        return (
                                          <span
                                            key={wi}
                                            className={`ss-lyric-word word-${wordState} ${
                                              w.uncertain && status !== "verified" && status !== "edited"
                                                ? "word-uncertain"
                                                : ""
                                            }`}
                                          >
                                            {w.word}
                                          </span>
                                        );
                                      })
                                    : seg.text || <em className="ss-empty-line-hint">(Dòng trống — nhấp để nhập lời)</em>}
                                </span>
                              </button>

                              <div className="ss-row-end">
                                {showChords && seg.chord && (
                                  <span className="line-chord-tag ss-chord-subtle">{seg.chord}</span>
                                )}
                                {showSegLang && (
                                  <span className="ss-row-language" title={`Ngôn ngữ: ${seg.lang.toUpperCase()}`}>
                                    {seg.lang.toUpperCase()}
                                  </span>
                                )}
                                {status === "needs_review" && (
                                  <span className="ss-status-chip needs-review" title={reason ?? "Cần kiểm tra"}>
                                    Cần kiểm tra
                                  </span>
                                )}
                                {status === "skipped" && (
                                  <span className="ss-status-chip skipped" title="Đã bỏ qua để xem lại sau">
                                    Xem lại sau
                                  </span>
                                )}
                                {status === "edited" && (
                                  <span className="ss-status-chip edited">Đã sửa</span>
                                )}
                                <button
                                  type="button"
                                  className="ss-row-expand-btn"
                                  aria-label={isSelected ? "Đóng chỉnh sửa câu này" : "Chỉnh sửa câu này"}
                                  aria-expanded={isSelected}
                                  onClick={() => {
                                    if (isSelected) setSelectedSegId(null);
                                    else selectAndSeekLine(seg, false);
                                  }}
                                >
                                  {isSelected ? "▴" : "✎"}
                                </button>
                              </div>
                            </div>

                            {/* Expanded Inline Editor (ONLY on active/selected row) */}
                            {isSelected && (
                              <div className="ss-inline-editor">
                                {flagged && reason && (
                                  <div className="ss-reason-banner">
                                    <span>⚠ {reason}</span>
                                    {seg.confidence != null && (
                                      <small>Độ tin cậy câu: {Math.round(seg.confidence * 100)}%</small>
                                    )}
                                  </div>
                                )}

                                <div className="ss-inline-text-wrap">
                                  <label className="sr-only" htmlFor={`ss-line-input-${seg.id}`}>
                                    Nội dung câu hát {seg.id}
                                  </label>
                                  <input
                                    id={`ss-line-input-${seg.id}`}
                                    className="ss-inline-text-input"
                                    value={seg.text}
                                    onChange={e => updateLineField(seg.id, { text: e.target.value })}
                                    onKeyDown={e => {
                                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                                        e.preventDefault();
                                        void saveAllDraftSegments();
                                      }
                                    }}
                                    placeholder="Nhập lời bài hát chính xác…"
                                  />
                                </div>

                                <div className="ss-inline-controls-grid">
                                  <div className="ss-time-group">
                                    <label>
                                      <span>Bắt đầu (s)</span>
                                      <input
                                        type="number"
                                        step={0.1}
                                        min={0}
                                        value={seg.start}
                                        onChange={e =>
                                          updateLineField(seg.id, { start: Math.max(0, Number(e.target.value)) })
                                        }
                                      />
                                    </label>
                                    <button
                                      type="button"
                                      className="ss-capture-time-btn"
                                      title="Đặt thời gian bắt đầu bằng vị trí phát hiện tại"
                                      onClick={() =>
                                        updateLineField(seg.id, { start: Number(currentTime.toFixed(2)) })
                                      }
                                    >
                                      ⏱ Lấy {formatClock(currentTime)}
                                    </button>
                                  </div>

                                  <div className="ss-time-group">
                                    <label>
                                      <span>Kết thúc (s)</span>
                                      <input
                                        type="number"
                                        step={0.1}
                                        min={seg.start}
                                        value={seg.end}
                                        onChange={e =>
                                          updateLineField(seg.id, {
                                            end: Math.max(seg.start + 0.05, Number(e.target.value)),
                                          })
                                        }
                                      />
                                    </label>
                                    <button
                                      type="button"
                                      className="ss-capture-time-btn"
                                      title="Đặt thời gian kết thúc bằng vị trí phát hiện tại"
                                      onClick={() =>
                                        updateLineField(seg.id, {
                                          end: Math.max(
                                            Number((seg.start + 0.1).toFixed(2)),
                                            Number(currentTime.toFixed(2)),
                                          ),
                                        })
                                      }
                                    >
                                      ⏱ Lấy {formatClock(currentTime)}
                                    </button>
                                  </div>

                                  <label className="ss-section-select-wrap">
                                    <span>Phân đoạn</span>
                                    <select
                                      value={seg.section ?? ""}
                                      onChange={e =>
                                        updateLineField(seg.id, { section: e.target.value || null }, false)
                                      }
                                    >
                                      {SECTION_OPTIONS.map(sec => (
                                        <option key={sec} value={sec}>
                                          {sec ? `[${sec}]` : "— Không gắn —"}
                                        </option>
                                      ))}
                                    </select>
                                  </label>

                                  {seg.words && seg.words.length > 0 && (
                                    <button
                                      type="button"
                                      className={`ss-toggle-chip ${showWordEditor ? "on" : ""}`}
                                      aria-expanded={showWordEditor}
                                      onClick={() => setShowWordEditor(v => !v)}
                                    >
                                      Thời gian từng từ ({seg.words.length})
                                    </button>
                                  )}
                                </div>

                                {showWordEditor && seg.words && seg.words.length > 0 && (
                                  <div className="ss-word-editor-strip">
                                    {seg.words.map((w, wi) => (
                                      <div key={wi} className={`ss-word-chip ${w.uncertain ? "uncertain" : ""}`}>
                                        <strong>{w.word.trim() || "·"}</strong>
                                        <input
                                          type="number"
                                          step={0.05}
                                          value={w.start}
                                          aria-label={`Bắt đầu từ ${w.word}`}
                                          onChange={e =>
                                            updateWordTimestamp(seg.id, wi, "start", Number(e.target.value))
                                          }
                                        />
                                        <span>–</span>
                                        <input
                                          type="number"
                                          step={0.05}
                                          value={w.end}
                                          aria-label={`Kết thúc từ ${w.word}`}
                                          onChange={e =>
                                            updateWordTimestamp(seg.id, wi, "end", Number(e.target.value))
                                          }
                                        />
                                      </div>
                                    ))}
                                  </div>
                                )}

                                <div className="ss-inline-footer">
                                  <div className="ss-inline-review-btns">
                                    <button
                                      type="button"
                                      className="ss-confirm-line-btn"
                                      disabled={savingAll || retranscribingTarget !== null}
                                      onClick={() => void markLineReviewStatus(seg.id, "verified", true)}
                                      title="Đánh dấu đã kiểm tra và chuyển sang câu cần kiểm tra tiếp theo (V)"
                                    >
                                      ✓ Đánh dấu đã kiểm tra
                                    </button>
                                    <button
                                      type="button"
                                      className="ss-ai-reread-line-btn"
                                      disabled={savingAll || retranscribingTarget !== null}
                                      onClick={() => void handleRetranscribeFlagged([seg.id])}
                                      title="Chạy AI đa lượt (Beam x8 + giọng tách thuần + ngữ cảnh) riêng cho câu này"
                                    >
                                      {retranscribingTarget === seg.id ? "⏳ Đang đọc lại…" : "✦ AI đọc lại câu này"}
                                    </button>
                                    <button
                                      type="button"
                                      className="ss-skip-line-btn"
                                      disabled={savingAll}
                                      onClick={() => void markLineReviewStatus(seg.id, "skipped", true)}
                                    >
                                      Bỏ qua / xem lại sau
                                    </button>
                                    {dirty && (
                                      <button
                                        type="button"
                                        className="ss-save-inline-btn"
                                        disabled={savingAll}
                                        onClick={() => void saveAllDraftSegments()}
                                      >
                                        {savingAll ? "Đang lưu…" : "Lưu"}
                                      </button>
                                    )}
                                  </div>

                                  <div className="ss-inline-struct-btns">
                                    <button
                                      type="button"
                                      onClick={() => splitLineAt(realIdx)}
                                      title="Tách dòng này thành 2 câu"
                                    >
                                      ✂ Tách
                                    </button>
                                    <button
                                      type="button"
                                      disabled={realIdx >= draftSegments.length - 1}
                                      onClick={() => mergeWithNext(realIdx)}
                                      title="Gộp với câu kế tiếp"
                                    >
                                      ⊕ Gộp xuống
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => insertLineAfter(realIdx)}
                                      title="Chèn thêm câu mới bên dưới"
                                    >
                                      ＋ Chèn dưới
                                    </button>
                                    <button
                                      type="button"
                                      className="danger"
                                      disabled={draftSegments.length <= 1}
                                      onClick={() => deleteLineAt(realIdx)}
                                      title="Xóa câu này"
                                    >
                                      × Xóa
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => setSelectedSegId(null)}
                                      title="Đóng trình sửa dòng này"
                                    >
                                      Đóng
                                    </button>
                                  </div>
                                </div>
                              </div>
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
        </section>
      </div>
    </div>
  );
}
