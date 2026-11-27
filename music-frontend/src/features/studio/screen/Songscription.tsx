import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
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
import { getSongscriptionModels, normalizeSongscriptionModelId } from "../utils/songscriptionModels";
import {
  YouTubeStagePlayer,
  extractYoutubeVideoId,
  type YouTubeStagePlayerHandle,
} from "../components/YouTubeStagePlayer";
import "../styles/songscription-youtube.css";
import {
  KARAOKE_SYNC_LEAD_SEC,
  getActiveSegmentIds,
  getPlaybackPercent,
  getPlaybackPosition,
  getWordPlaybackState,
  prepareWordsForKaraoke,
} from "../utils/lyricsPlayback";
import {
  PIPELINE_STAGES,
  STAGE_LABELS,
  computeAvgConfidence,
  computeReviewStats,
  exportTranscriptFiles,
  formatClock,
  getEffectiveEventType,
  getEffectiveReviewStatus,
  isAdlibOrVocalise,
  isLineFlaggedForReview,
  isOverlapSegment,
  resolveStageIndex,
  streamLabel,
  type ExportPolicy,
} from "../utils/lyricsUtils";

const EMPTY_ITEMS: SongscriptionItem[] = [];
const PLAYBACK_SPEEDS = [0.5, 0.75, 1, 1.25, 1.5] as const;

type ReviewFilter = "all" | "adlib" | "overlap" | "needs_review" | "edited";
type AudioSourceMode = "original" | "vocal" | "instrumental" | `stream:${string}`;

function getStemAudioHint(mode: AudioSourceMode, volume01: number, isMuted: boolean): string {
  const channelLabel =
    mode === "instrumental"
      ? "nhạc nền không lời"
      : mode === "vocal"
        ? "kênh vocal"
        : mode.startsWith("stream:")
          ? "solo luồng"
          : "";
  if (!channelLabel) return "";
  if (isMuted || volume01 <= 0.001) {
    return `Đã tắt tiếng ${channelLabel}`;
  }
  const pct = Math.round(Math.min(1, Math.max(0, volume01)) * 100);
  return `Đang phát ${channelLabel} (${pct}%)`;
}

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
  isActiveView = true,
}: {
  path: string;
  navigate: (to: string) => void;
  projects: ProjectSummary[];
  youtubeSupported: boolean;
  isActiveView?: boolean;
}) {
  const qc = useQueryClient();
  const route = useMemo(() => parseSongscriptionRoute(path), [path]);
  const [retainedSessionId, setRetainedSessionId] = useState<string | null>(() =>
    route.mode === "detail" ? route.sessionId : null,
  );
  const activeId = route.mode === "detail" ? route.sessionId : retainedSessionId;
  const isDetailVisible = isActiveView && route.mode === "detail" && Boolean(route.sessionId);
  const isLibraryVisible = isActiveView && route.mode !== "detail";

  const listQuery = useQuery({
    queryKey: ["songscriptions"],
    queryFn: api.songscriptions,
    refetchInterval: q =>
      q.state.data?.items.some(item => item.state === "queued" || item.state === "running") ? 1200 : false,
  });

  const items = listQuery.data?.items ?? EMPTY_ITEMS;
  const models = getSongscriptionModels(listQuery.data?.models);
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
  const [audioSourceMode, setAudioSourceMode] = useState<AudioSourceMode>("original");
  const [stemLoading, setStemLoading] = useState(false);
  const [exportPolicy, setExportPolicy] = useState<ExportPolicy>("with_adlib");
  const [ytPanelTab, setYtPanelTab] = useState<"lyrics" | "review" | "plain">("lyrics");
  const [ytLinkEditorOpen, setYtLinkEditorOpen] = useState(false);
  const [ytLinkDraft, setYtLinkDraft] = useState("");
  const [savingYtLink, setSavingYtLink] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string;
    message: string;
    confirmLabel: string;
    tone?: "danger" | "primary";
    onConfirm: () => void | Promise<void>;
  } | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showChords, setShowChords] = useState(true);
  const [showWordHighlight, setShowWordHighlight] = useState(true);
  const [autoFollow, setAutoFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  // Selected / Expanded line for inline editing
  const [selectedSegId, setSelectedSegId] = useState<number | null>(null);
  const [ytEditingSegId, setYtEditingSegId] = useState<number | null>(null);
  const [loopSelectedLine, setLoopSelectedLine] = useState(false);

  // Precision AI Re-transcribe state for flagged ("Cần kiểm tra") lines
  const [retranscribingTarget, setRetranscribingTarget] = useState<"all" | number | null>(null);
  const [aiRefineResults, setAiRefineResults] = useState<RetranscribeLineResult[]>([]);
  const [aiRefineOpen, setAiRefineOpen] = useState(false);

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
  const ytPlayerRef = useRef<YouTubeStagePlayerHandle | null>(null);
  const lastSyncedTimeRef = useRef(-1);
  const listScrollRef = useRef<HTMLDivElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const moreMenuRef = useRef<HTMLDivElement>(null);
  const aiRefineMenuRef = useRef<HTMLDivElement>(null);
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

  const proceedOpenSession = (sid: string) => {
    setCreateModalOpen(false);
    setRetainedSessionId(sid);
    if (sid !== activeId) {
      audioRef.current?.pause();
      ytPlayerRef.current?.pause();
      setSelectedSegId(null);
      setAiRefineResults([]);
      setAiRefineOpen(false);
      setAudioSourceMode("original");
      setCurrentTime(0);
      setAudioDuration(0);
      setPlaying(false);
      setLoopSelectedLine(false);
    }
    navigate(`/songscription/${encodeURIComponent(sid)}`);
  };

  const openSession = (sid: string) => {
    if (dirty && activeId && sid !== activeId) {
      setConfirmDialog({
        title: "Chưa lưu chỉnh sửa",
        message: "Bạn có thay đổi lời bài hát chưa lưu. Rời khỏi phiên này mà không lưu?",
        confirmLabel: "Rời khỏi",
        tone: "danger",
        onConfirm: () => proceedOpenSession(sid),
      });
      return;
    }
    proceedOpenSession(sid);
  };

  const proceedBackToLibrary = () => {
    setSelectedSegId(null);
    setLoopSelectedLine(false);
    setCreateModalOpen(false);
    if (activeId) {
      setRetainedSessionId(activeId);
    }
    navigate("/songscription");
  };

  const backToLibrary = () => {
    if (dirty && activeId) {
      setConfirmDialog({
        title: "Chưa lưu chỉnh sửa",
        message: "Bạn có thay đổi lời bài hát chưa lưu. Quay lại thư viện Songscription mà không lưu?",
        confirmLabel: "Quay lại",
        tone: "danger",
        onConfirm: proceedBackToLibrary,
      });
      return;
    }
    proceedBackToLibrary();
  };

  const closeCreateModal = () => {
    setCreateModalOpen(false);
    if (route.mode === "new") {
      navigate("/songscription");
    }
  };

  // Close dropdown menus & AI refine popover on outside click or Escape
  useEffect(() => {
    if (!exportMenuOpen && !moreMenuOpen && !aiRefineOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (exportMenuOpen && exportMenuRef.current && !exportMenuRef.current.contains(target)) {
        setExportMenuOpen(false);
      }
      if (moreMenuOpen && moreMenuRef.current && !moreMenuRef.current.contains(target)) {
        setMoreMenuOpen(false);
      }
      if (aiRefineOpen && aiRefineMenuRef.current && !aiRefineMenuRef.current.contains(target)) {
        setAiRefineOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setExportMenuOpen(false);
        setMoreMenuOpen(false);
        setAiRefineOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [exportMenuOpen, moreMenuOpen, aiRefineOpen]);

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

  const hasVocalStemAudio = Boolean(activeItem?.vocal_cached || activeItem?.vocal_isolated);
  const hasInstrumentalAudio = Boolean(activeItem?.instrumental_cached || hasVocalStemAudio);
  const youtubeVideoId = useMemo(
    () => activeItem?.youtube_id || extractYoutubeVideoId(activeItem?.youtube_url),
    [activeItem?.youtube_id, activeItem?.youtube_url],
  );
  const useYouTubeAudio = Boolean(youtubeVideoId && audioSourceMode === "original");

  const isolatedStreams = useMemo(() => {
    const fromTranscript = (activeItem?.transcript?.streams ?? []).filter(st => st.has_isolated_audio);
    if (fromTranscript.length > 0) return fromTranscript;
    return (activeItem?.available_streams ?? []).map(id => ({
      id,
      label: streamLabel(id),
      has_isolated_audio: true,
    }));
  }, [activeItem?.transcript?.streams, activeItem?.available_streams]);

  const activeAudioSrc = useMemo(() => {
    if (!activeItem) return "";
    if (audioSourceMode.startsWith("stream:")) {
      const stId = audioSourceMode.slice(7);
      if (isolatedStreams.some(st => st.id === stId)) {
        return urls.songscriptionStreamAudio(activeItem.id, stId);
      }
    }
    if (audioSourceMode === "vocal" && hasVocalStemAudio) {
      return urls.songscriptionVocalAudio(activeItem.id);
    }
    if (audioSourceMode === "instrumental" && hasInstrumentalAudio) {
      return urls.songscriptionInstrumentalAudio(activeItem.id);
    }
    return urls.songscriptionAudio(activeItem.id);
  }, [activeItem, audioSourceMode, hasVocalStemAudio, hasInstrumentalAudio, isolatedStreams]);

  const loadLocalAudioAt = (resumeTime: number, autoPlay: boolean) => {
    setStemLoading(true);
    requestAnimationFrame(() => {
      const a = audioRef.current;
      if (!a) {
        setStemLoading(false);
        return;
      }
      const cleanup = () => {
        a.removeEventListener("loadedmetadata", restore);
        a.removeEventListener("error", onErr);
      };
      const onErr = () => {
        setStemLoading(false);
        cleanup();
      };
      const restore = () => {
        setStemLoading(false);
        a.currentTime = Math.max(0, resumeTime);
        a.playbackRate = playbackRate;
        a.volume = Math.min(1, Math.max(0, volume));
        a.muted = muted || volume <= 0.001;
        lastSyncedTimeRef.current = a.currentTime;
        setCurrentTime(a.currentTime);
        if (autoPlay) void a.play().catch(() => {});
        cleanup();
      };
      a.addEventListener("loadedmetadata", restore);
      a.addEventListener("error", onErr);
      a.load();
    });
  };

  const switchAudioSource = (nextMode: AudioSourceMode) => {
    if (nextMode === audioSourceMode) return;
    if (nextMode === "vocal" && !hasVocalStemAudio) return;
    if (nextMode === "instrumental" && !hasInstrumentalAudio) return;
    if (nextMode.startsWith("stream:") && !isolatedStreams.some(st => st.id === nextMode.slice(7))) return;

    const el = audioRef.current;
    if (youtubeVideoId) {
      if (nextMode === "original") {
        // Switching from separated local stem (Vocal / Instrumental / Solo) back to native YouTube audio
        const resumeTime = el ? el.currentTime : currentTime;
        const wasPlaying = el ? !el.paused : playing;
        if (el && !el.paused) el.pause();
        setAudioSourceMode("original");
        setCurrentTime(resumeTime);
        lastSyncedTimeRef.current = resumeTime;
        requestAnimationFrame(() => {
          ytPlayerRef.current?.seekTo(resumeTime, wasPlaying);
        });
        return;
      }

      // Switching to separated Vocal / Instrumental / Solo stem (YouTube mutes automatically)
      const resumeTime =
        audioSourceMode === "original"
          ? ytPlayerRef.current?.getCurrentTime() || currentTime
          : el
            ? el.currentTime
            : currentTime;
      setAudioSourceMode(nextMode);
      loadLocalAudioAt(resumeTime, playing);
      return;
    }

    const resumeTime = el ? el.currentTime : currentTime;
    const wasPlaying = el ? !el.paused : false;
    setAudioSourceMode(nextMode);
    loadLocalAudioAt(resumeTime, wasPlaying);
  };

  const categoryCounts = useMemo(() => {
    let adlib = 0;
    let overlap = 0;
    const streams = new Set<string>();
    for (const seg of draftSegments) {
      streams.add(seg.stream_id || "stream_a");
      if (isAdlibOrVocalise(seg)) adlib++;
      if (isOverlapSegment(seg, draftSegments)) overlap++;
    }
    return { adlib, overlap, multiStream: streams.size > 1 };
  }, [draftSegments]);

  const filteredSegments = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return draftSegments.filter(seg => {
      const st = getEffectiveReviewStatus(seg);
      if (reviewFilter === "adlib" && !isAdlibOrVocalise(seg)) return false;
      if (reviewFilter === "overlap" && !isOverlapSegment(seg, draftSegments)) return false;
      if (reviewFilter === "needs_review" && !isLineFlaggedForReview(seg)) return false;
      if (reviewFilter === "edited" && st !== "edited") return false;
      if (!q) return true;
      return (
        seg.text.toLowerCase().includes(q) ||
        Boolean(seg.chord && seg.chord.toLowerCase().includes(q)) ||
        Boolean(seg.section && seg.section.toLowerCase().includes(q)) ||
        streamLabel(seg.stream_id).toLowerCase().includes(q)
      );
    });
  }, [draftSegments, reviewFilter, searchQuery]);

  const selectedSegment = useMemo(
    () => draftSegments.find(s => s.id === selectedSegId) ?? null,
    [draftSegments, selectedSegId],
  );

  const highlightTime = playing ? currentTime + KARAOKE_SYNC_LEAD_SEC : currentTime;

  // Precompute playback-repaired karaoke words & index map once per draft change (zero per-frame overhead)
  const { karaokeWordsBySegId, playbackBounds, segIndexById } = useMemo(() => {
    const wordsMap = new Map<number, SongscriptionSegment["words"]>();
    const idxMap = new Map<number, number>();
    const bounds: Pick<SongscriptionSegment, "id" | "start" | "end" | "stream_id">[] = [];
    for (let i = 0; i < draftSegments.length; i++) {
      const seg = draftSegments[i];
      idxMap.set(seg.id, i);
      const stream = seg.stream_id || "stream_a";
      let nextSameStreamStart: number | null = null;
      for (let j = i + 1; j < draftSegments.length; j++) {
        if ((draftSegments[j].stream_id || "stream_a") === stream) {
          nextSameStreamStart = draftSegments[j].start;
          break;
        }
      }
      const repaired = prepareWordsForKaraoke(seg.words, seg.start, seg.end, nextSameStreamStart);
      wordsMap.set(seg.id, repaired);
      const rawStart = repaired.length > 0 ? Math.min(seg.start, repaired[0].start) : seg.start;
      const rawEnd = repaired.length > 0 ? Math.max(seg.end, repaired[repaired.length - 1].end) : seg.end;
      const effEnd =
        nextSameStreamStart !== null && nextSameStreamStart > rawStart + 0.05
          ? Math.min(rawEnd, nextSameStreamStart)
          : rawEnd;
      bounds.push({ id: seg.id, start: rawStart, end: effEnd, stream_id: seg.stream_id });
    }
    return { karaokeWordsBySegId: wordsMap, playbackBounds: bounds, segIndexById: idxMap };
  }, [draftSegments]);

  const playingSegmentIds = useMemo(
    () => getActiveSegmentIds(playbackBounds, highlightTime),
    [playbackBounds, highlightTime],
  );

  const playingSegmentId = useMemo(() => {
    for (let i = playbackBounds.length - 1; i >= 0; i--) {
      const seg = playbackBounds[i];
      if (playingSegmentIds.has(seg.id)) {
        return seg.id;
      }
    }
    return null;
  }, [playbackBounds, playingSegmentIds]);

  // Auto-loop selected segment on the single authoritative <audio> timeline
  const handleAudioTimeUpdate = useCallback((el: HTMLAudioElement) => {
    const t = el.currentTime;
    if (loopSelectedLine && selectedSegment && !el.paused) {
      const endBound = Math.max(selectedSegment.start + 0.2, selectedSegment.end + 0.08);
      if (t >= endBound || t < selectedSegment.start - 0.25) {
        el.currentTime = Math.max(0, selectedSegment.start);
        lastSyncedTimeRef.current = el.currentTime;
        setCurrentTime(el.currentTime);
        return;
      }
    }
    if (Math.abs(t - lastSyncedTimeRef.current) >= 0.008) {
      lastSyncedTimeRef.current = t;
      setCurrentTime(t);
    }
  }, [loopSelectedLine, selectedSegment]);

  // Keep hidden local <audio> silent & paused whenever native YouTube audio ("Bản gốc") is active
  useEffect(() => {
    if (!useYouTubeAudio) return;
    const el = audioRef.current;
    if (el && !el.paused) {
      el.pause();
    }
  }, [useYouTubeAudio]);

  // Native 60fps audio-clock sync while playing local audio / vocal stem
  useEffect(() => {
    if (!playing || useYouTubeAudio) return;
    let frame = 0;
    const syncTime = () => {
      const audio = audioRef.current;
      if (audio) {
        handleAudioTimeUpdate(audio);
      }
      frame = requestAnimationFrame(syncTime);
    };
    frame = requestAnimationFrame(syncTime);
    return () => cancelAnimationFrame(frame);
  }, [playing, useYouTubeAudio, handleAudioTimeUpdate]);

  // Handle time updates coming directly from the native YouTube player ("Bản gốc")
  const handleYouTubeTimeUpdate = useCallback(
    (t: number) => {
      if (!useYouTubeAudio) return;
      if (loopSelectedLine && selectedSegment && playing) {
        const endBound = Math.max(selectedSegment.start + 0.2, selectedSegment.end + 0.08);
        if (t >= endBound || t < selectedSegment.start - 0.35) {
          ytPlayerRef.current?.seekTo(Math.max(0, selectedSegment.start), true);
          lastSyncedTimeRef.current = Math.max(0, selectedSegment.start);
          setCurrentTime(Math.max(0, selectedSegment.start));
          return;
        }
      }
      if (Math.abs(t - lastSyncedTimeRef.current) >= 0.008) {
        lastSyncedTimeRef.current = t;
        setCurrentTime(t);
      }
    },
    [useYouTubeAudio, loopSelectedLine, selectedSegment, playing],
  );

  const handleYouTubePlayingChange = useCallback(
    (isPlaying: boolean, t: number) => {
      if (isPlaying && activeId) {
        setRetainedSessionId(activeId);
      }
      if (useYouTubeAudio) {
        setPlaying(isPlaying);
        if (Math.abs(t - lastSyncedTimeRef.current) >= 0.008) {
          lastSyncedTimeRef.current = t;
          setCurrentTime(t);
        }
        return;
      }
      // In separated Vocal / Solo stem mode, clicking YouTube's native Play/Pause also controls the stem!
      const el = audioRef.current;
      if (!el) return;
      if (isPlaying && el.paused) {
        if (Math.abs(el.currentTime - t) > 0.65) {
          el.currentTime = Math.max(0, t);
        }
        void el.play().catch(() => {});
      } else if (!isPlaying && !el.paused) {
        el.pause();
      }
    },
    [useYouTubeAudio, activeId],
  );

  const handleYouTubeManualSeek = useCallback(
    (t: number) => {
      if (useYouTubeAudio) {
        lastSyncedTimeRef.current = t;
        setCurrentTime(t);
        return;
      }
      const el = audioRef.current;
      if (!el) return;
      const end = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : duration;
      el.currentTime = getPlaybackPosition(t, end);
      lastSyncedTimeRef.current = el.currentTime;
      setCurrentTime(el.currentTime);
    },
    [useYouTubeAudio, duration],
  );

  const handleYouTubeRateChange = useCallback((rate: number) => {
    setPlaybackRate(rate);
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
  }, []);

  const handleYouTubeVolumeChange = useCallback((vol01: number, isMuted: boolean) => {
    const next = Math.min(1, Math.max(0, vol01));
    const effMuted = isMuted || next <= 0.001;
    setVolume(next);
    setMuted(effMuted);
    const audio = audioRef.current;
    if (audio) {
      audio.volume = next;
      audio.muted = effMuted;
    }
  }, []);

  // Auto-follow active playing line smoothly ahead of the bottom viewport edge
  useEffect(() => {
    if (!isDetailVisible || !playing || !autoFollow || playingSegmentId === null) return;
    const container = listScrollRef.current;
    const row = container?.querySelector<HTMLElement>(`[data-seg-id="${playingSegmentId}"]`);
    if (!container || !row) return;
    const rowRect = row.getBoundingClientRect();
    const boxRect = container.getBoundingClientRect();
    if (rowRect.top < boxRect.top + 36 || rowRect.bottom > boxRect.bottom - Math.max(64, boxRect.height * 0.28)) {
      container.scrollTo({
        top: Math.max(0, container.scrollTop + (rowRect.top - boxRect.top) - boxRect.height * 0.26),
        behavior: "smooth",
      });
    }
  }, [isDetailVisible, playingSegmentId, playing, autoFollow]);

  // Start each panel/song view at the beginning so the first lines are never
  // left clipped by the previous song's auto-follow scroll position.
  useEffect(() => {
    if (!listScrollRef.current) return;
    listScrollRef.current.scrollTop = 0;
  }, [ytPanelTab, activeItem?.id]);

  const scrollRowIntoView = (segId: number) => {
    requestAnimationFrame(() => {
      const container = listScrollRef.current;
      const row = container?.querySelector<HTMLElement>(`[data-seg-id="${segId}"]`);
      if (!container || !row) return;
      // Scroll ONLY the internal lyrics container — never yank the outer page viewport
      const rowRect = row.getBoundingClientRect();
      const boxRect = container.getBoundingClientRect();
      const targetTop = Math.max(
        0,
        container.scrollTop + (rowRect.top - boxRect.top) - Math.max(0, (boxRect.height - rowRect.height) / 2),
      );
      container.scrollTo({ top: targetTop, behavior: "smooth" });
    });
  };

  const seekAudio = (sec: number, autoPlay = false) => {
    if (useYouTubeAudio) {
      const target = getPlaybackPosition(sec, duration);
      lastSyncedTimeRef.current = target;
      setCurrentTime(target);
      ytPlayerRef.current?.seekTo(target, autoPlay);
      return;
    }
    const el = audioRef.current;
    if (!el) return;
    const end = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : duration;
    el.currentTime = getPlaybackPosition(sec, end);
    lastSyncedTimeRef.current = el.currentTime;
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
    seekAudio(seg.start, autoPlay);
    scrollRowIntoView(seg.id);
  };

  const replaySelectedLine = () => {
    if (!selectedSegment) return;
    seekAudio(selectedSegment.start, true);
  };

  const toggleLocalAudioPlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void el.play().catch(() => {});
    else el.pause();
  };

  const cyclePlaybackSpeed = () => {
    const idx = PLAYBACK_SPEEDS.findIndex(r => Math.abs(r - playbackRate) < 0.01);
    const next = PLAYBACK_SPEEDS[(idx + 1) % PLAYBACK_SPEEDS.length];
    handleYouTubeRateChange(next);
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

  // Keyboard shortcuts for efficient review workflow (when viewing Songscription detail and not typing)
  useEffect(() => {
    if (!isDetailVisible || !isDone || ytPanelTab === "plain") return;
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
        if (useYouTubeAudio) {
          ytPlayerRef.current?.togglePlay();
          return;
        }
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
  }, [isDetailVisible, isDone, ytPanelTab, dirty, selectedSegment, draftSegments, filteredSegments, loopSelectedLine, useYouTubeAudio]);

  const handleCreated = async (created: SongscriptionItem) => {
    setCreateModalOpen(false);
    setSelectedSegId(null);
    setEditSession(null);
    setRetainedSessionId(created.id);
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

  const executeRetry = async (sid: string) => {
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

  const handleRetry = async (sid: string, confirmOverwrite = false) => {
    if (confirmOverwrite && hasUserCorrections) {
      setConfirmDialog({
        title: "Chạy lại AI trích xuất?",
        message:
          "Bạn đã chỉnh sửa hoặc xác nhận lời bài hát này. Chạy lại AI trích xuất sẽ thay thế bản lời hiện tại.",
        confirmLabel: "Chạy lại AI",
        tone: "danger",
        onConfirm: () => executeRetry(sid),
      });
      return;
    }
    await executeRetry(sid);
  };

  const handleRetranscribeFlagged = async (segmentIds?: number[], languageOverride?: string) => {
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
        language: languageOverride || undefined,
      });
      qc.setQueryData(["songscription", activeItem.id], res.item);
      setEditSession(null);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      setAiRefineResults(res.results);
      if (res.retranscribed_count === 0 || res.results.length === 0) {
        setAiRefineOpen(false);
        flashToast("Không còn câu nào cần kiểm tra để đọc lại.");
      } else {
        setExportMenuOpen(false);
        setMoreMenuOpen(false);
        setAiRefineOpen(true);
      }
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRetranscribingTarget(null);
    }
  };

  const aiRefineImprovedCount = useMemo(
    () => aiRefineResults.filter(r => r.improved).length,
    [aiRefineResults],
  );

  const jumpToRefinedLine = (segId: number) => {
    const seg = draftSegments.find(s => s.id === segId);
    if (!seg) return;
    if (ytPanelTab === "plain") setYtPanelTab("lyrics");
    setReviewFilter("all");
    setSearchQuery("");
    setAiRefineOpen(false);
    selectAndSeekLine(seg, false);
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
    setAiRefineResults(prev => {
      const remaining = prev.filter(r => r.id !== res.id);
      if (remaining.length === 0) setAiRefineOpen(false);
      return remaining;
    });
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

  const executeRealignFromEditor = async () => {
    if (!activeItem || realigning) return;
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
        model: normalizeSongscriptionModelId(activeItem.model),
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

  const handleRealignFromEditor = async () => {
    if (!activeItem || realigning) return;
    if (dirty) {
      setConfirmDialog({
        title: "Đồng bộ lại dấu thời gian?",
        message: "Hệ thống sẽ giữ nguyên câu chữ hiện tại trong trình sửa và tính lại dấu thời gian theo giọng hát.",
        confirmLabel: "Đồng bộ lại",
        tone: "primary",
        onConfirm: executeRealignFromEditor,
      });
      return;
    }
    await executeRealignFromEditor();
  };

  const executeDelete = async (item: SongscriptionItem) => {
    try {
      await api.deleteSongscription(item.id);
      setSelectedSegId(null);
      if (activeId === item.id) {
        audioRef.current?.pause();
        ytPlayerRef.current?.pause();
        setPlaying(false);
        setRetainedSessionId(null);
      }
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      flashToast(`Đã xóa "${item.title}"`);
      if (route.mode === "detail" && route.sessionId === item.id) {
        navigate("/songscription");
      }
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const handleDelete = async (item: SongscriptionItem) => {
    setConfirmDialog({
      title: "Xóa Songscription",
      message: `Bạn có chắc chắn muốn xóa "${item.title}"? Hành động này không thể hoàn tác.`,
      confirmLabel: "Xóa vĩnh viễn",
      tone: "danger",
      onConfirm: () => executeDelete(item),
    });
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
    setConfirmDialog({
      title: "Xóa câu hát",
      message: "Bạn có chắc chắn muốn xóa dòng lời bài hát này?",
      confirmLabel: "Xóa dòng",
      tone: "danger",
      onConfirm: () => {
        const next = draftSegments.filter((_, i) => i !== idx);
        applyDraftChange(next);
        setSelectedSegId(null);
      },
    });
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
  const activePlayingSegments = useMemo(
    () => draftSegments.filter(s => playingSegmentIds.has(s.id)),
    [draftSegments, playingSegmentIds],
  );

  const handleSaveYoutubeLink = async () => {
    if (!activeItem || savingYtLink) return;
    setSavingYtLink(true);
    setFormError("");
    try {
      const updated = await api.updateSongscription(activeItem.id, { youtube_url: ytLinkDraft.trim() });
      qc.setQueryData(["songscription", activeItem.id], updated);
      await qc.invalidateQueries({ queryKey: ["songscriptions"] });
      setYtLinkEditorOpen(false);
      setToastMsg(ytLinkDraft.trim() ? "Đã gắn video YouTube cho phiên này" : "Đã gỡ liên kết YouTube");
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSavingYtLink(false);
    }
  };
  const stageLabel = activeItem?.message
    ? STAGE_LABELS[activeItem.message] ?? activeItem.message
    : activeItem?.state === "queued"
      ? "Đang chờ trong hàng đợi GPU…"
      : "Đang xử lý…";

  const confirmModalNode = confirmDialog ? (
    <div
      className="ss-confirm-backdrop"
      role="presentation"
      onMouseDown={e => {
        if (e.target === e.currentTarget && !confirmBusy) setConfirmDialog(null);
      }}
    >
      <div
        className="ss-confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="ss-confirm-title"
        aria-describedby="ss-confirm-desc"
      >
        <div className="ss-confirm-head">
          <span className={`ss-confirm-badge ${confirmDialog.tone === "danger" ? "is-danger" : ""}`} aria-hidden="true">
            <Icon
              name={confirmDialog.tone === "danger" ? "trash" : "songscription"}
              color={confirmDialog.tone === "danger" ? colors.danger : colors.primary}
              size={18}
            />
          </span>
          <h3 id="ss-confirm-title">{confirmDialog.title}</h3>
        </div>
        <p id="ss-confirm-desc" className="ss-confirm-message">
          {confirmDialog.message}
        </p>
        <div className="ss-confirm-actions">
          <button
            type="button"
            className="ss-secondary-btn"
            disabled={confirmBusy}
            onClick={() => setConfirmDialog(null)}
          >
            Hủy
          </button>
          <button
            type="button"
            className={confirmDialog.tone === "danger" ? "ss-danger-btn" : "ss-primary-btn"}
            disabled={confirmBusy}
            onClick={async () => {
              setConfirmBusy(true);
              try {
                await confirmDialog.onConfirm();
                setConfirmDialog(null);
              } finally {
                setConfirmBusy(false);
              }
            }}
          >
            {confirmBusy ? "Đang xử lý…" : confirmDialog.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  ) : null;

  const pauseBackgroundPlayback = () => {
    if (useYouTubeAudio) {
      ytPlayerRef.current?.pause();
      setPlaying(false);
      return;
    }
    audioRef.current?.pause();
  };

  const bgChannelLabel =
    audioSourceMode === "vocal"
      ? "Kênh vocal"
      : audioSourceMode === "instrumental"
        ? "Instrumental"
        : audioSourceMode.startsWith("stream:")
          ? streamLabel(audioSourceMode.slice(7))
          : "Bản gốc";

  const backgroundMiniPlayer =
    !isDetailVisible && playing && activeItem && typeof document !== "undefined"
      ? createPortal(
          <div className="ss-bg-mini-player" role="region" aria-label="Đang phát nhạc nền Songscription">
            <button
              type="button"
              className="ss-bg-mini-open"
              onClick={() => navigate(`/songscription/${encodeURIComponent(activeItem.id)}`)}
              title="Quay lại màn hình bài hát đang phát"
            >
              <span className="ss-bg-mini-dot" aria-hidden="true">♪</span>
              <span className="ss-bg-mini-info">
                <strong>{activeItem.title}</strong>
                <small>
                  {bgChannelLabel} · {formatClock(position)}
                </small>
              </span>
            </button>
            <button
              type="button"
              className="ss-bg-mini-pause"
              onClick={pauseBackgroundPlayback}
              aria-label="Tạm dừng phát nhạc"
              title="Tạm dừng"
            >
              <Icon name="pause" color={colors.text} size={15} />
            </button>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      {isActiveView && toastMsg && (
        <div className="ss-toast" role="status" aria-live="polite">
          {toastMsg}
        </div>
      )}

      {isLibraryVisible && (
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
      )}

      {isActiveView && (
        <>
          <SongscriptionCreateModal
            open={isCreateOpen}
            projects={projects}
            models={models}
            vocalAvailable={vocalAvailable}
            youtubeSupported={youtubeSupported}
            onClose={closeCreateModal}
            onCreated={created => void handleCreated(created)}
          />
          {confirmModalNode}
        </>
      )}

      {backgroundMiniPlayer}

      {activeId && (
        <div
          key="ss-detail-host"
          className={isDetailVisible ? "songscription-studio ss-workspace" : "ss-bg-dock"}
          aria-hidden={!isDetailVisible || undefined}
        >
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

          {formError && (
            <div className="upload-error ss-workspace-error" role="alert">
              {formError}
            </div>
          )}

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
                        title="Điểm xác suất ASR trung bình (chỉ tính trên các câu có điểm ASR, không phải độ chính xác đã kiểm định)"
                      >
                        Điểm ASR TB {Math.round(avgConfidence * 100)}%
                      </span>
                    )}
                    {activeItem.transcript?.is_multilingual && (
                      <span className="ss-meta-note">Đa ngôn ngữ</span>
                    )}
                    {categoryCounts.multiStream && (
                      <span className="ss-meta-note">Đa luồng</span>
                    )}
                    {activeItem.transcript?.multi_singer_separation?.available && (
                      <span
                        className="ss-meta-note"
                        title={activeItem.transcript.multi_singer_separation.note || "Đã tách giọng chính và giọng bè / ad-lib"}
                      >
                        Lead / Backing
                      </span>
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
                        if (ytPanelTab === "plain") setYtPanelTab("review");
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

                  {aiRefineResults.length > 0 && (
                    <div className="ss-refine-menu-wrap" ref={aiRefineMenuRef}>
                      <div className={`ss-refine-trigger-group ${aiRefineOpen ? "is-open" : ""}`}>
                        <button
                          type="button"
                          className="ss-refine-trigger-btn"
                          aria-haspopup="dialog"
                          aria-expanded={aiRefineOpen}
                          onClick={() => {
                            setAiRefineOpen(v => !v);
                            setExportMenuOpen(false);
                            setMoreMenuOpen(false);
                          }}
                          title="Xem bảng kết quả AI vừa đọc lại"
                        >
                          <span>✦ Kết quả đọc lại</span>
                          <span className={`ss-refine-badge ${aiRefineImprovedCount > 0 ? "is-improved" : ""}`}>
                            {aiRefineImprovedCount > 0
                              ? `${aiRefineImprovedCount}/${aiRefineResults.length}`
                              : aiRefineResults.length}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="ss-refine-dismiss-btn"
                          aria-label="Xóa thông báo kết quả AI đọc lại"
                          title="Xóa kết quả đọc lại"
                          onClick={() => {
                            setAiRefineResults([]);
                            setAiRefineOpen(false);
                          }}
                        >
                          ×
                        </button>
                      </div>

                      {aiRefineOpen && (
                        <div
                          className="ss-refine-popover"
                          role="dialog"
                          aria-label="Kết quả AI đọc lại"
                        >
                          <div className="ss-refine-popover-head">
                            <div className="ss-refine-popover-title">
                              <strong>✦ Kết quả AI đọc lại</strong>
                              <span>
                                {aiRefineResults.length} câu · {aiRefineImprovedCount} cải thiện
                              </span>
                            </div>
                            <button
                              type="button"
                              className="ss-refine-popover-close"
                              aria-label="Thu gọn bảng kết quả"
                              title="Thu gọn"
                              onClick={() => setAiRefineOpen(false)}
                            >
                              ×
                            </button>
                          </div>

                          <div className="ss-refine-popover-list">
                            {aiRefineResults.map(r => {
                              const seg = draftSegments.find(s => s.id === r.id);
                              const hasConf = r.old_confidence > 0 || r.new_confidence > 0;
                              return (
                                <div
                                  key={r.id}
                                  className={`ss-refine-card ${r.improved ? "is-improved" : ""}`}
                                  title={r.variant || undefined}
                                >
                                  <div className="ss-refine-card-top">
                                    <button
                                      type="button"
                                      className="ss-refine-jump-btn"
                                      onClick={() => jumpToRefinedLine(r.id)}
                                      title={`Đi tới câu #${r.id}`}
                                    >
                                      <span className="ss-refine-id">Câu #{r.id}</span>
                                      {seg && (
                                        <span className="ss-refine-time">{formatClock(seg.start)}</span>
                                      )}
                                    </button>
                                    <div className="ss-refine-card-tags">
                                      {hasConf && (
                                        <span className="ss-refine-conf">
                                          {Math.round(r.old_confidence * 100)}% →{" "}
                                          <strong>{Math.round(r.new_confidence * 100)}%</strong>
                                        </span>
                                      )}
                                      {r.auto_resolved ? (
                                        <span className="ss-status-chip verified">✓ Đạt chuẩn</span>
                                      ) : !r.changed ? (
                                        <span className="ss-refine-kept-tag">Giữ nguyên</span>
                                      ) : null}
                                    </div>
                                  </div>

                                  <button
                                    type="button"
                                    className="ss-refine-diff"
                                    onClick={() => jumpToRefinedLine(r.id)}
                                  >
                                    {r.changed ? (
                                      <>
                                        <span className="ss-refine-old">{r.old_text}</span>
                                        <span className="ss-refine-arrow" aria-hidden="true">→</span>
                                        <span className="ss-refine-new">{r.new_text}</span>
                                      </>
                                    ) : (
                                      <span className="ss-refine-same">{r.new_text}</span>
                                    )}
                                  </button>

                                  {r.changed && (
                                    <div className="ss-refine-card-actions">
                                      <button
                                        type="button"
                                        className="ss-refine-undo-btn"
                                        onClick={() => void undoAiRefineLine(r)}
                                      >
                                        ↶ Hoàn tác
                                      </button>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>

                          <div className="ss-refine-popover-foot">
                            <span>Nhấp vào câu để đi tới dòng</span>
                            <button
                              type="button"
                              className="ss-link-btn"
                              onClick={() => {
                                setAiRefineResults([]);
                                setAiRefineOpen(false);
                              }}
                            >
                              Xóa thông báo
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
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
                        setAiRefineOpen(false);
                      }}
                    >
                      Xuất lời bài hát ▾
                    </button>
                    {exportMenuOpen && (
                      <div className="ss-dropdown-menu ss-export-menu" role="menu" aria-label="Định dạng xuất lời bài hát">
                        <div className="ss-export-policy-box" role="group" aria-label="Chính sách nội dung khi xuất">
                          <span className="ss-export-policy-title">Chính sách xuất TXT/LRC/SRT</span>
                          <div className="ss-export-policy-pills">
                            {(
                              [
                                ["main_only", "Chỉ lời chính"],
                                ["with_adlib", "Gồm ad-lib"],
                                ["merged_labeled", "Gộp có nhãn luồng"],
                                ["per_stream", "Riêng từng luồng"],
                              ] as [ExportPolicy, string][]
                            ).map(([pol, lbl]) => (
                              <button
                                key={pol}
                                type="button"
                                className={`ss-policy-pill ${exportPolicy === pol ? "on" : ""}`}
                                aria-pressed={exportPolicy === pol}
                                disabled={pol === "per_stream" && !exporters.multiStream}
                                onClick={() => setExportPolicy(pol)}
                              >
                                {lbl}
                              </button>
                            ))}
                          </div>
                        </div>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportTxt(true, exportPolicy);
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>TXT — Văn bản lời bài hát</strong>
                          <small>Văn bản thuần UTF-8 theo chính sách đã chọn</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportLrc(exportPolicy);
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
                              exporters.exportEnhancedLrc(exportPolicy);
                              setExportMenuOpen(false);
                            }}
                          >
                            <strong>Enhanced LRC — Đồng bộ từng chữ</strong>
                            <small>Kèm thẻ &lt;mm:ss.xx&gt; theo từng từ (không ép cho humming)</small>
                          </button>
                        )}
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportSrt(exportPolicy);
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>SRT — Phụ đề video</strong>
                          <small>Chuẩn SubRip (hiển thị câu chồng nhau tùy thuộc trình phát)</small>
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            exporters.exportJson({
                              schema_version: activeItem.transcript?.schema_version ?? 2,
                              mode: activeItem.transcript?.mode,
                              alignment_method: activeItem.transcript?.alignment_method,
                              streams: activeItem.transcript?.streams,
                              multi_singer_separation: activeItem.transcript?.multi_singer_separation,
                              raw_asr_segments: activeItem.transcript?.raw_asr_segments,
                              diagnostics: activeItem.transcript?.diagnostics,
                            });
                            setExportMenuOpen(false);
                          }}
                        >
                          <strong>JSON — Dữ liệu đầy đủ (bảo toàn 100%)</strong>
                          <small>Giữ đủ stream, event_type, overlap, raw ASR &amp; provenance</small>
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
                        setAiRefineOpen(false);
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
                          onClick={() => {
                            setMoreMenuOpen(false);
                            setYtLinkDraft(activeItem.youtube_url || "");
                            setYtLinkEditorOpen(v => !v);
                          }}
                        >
                          <strong>{youtubeVideoId ? "📺 Đổi / gỡ link YouTube" : "📺 Gắn link YouTube"}</strong>
                          <small>
                            {youtubeVideoId
                              ? "Thay đổi hoặc tắt giao diện xem video YouTube"
                              : "Mở giao diện xem video YouTube đồng bộ bên cạnh lời hát"}
                          </small>
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

              {ytLinkEditorOpen && (
                <div className="ss-yt-link-box" role="region" aria-label="Gắn liên kết YouTube">
                  <input
                    type="url"
                    value={ytLinkDraft}
                    onChange={e => setYtLinkDraft(e.target.value)}
                    placeholder="Dán link YouTube (https://www.youtube.com/watch?v=... hoặc để trống để gỡ)"
                    aria-label="Đường dẫn video YouTube"
                    onKeyDown={e => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void handleSaveYoutubeLink();
                      }
                    }}
                  />
                  <button
                    type="button"
                    className="ss-primary-btn"
                    disabled={savingYtLink}
                    onClick={() => void handleSaveYoutubeLink()}
                  >
                    {savingYtLink ? "Đang lưu…" : "Áp dụng"}
                  </button>
                  <button
                    type="button"
                    className="ss-secondary-btn"
                    disabled={savingYtLink}
                    onClick={() => setYtLinkEditorOpen(false)}
                  >
                    Hủy
                  </button>
                </div>
              )}

              {/* Local audio element: authoritative in Local File mode & separated Vocal/Solo stem mode; stays paused in YouTube Bản gốc mode */}
              <audio
                key={`${activeItem.id}:${audioSourceMode}`}
                ref={audioRef}
                src={activeAudioSrc}
                preload="metadata"
                onPlay={() => {
                  if (!useYouTubeAudio) {
                    setPlaying(true);
                    if (activeId) setRetainedSessionId(activeId);
                  }
                }}
                onPause={e => {
                  if (!useYouTubeAudio) {
                    setPlaying(false);
                    handleAudioTimeUpdate(e.currentTarget);
                  }
                }}
                onEnded={e => {
                  if (!useYouTubeAudio) {
                    setPlaying(false);
                    handleAudioTimeUpdate(e.currentTarget);
                  }
                }}
                onTimeUpdate={e => {
                  if (!useYouTubeAudio) handleAudioTimeUpdate(e.currentTarget);
                }}
                onSeeking={e => {
                  if (!useYouTubeAudio) handleAudioTimeUpdate(e.currentTarget);
                }}
                onSeeked={e => {
                  if (!useYouTubeAudio) handleAudioTimeUpdate(e.currentTarget);
                }}
                onLoadedMetadata={e => {
                  const loaded = e.currentTarget.duration;
                  setAudioDuration(Number.isFinite(loaded) && loaded > 0 ? loaded : activeItem.duration || 0);
                  if (!useYouTubeAudio) handleAudioTimeUpdate(e.currentTarget);
                }}
                onVolumeChange={e => {
                  const audio = e.currentTarget;
                  setVolume(audio.volume);
                  setMuted(audio.muted);
                }}
              />

              {/* Unified Studio Toolbar above split-screen: Audio Source & Display Toggles */}
              <div className="ss-yt-studio-bar" role="toolbar" aria-label="Công cụ giọng hát và hiển thị">
                <div className="ss-yt-studio-group">
                  <span className="ss-yt-studio-label">Âm thanh</span>
                  <div className="ss-yt-segmented" role="group" aria-label="Nguồn âm thanh">
                    <button
                      type="button"
                      className={`ss-yt-seg-btn ${audioSourceMode === "original" ? "on" : ""}`}
                      aria-pressed={audioSourceMode === "original"}
                      onClick={() => switchAudioSource("original")}
                      title={youtubeVideoId ? "Nghe trực tiếp từ trình phát YouTube gốc" : "Nghe bản thu âm thanh gốc"}
                    >
                      Bản gốc
                    </button>
                    <button
                      type="button"
                      className={`ss-yt-seg-btn ${audioSourceMode === "vocal" ? "on" : ""}`}
                      aria-pressed={audioSourceMode === "vocal"}
                      disabled={!hasVocalStemAudio}
                      onClick={() => switchAudioSource("vocal")}
                      title={
                        hasVocalStemAudio
                          ? youtubeVideoId
                            ? "Tắt tiếng YouTube và phát kênh giọng hát tách riêng (BS-RoFormer)"
                            : "Chỉ phát kênh giọng hát tách riêng (BS-RoFormer)"
                          : "Chưa có kênh giọng hát tách riêng"
                      }
                    >
                      Kênh vocal
                    </button>
                    <button
                      type="button"
                      className={`ss-yt-seg-btn ${audioSourceMode === "instrumental" ? "on" : ""}`}
                      aria-pressed={audioSourceMode === "instrumental"}
                      disabled={!hasInstrumentalAudio}
                      onClick={() => switchAudioSource("instrumental")}
                      title={
                        hasInstrumentalAudio
                          ? youtubeVideoId
                            ? "Tắt tiếng YouTube và chỉ phát nhạc nền không lời (Instrumental)"
                            : "Chỉ phát nhạc nền không lời (Instrumental)"
                          : "Chưa có kênh nhạc nền tách riêng"
                      }
                    >
                      Instrumental
                    </button>
                    {isolatedStreams.map(st => {
                      const mk: AudioSourceMode = `stream:${st.id}`;
                      return (
                        <button
                          key={st.id}
                          type="button"
                          className={`ss-yt-seg-btn ${audioSourceMode === mk ? "on" : ""}`}
                          aria-pressed={audioSourceMode === mk}
                          onClick={() => switchAudioSource(mk)}
                        >
                          Solo {st.label}
                        </button>
                      );
                    })}
                  </div>
                  {audioSourceMode !== "original" && (
                    <>
                      {youtubeVideoId && (
                        <VolumeControl
                          value={muted ? 0 : volume}
                          onChange={changeVolume}
                          className="ss-yt-stem-vol"
                          buttonClassName="ss-yt-stem-vol-btn"
                          label={
                            audioSourceMode === "instrumental"
                              ? "Âm lượng nhạc nền Instrumental"
                              : audioSourceMode === "vocal"
                                ? "Âm lượng kênh vocal"
                                : "Âm lượng kênh đang phát"
                          }
                          sliderLabel={
                            audioSourceMode === "instrumental"
                              ? "Âm lượng Instrumental"
                              : audioSourceMode === "vocal"
                                ? "Âm lượng kênh vocal"
                                : "Âm lượng kênh tách"
                          }
                          onOpen={() => {
                            setExportMenuOpen(false);
                            setMoreMenuOpen(false);
                            setAiRefineOpen(false);
                          }}
                        />
                      )}
                      <span className="ss-yt-stem-hint">
                        {stemLoading && audioSourceMode === "instrumental"
                          ? "Đang tách nhạc nền không lời bằng BS-RoFormer…"
                          : getStemAudioHint(audioSourceMode, volume, muted)}
                      </span>
                    </>
                  )}
                </div>

                <div className="ss-yt-studio-group">
                  <button
                    type="button"
                    className={`ss-yt-toggle-pill ${autoFollow ? "on" : ""}`}
                    aria-pressed={autoFollow}
                    onClick={() => setAutoFollow(v => !v)}
                    title="Tự động cuộn theo câu đang phát"
                  >
                    Tự cuộn
                  </button>
                  {hasAnyChords && (
                    <button
                      type="button"
                      className={`ss-yt-toggle-pill ${showChords ? "on" : ""}`}
                      aria-pressed={showChords}
                      onClick={() => setShowChords(v => !v)}
                      title="Hiện/ẩn hợp âm trên câu hát"
                    >
                      Hợp âm
                    </button>
                  )}
                  {hasWordTimings && (
                    <button
                      type="button"
                      className={`ss-yt-toggle-pill ${showWordHighlight ? "on" : ""}`}
                      aria-pressed={showWordHighlight}
                      onClick={() => setShowWordHighlight(v => !v)}
                      title="Tô sáng từng từ khi đang phát"
                    >
                      Tô sáng từ
                    </button>
                  )}
                  <button
                    type="button"
                    className={`ss-yt-toggle-pill ${loopSelectedLine ? "on" : ""}`}
                    aria-pressed={loopSelectedLine}
                    onClick={() => {
                      const next = !loopSelectedLine;
                      setLoopSelectedLine(next);
                      if (next) {
                        const target =
                          selectedSegment ??
                          draftSegments.find(s => s.id === playingSegmentId) ??
                          draftSegments[0] ??
                          null;
                        if (target) {
                          setSelectedSegId(target.id);
                          seekAudio(target.start, true);
                        }
                      }
                    }}
                    title="Lặp lại câu đang chọn (L)"
                  >
                    ⟳ {loopSelectedLine ? "Đang lặp câu" : "Lặp câu"}
                  </button>
                </div>
              </div>

              {/* Unified Split-Screen Layout (YouTube Player or Audio Theater Left, Tabbed Synced Lyrics Right) */}
              <div className="ss-yt-layout">
                <div className="ss-yt-stage-col">
                  {youtubeVideoId ? (
                    <YouTubeStagePlayer
                      key={youtubeVideoId}
                      ref={ytPlayerRef}
                      videoId={youtubeVideoId}
                      title={activeItem.title}
                      muteYouTube={audioSourceMode !== "original"}
                      externalPosition={position}
                      externalPlaying={playing}
                      isBackground={!isDetailVisible}
                      externalVolume={volume}
                      externalMuted={muted}
                      playbackRate={playbackRate}
                      showChords={showChords}
                      showWordHighlight={showWordHighlight}
                      activeSegments={activePlayingSegments}
                      karaokeWordsBySegId={karaokeWordsBySegId}
                      syncTime={highlightTime}
                      onYouTubeTimeUpdate={handleYouTubeTimeUpdate}
                      onYouTubePlayingChange={handleYouTubePlayingChange}
                      onYouTubeDurationChange={d => {
                        if (d > 0) setAudioDuration(d);
                      }}
                      onYouTubeRateChange={handleYouTubeRateChange}
                      onYouTubeManualSeek={handleYouTubeManualSeek}
                      onYouTubeVolumeChange={handleYouTubeVolumeChange}
                    />
                  ) : (
                    <div className="ss-yt-stage ss-audio-stage" aria-label="Sân khấu phát âm thanh đồng bộ">
                      <div className="ss-audio-stage-head">
                        <span className="ss-audio-badge">
                          <span className={`ss-audio-badge-dot ${playing ? "is-playing" : ""}`} aria-hidden="true" />
                          <span>{bgChannelLabel}</span>
                        </span>
                        <button
                          type="button"
                          className="ss-audio-yt-btn"
                          onClick={() => {
                            setYtLinkDraft(activeItem.youtube_url || "");
                            setYtLinkEditorOpen(v => !v);
                          }}
                          title="Gắn link YouTube để xem video đồng bộ"
                        >
                          📺 Gắn YouTube
                        </button>
                      </div>

                      <div className="ss-audio-stage-body" onClick={toggleLocalAudioPlay}>
                        <div className={`ss-audio-eq ${playing ? "is-playing" : ""}`} aria-hidden="true">
                          <span />
                          <span />
                          <span />
                          <span />
                          <span />
                          <span />
                          <span />
                          <span />
                          <span />
                        </div>

                        {activePlayingSegments.length > 0 ? (
                          <div className="ss-audio-lyrics-stack" aria-live="polite">
                            {activePlayingSegments.slice(0, 2).map((seg, idx) => {
                              const evType = getEffectiveEventType(seg);
                              const karaokeWords = karaokeWordsBySegId.get(seg.id) ?? seg.words ?? [];
                              const useWordKaraoke =
                                showWordHighlight &&
                                evType !== "humming" &&
                                evType !== "unknown_vocal" &&
                                karaokeWords.length > 0;
                              return (
                                <div key={seg.id} className={`ss-audio-lyric-line ${idx > 0 ? "is-secondary" : ""}`}>
                                  {showChords && seg.chord && (
                                    <span className="ss-yt-sub-chord">{seg.chord}</span>
                                  )}
                                  <p className="ss-audio-lyric-text">
                                    {useWordKaraoke
                                      ? karaokeWords.map((w, wi) => {
                                          const nextWordStart =
                                            wi + 1 < karaokeWords.length ? karaokeWords[wi + 1].start : null;
                                          const wordState = getWordPlaybackState(w, highlightTime, nextWordStart);
                                          const cls =
                                            wordState === "active"
                                              ? "active"
                                              : wordState === "sung"
                                                ? "passed"
                                                : "";
                                          return (
                                            <span key={wi} className={`ss-yt-word ${cls}`}>
                                              {w.word}
                                            </span>
                                          );
                                        })
                                      : seg.text}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="ss-audio-idle">
                            <span
                              className={`ss-audio-center-play ${playing ? "is-playing" : ""}`}
                              aria-hidden="true"
                            >
                              {playing ? "♪" : <Icon name="play" color={colors.onPrimary} size={26} />}
                            </span>
                            <h3 className="ss-audio-idle-title">{activeItem.title}</h3>
                            <p className="ss-audio-idle-sub">
                              {playing
                                ? "Đoạn nhạc không lời…"
                                : "Nhấn vào màn hình hoặc phím Space để phát"}
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="ss-audio-bar" onClick={e => e.stopPropagation()}>
                        <input
                          className="ss-audio-scrubber"
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
                        <div className="ss-audio-controls-row">
                          <div className="ss-audio-controls-group">
                            <button
                              type="button"
                              className="ss-audio-btn"
                              onClick={() => seekAudio(currentTime - 5)}
                              title="Lùi 5 giây"
                              aria-label="Lùi 5 giây"
                            >
                              <Icon name="rewind5" color={colors.instrumentText} size={19} />
                            </button>
                            <button
                              type="button"
                              className="ss-audio-play-btn"
                              onClick={toggleLocalAudioPlay}
                              aria-label={playing ? "Tạm dừng (Space)" : "Phát bài hát (Space)"}
                              title={playing ? "Tạm dừng (Space)" : "Phát bài hát (Space)"}
                            >
                              <Icon name={playing ? "pause" : "play"} color={colors.onPrimary} size={18} />
                            </button>
                            <button
                              type="button"
                              className="ss-audio-btn"
                              onClick={() => seekAudio(currentTime + 5)}
                              title="Tua tới 5 giây"
                              aria-label="Tua tới 5 giây"
                            >
                              <Icon name="forward5" color={colors.instrumentText} size={19} />
                            </button>
                            <VolumeControl
                              value={muted ? 0 : volume}
                              onChange={changeVolume}
                              placement="top"
                              iconColor={colors.instrumentText}
                              className="ss-audio-vol"
                              buttonClassName="ss-audio-btn"
                              label="Âm lượng trình phát"
                              sliderLabel="Âm lượng"
                              onOpen={() => {
                                setExportMenuOpen(false);
                                setMoreMenuOpen(false);
                                setAiRefineOpen(false);
                              }}
                            />
                            <span className="ss-audio-clock">
                              {formatClock(position)} / {formatClock(duration)}
                            </span>
                          </div>
                          <div className="ss-audio-controls-group">
                            <button
                              type="button"
                              className="ss-audio-speed-btn"
                              onClick={cyclePlaybackSpeed}
                              title="Nhấn để đổi tốc độ phát (0.5× – 1.5×)"
                              aria-label={`Tốc độ phát ${playbackRate} lần`}
                            >
                              {playbackRate}×
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Right Tabbed Lyrics Panel (Shared by YouTube & Local Audio) */}
                <aside className="ss-yt-panel-col" aria-label="Danh sách lời bài hát đồng bộ">
                  <div className="ss-yt-tabs" role="tablist" aria-label="Chế độ bảng lời">
                    <button
                      type="button"
                      role="tab"
                      className={`ss-yt-tab ${ytPanelTab === "lyrics" ? "active" : ""}`}
                      aria-selected={ytPanelTab === "lyrics"}
                      onClick={() => {
                        setYtPanelTab("lyrics");
                        setReviewFilter("all");
                      }}
                    >
                      <span>Lời hát</span>
                      <span className="ss-yt-tab-count">{draftSegments.length}</span>
                    </button>
                    <button
                      type="button"
                      role="tab"
                      className={`ss-yt-tab ${ytPanelTab === "review" ? "active" : ""}`}
                      aria-selected={ytPanelTab === "review"}
                      onClick={() => setYtPanelTab("review")}
                    >
                      <span>Rà soát</span>
                      <span className="ss-yt-tab-count">{reviewStats.needsReview}</span>
                    </button>
                    <button
                      type="button"
                      role="tab"
                      className={`ss-yt-tab ${ytPanelTab === "plain" ? "active" : ""}`}
                      aria-selected={ytPanelTab === "plain"}
                      onClick={() => setYtPanelTab("plain")}
                    >
                      <span>Văn bản</span>
                    </button>
                  </div>

                  {ytPanelTab !== "plain" && (
                    <div className="ss-yt-panel-tools">
                      <div className="ss-yt-panel-search-row">
                        <input
                          className="songscription-search ss-search-input"
                          value={searchQuery}
                          onChange={e => setSearchQuery(e.target.value)}
                          placeholder="Tìm câu hát, hợp âm…"
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
                      </div>

                      {ytPanelTab === "review" && (
                        <div className="ss-yt-filter-strip" role="tablist" aria-label="Bộ lọc rà soát">
                          {(
                            [
                              ["all", "Tất cả", reviewStats.total],
                              ["needs_review", "Cần kiểm tra", reviewStats.needsReview],
                              ["adlib", "Ad-lib", categoryCounts.adlib],
                              ["overlap", "Chồng giọng", categoryCounts.overlap],
                              ["edited", "Đã sửa", reviewStats.edited],
                            ] as [ReviewFilter, string, number][]
                          ).map(([key, label, count]) => (
                            <button
                              key={key}
                              type="button"
                              role="tab"
                              className="ss-yt-filter-pill"
                              aria-selected={reviewFilter === key}
                              onClick={() => setReviewFilter(key)}
                            >
                              {label} ({count})
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <div
                    className={`ss-yt-panel-scroll ${ytPanelTab === "plain" ? "is-plain" : ""}`}
                    ref={listScrollRef}
                    onWheel={() => {
                      if (playing && autoFollow) setAutoFollow(false);
                    }}
                  >
                    {!draftSegments.length ? (
                      <div className="songscription-no-lyrics">
                        <p>Chưa có dòng lời nào cho bài hát này.</p>
                        <button type="button" className="ss-secondary-btn" onClick={() => insertLineAfter(-1)}>
                          ＋ Thêm dòng lời
                        </button>
                      </div>
                    ) : ytPanelTab === "plain" ? (
                      <div className="songscription-plain-text">
                        {draftSegments
                          .map(s => (s.section ? `[${s.section}]\n${s.text.trim()}` : s.text.trim()))
                          .filter(Boolean)
                          .join("\n")}
                      </div>
                    ) : !filteredSegments.length ? (
                      <div className="songscription-no-lyrics">
                        <p>Không có câu hát nào khớp bộ lọc.</p>
                        <button
                          type="button"
                          className="ss-secondary-btn"
                          onClick={() => {
                            setReviewFilter("all");
                            setSearchQuery("");
                          }}
                        >
                          Xem tất cả ({draftSegments.length})
                        </button>
                      </div>
                    ) : (
                      filteredSegments.map(seg => {
                        const realIdx = segIndexById.get(seg.id) ?? -1;
                        const karaokeWords = karaokeWordsBySegId.get(seg.id) ?? seg.words ?? [];
                        const isPlayingRow = playingSegmentIds.has(seg.id);
                        const isEditingRow = seg.id === ytEditingSegId;
                        const status = getEffectiveReviewStatus(seg);
                        const evType = getEffectiveEventType(seg);

                        return (
                          <div
                            key={seg.id}
                            data-seg-id={seg.id}
                            className={`ss-yt-row ${isPlayingRow ? "is-playing" : ""} ${isEditingRow ? "is-selected" : ""} event-${evType} ${seg.is_adlib ? "is-adlib" : ""}`}
                            aria-current={isPlayingRow ? "true" : undefined}
                          >
                            <div className="ss-yt-row-head">
                              <span className={`ss-yt-dot status-${status}`} aria-hidden="true" />
                              <div
                                className="ss-yt-row-content"
                                role="button"
                                tabIndex={0}
                                onClick={() => selectAndSeekLine(seg, true)}
                                onKeyDown={e => {
                                  if (e.key === "Enter" || e.key === " ") {
                                    e.preventDefault();
                                    selectAndSeekLine(seg, true);
                                  }
                                }}
                              >
                                <div className="ss-yt-row-meta">
                                  <span className="ss-yt-row-time">{formatClock(seg.start)}</span>
                                  {seg.section && <span>[{seg.section}]</span>}
                                  {showChords && seg.chord && <span className="line-chord-tag">{seg.chord}</span>}
                                  {evType === "vocable" && (
                                    <span className="ss-event-pill vocable">Vocalise</span>
                                  )}
                                  {evType === "humming" && (
                                    <span className="ss-event-pill humming">Humming</span>
                                  )}
                                  {evType === "unknown_vocal" && (
                                    <span className="ss-event-pill unknown">Nghi có vocal</span>
                                  )}
                                  {evType === "lyric" && seg.is_adlib === true && (
                                    <span className="ss-event-pill adlib">Ad-lib</span>
                                  )}
                                </div>
                                <div className="ss-yt-row-text">
                                  {showWordHighlight &&
                                  evType !== "humming" &&
                                  evType !== "unknown_vocal" &&
                                  karaokeWords.length > 0
                                    ? karaokeWords.map((w, wi) => {
                                        const nextWordStart =
                                          wi + 1 < karaokeWords.length ? karaokeWords[wi + 1].start : null;
                                        const wordState = getWordPlaybackState(w, highlightTime, nextWordStart);
                                        return (
                                          <span key={wi} className={`ss-lyric-word word-${wordState}`}>
                                            {w.word}
                                          </span>
                                        );
                                      })
                                    : seg.text || (
                                        <em className="ss-empty-line-hint">
                                          {evType === "unknown_vocal"
                                            ? "(Nghi có giọng hát — nhấp để nghe đối chiếu)"
                                            : "(Dòng trống)"}
                                        </em>
                                      )}
                                </div>
                              </div>
                              <button
                                type="button"
                                className="ss-yt-row-edit-btn"
                                aria-label={isEditingRow ? "Đóng chỉnh sửa câu này" : "Chỉnh sửa câu này"}
                                title={isEditingRow ? "Đóng chỉnh sửa" : "Chỉnh sửa câu này"}
                                onClick={() => {
                                  if (isEditingRow) {
                                    setYtEditingSegId(null);
                                  } else {
                                    setYtEditingSegId(seg.id);
                                    setSelectedSegId(seg.id);
                                  }
                                }}
                              >
                                {isEditingRow ? "▴" : "✎"}
                              </button>
                            </div>

                            {isEditingRow && (
                              <div className="ss-inline-editor">
                                <div className="ss-inline-text-wrap">
                                  <input
                                    className="ss-inline-text-input"
                                    value={seg.text}
                                    onChange={e => updateLineField(seg.id, { text: e.target.value })}
                                    aria-label={`Sửa câu hát ${seg.id}`}
                                  />
                                </div>
                                <div className="ss-inline-footer">
                                  <div className="ss-inline-review-btns">
                                    <button
                                      type="button"
                                      className="ss-confirm-line-btn"
                                      disabled={savingAll}
                                      onClick={() => void markLineReviewStatus(seg.id, "verified", true)}
                                    >
                                      ✓ Đã kiểm tra
                                    </button>
                                    <button
                                      type="button"
                                      className="ss-ai-reread-line-btn"
                                      disabled={savingAll || retranscribingTarget !== null}
                                      onClick={() => void handleRetranscribeFlagged([seg.id], seg.language ?? undefined)}
                                    >
                                      {retranscribingTarget === seg.id ? "⏳ Đang đọc…" : "✦ AI đọc lại"}
                                    </button>
                                    {dirty && (
                                      <button
                                        type="button"
                                        className="ss-save-inline-btn"
                                        disabled={savingAll}
                                        onClick={() => void saveAllDraftSegments()}
                                      >
                                        Lưu
                                      </button>
                                    )}
                                  </div>
                                  <div className="ss-inline-struct-btns">
                                    <button type="button" onClick={() => splitLineAt(realIdx)}>
                                      ✂ Tách
                                    </button>
                                    <button
                                      type="button"
                                      disabled={realIdx >= draftSegments.length - 1}
                                      onClick={() => mergeWithNext(realIdx)}
                                    >
                                      ⊕ Gộp
                                    </button>
                                    <button
                                      type="button"
                                      className="danger"
                                      disabled={draftSegments.length <= 1}
                                      onClick={() => deleteLineAt(realIdx)}
                                    >
                                      × Xóa
                                    </button>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </aside>
              </div>
            </div>
          )}
        </section>
      </div>
        </div>
      )}
    </>
  );
}
