import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  api,
  type MidiRepairDiffNote,
  type MidiRepairIssue,
  type MidiRepairIssueType,
  type MidiRepairManualUpdate,
  type MidiRepairOptions,
  type MidiRepairPreviewResponse,
  type ProjectDetail,
  type Take,
} from "../api";
import { Icon } from "../../../components/ui/StudioIcon";
import { takeLabel } from "../../../components/studio/CreatePanel";
import { usePlayerState } from "../../../components/piano/PlayerBar";
import { useT } from "../../../i18n";
import { formatTime } from "../utils/params";
import type { Player } from "../utils/player";
import "../styles/smart-midi-repair.css";

type PresetId = "gentle" | "concert" | "playable" | "score" | "custom";
type AuditionMode = "after" | "before";
type SideTab = "controls" | "issues";

type ManualState = {
  deletes: number[];
  keeps: number[];
  updates: Record<number, MidiRepairManualUpdate>;
};

const EMPTY_MANUAL: ManualState = { deletes: [], keeps: [], updates: {} };

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const BLACK_PITCH_CLASSES = [1, 3, 6, 8, 10];
const MIN_PITCH = 21;
const MAX_PITCH = 108;
const CHORD_H = 22;
const PEDAL_H = 12;
const CANVAS_FONT = `system-ui, "Segoe UI", sans-serif`;

function noteName(pitch: number): string {
  return NOTE_NAMES[((pitch % 12) + 12) % 12] + (Math.floor(pitch / 12) - 1);
}

function spanLabel(semitones: number): string {
  const map: Record<number, string> = {
    12: "Quãng 8",
    13: "Quãng 9b",
    14: "Quãng 9",
    15: "Quãng 10b",
    16: "Quãng 10",
    17: "Quãng 11",
    18: "Quãng 11+",
  };
  return map[semitones] ?? `${semitones} nửa cung`;
}

const DEFAULT_OPTIONS: MidiRepairOptions = {
  remove_ghost_notes: true,
  min_duration_ms: 35,
  min_velocity: 18,
  fix_overlaps: true,
  duplicate_window_ms: 30,
  rearticulate_gap_ms: 15,
  clamp_piano_range: true,
  fix_hand_span: true,
  split_pitch: 60,
  max_hand_span: 15,
  max_polyphony: 10,
  smart_quantize: true,
  quantize_strength: 0.5,
  grid_subdivision: "16th",
  normalize_velocity: true,
  vel_min: 34,
  vel_max: 114,
  melody_boost: 8,
  clean_pedals: true,
  max_pedal_sec: 4.0,
  repedal_on_chord: true,
};

const PRESETS: Record<Exclude<PresetId, "custom">, { label: string; tip: string; options: MidiRepairOptions }> = {
  gentle: {
    label: "Nhẹ",
    tip: "Chỉ lọc nốt rác & trùng phím",
    options: {
      ...DEFAULT_OPTIONS,
      min_duration_ms: 28,
      min_velocity: 15,
      fix_hand_span: false,
      smart_quantize: false,
      quantize_strength: 0,
      normalize_velocity: false,
      clean_pedals: false,
    },
  },
  concert: {
    label: "Biểu diễn",
    tip: "Cân bằng toàn diện",
    options: { ...DEFAULT_OPTIONS },
  },
  playable: {
    label: "Dễ chơi",
    tip: "Thu gọn tầm tay tối đa quãng 8",
    options: {
      ...DEFAULT_OPTIONS,
      min_duration_ms: 42,
      min_velocity: 22,
      max_hand_span: 12,
      max_polyphony: 8,
      quantize_strength: 0.6,
      vel_min: 38,
      vel_max: 112,
      melody_boost: 12,
      max_pedal_sec: 3.2,
    },
  },
  score: {
    label: "Xuất Sheet",
    tip: "Căn nhịp chặt để xuất MusicXML",
    options: {
      ...DEFAULT_OPTIONS,
      min_duration_ms: 40,
      min_velocity: 20,
      rearticulate_gap_ms: 20,
      max_hand_span: 14,
      max_polyphony: 8,
      smart_quantize: true,
      quantize_strength: 0.82,
      grid_subdivision: "16th",
      melody_boost: 6,
      max_pedal_sec: 3.5,
    },
  },
};

const ISSUE_SHORT: Record<MidiRepairIssueType, string> = {
  ghost_notes: "Nốt rác",
  overlapping_notes: "Trùng phím",
  hand_span: "Tầm tay",
  timing_jitter: "Lệch nhịp",
  pedal_glitches: "Pedal",
  velocity_spikes: "Sắc thái",
  out_of_range: "Ngoài 88 phím",
};

type Props = {
  pid: string;
  project: ProjectDetail;
  selectedTake: Take | null;
  player: Player;
  onSelectTake: (tid: string) => void;
  onProjectUpdated: (detail: ProjectDetail) => void;
  onSwitchToPiano: () => void;
};

export function SmartMidiRepair({
  pid,
  project,
  selectedTake,
  player,
  onSelectTake,
  onProjectUpdated,
  onSwitchToPiano,
}: Props) {
  usePlayerState(player);
  const t = useT();

  const doneTakes = useMemo(
    () => (project.takes ?? []).filter(tk => tk.state === "done"),
    [project.takes],
  );
  const hasSource = Boolean(project.source);
  const sourceCount = doneTakes.length + (hasSource ? 1 : 0);

  const [target, setTarget] = useState<"take" | "source">(() =>
    doneTakes.length > 0 ? "take" : "source",
  );
  const [targetTakeId, setTargetTakeId] = useState<string | null>(() => {
    if (selectedTake?.state === "done") return selectedTake.id;
    return doneTakes[0]?.id ?? null;
  });

  const effectiveTarget: "take" | "source" =
    target === "take" && doneTakes.length === 0 && hasSource ? "source" : target;
  const effectiveTakeId =
    effectiveTarget === "take"
      ? (doneTakes.some(tk => tk.id === targetTakeId) ? targetTakeId : doneTakes[0]?.id ?? null)
      : null;

  const [presetId, setPresetId] = useState<PresetId>("concert");
  const [options, setOptions] = useState<MidiRepairOptions>(() => ({ ...PRESETS.concert.options }));
  const [auditionMode, setAuditionMode] = useState<AuditionMode>("after");
  const [sideTab, setSideTab] = useState<SideTab>("controls");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);

  const [focusedIssueTime, setFocusedIssueTime] = useState<number | null>(null);
  const [selectedNoteIdx, setSelectedNoteIdx] = useState<number | null>(null);

  const [manual, setManual] = useState<ManualState>(EMPTY_MANUAL);
  const [manualHistory, setManualHistory] = useState<ManualState[]>([]);

  const [preview, setPreview] = useState<MidiRepairPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState<null | "new_take" | "overwrite" | "revert">(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; showGoPiano?: boolean } | null>(null);

  const flashToast = useCallback((message: string, showGoPiano = false) => {
    setToast({ message, showGoPiano });
    window.setTimeout(() => {
      setToast(cur => (cur?.message === message ? null : cur));
    }, 4200);
  }, []);

  useEffect(() => {
    if (!moreOpen) return;
    const onDown = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) setMoreOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [moreOpen]);

  const resetManualEdits = useCallback(() => {
    setManual(EMPTY_MANUAL);
    setManualHistory([]);
    setSelectedNoteIdx(null);
    setFocusedIssueTime(null);
  }, []);

  const hasValidTarget =
    (effectiveTarget === "take" && Boolean(effectiveTakeId)) || (effectiveTarget === "source" && hasSource);
  const activePreview = hasValidTarget ? preview : null;
  const manualUpdateList = useMemo(() => Object.values(manual.updates), [manual.updates]);

  const reqSeqRef = useRef(0);
  useEffect(() => {
    if (!hasValidTarget) return;
    const seq = ++reqSeqRef.current;
    const timer = window.setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.previewMidiRepair(pid, {
          target: effectiveTarget,
          take_id: effectiveTakeId,
          options,
          manual_deletes: manual.deletes,
          manual_keeps: manual.keeps,
          manual_updates: manualUpdateList,
        });
        if (seq === reqSeqRef.current) setPreview(res);
      } catch (e) {
        if (seq === reqSeqRef.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (seq === reqSeqRef.current) setLoading(false);
      }
    }, 120);
    return () => window.clearTimeout(timer);
  }, [pid, hasValidTarget, effectiveTarget, effectiveTakeId, options, manual.deletes, manual.keeps, manualUpdateList]);

  useEffect(() => {
    if (!activePreview) return;
    player.setNotes(auditionMode === "after" ? activePreview.after.notes : activePreview.before.notes);
  }, [activePreview, auditionMode, player]);

  const previewTone = useCallback(
    (pitch: number, velocity = 84) => {
      const clamped = Math.max(MIN_PITCH, Math.min(MAX_PITCH, pitch));
      if (player.startManualNote(-900 - clamped, clamped, velocity)) {
        window.setTimeout(() => player.stopManualNote(-900 - clamped), 220);
      }
    },
    [player],
  );

  const selectedNote = useMemo(() => {
    if (selectedNoteIdx == null || !activePreview) return null;
    return activePreview.diff.notes.find(n => n.idx === selectedNoteIdx) ?? null;
  }, [selectedNoteIdx, activePreview]);

  const commitManual = useCallback((updater: (cur: ManualState) => ManualState) => {
    setManual(cur => {
      setManualHistory(hist => [...hist.slice(-29), cur]);
      return updater(cur);
    });
  }, []);

  const handleToggleNoteDelete = useCallback(
    (note: MidiRepairDiffNote) => {
      commitManual(cur => {
        if (note.status === "manual_deleted") {
          return { ...cur, deletes: cur.deletes.filter(id => id !== note.idx) };
        }
        if (note.status === "removed") {
          return {
            ...cur,
            keeps: cur.keeps.includes(note.idx) ? cur.keeps : [...cur.keeps, note.idx],
            deletes: cur.deletes.filter(id => id !== note.idx),
          };
        }
        const nextUpdates = { ...cur.updates };
        delete nextUpdates[note.idx];
        return {
          deletes: cur.deletes.includes(note.idx) ? cur.deletes : [...cur.deletes, note.idx],
          keeps: cur.keeps.filter(id => id !== note.idx),
          updates: nextUpdates,
        };
      });
    },
    [commitManual],
  );

  const handlePatchSelectedNote = useCallback(
    (note: MidiRepairDiffNote, patch: Partial<Omit<MidiRepairManualUpdate, "idx">>, audition = false) => {
      const nextPitch = patch.pitch != null ? Math.max(MIN_PITCH, Math.min(MAX_PITCH, patch.pitch)) : note.pitch;
      const nextDur = patch.duration != null ? Math.max(0.05, Math.min(8.0, Number(patch.duration.toFixed(3)))) : note.duration;
      commitManual(cur => ({
        deletes: cur.deletes.filter(id => id !== note.idx),
        keeps: cur.keeps.includes(note.idx) ? cur.keeps : [...cur.keeps, note.idx],
        updates: {
          ...cur.updates,
          [note.idx]: {
            ...(cur.updates[note.idx] ?? { idx: note.idx }),
            idx: note.idx,
            pitch: nextPitch,
            duration: nextDur,
          },
        },
      }));
      if (audition) previewTone(nextPitch, note.velocity);
    },
    [commitManual, previewTone],
  );

  const undoManual = useCallback(() => {
    setManualHistory(hist => {
      if (!hist.length) return hist;
      const last = hist[hist.length - 1];
      setManual(last);
      return hist.slice(0, -1);
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      if ((e.key === "Delete" || e.key === "Backspace") && selectedNote) {
        e.preventDefault();
        handleToggleNoteDelete(selectedNote);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && manualHistory.length > 0) {
        e.preventDefault();
        undoManual();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedNote, handleToggleNoteDelete, manualHistory.length, undoManual]);

  const applyPreset = (id: Exclude<PresetId, "custom">) => {
    setPresetId(id);
    setOptions({ ...PRESETS[id].options });
  };

  const patchOptions = (patch: Partial<MidiRepairOptions>) => {
    setPresetId("custom");
    setOptions(prev => ({ ...prev, ...patch }));
  };

  const handleJumpToIssue = (issue: MidiRepairIssue) => {
    setFocusedIssueTime(issue.time);
    if (issue.note_idx != null) setSelectedNoteIdx(issue.note_idx);
    const start = Math.max(0, issue.time - 0.8);
    const end = Math.min(player.duration || issue.time + 2.4, issue.time + 2.2);
    if (end - start >= 0.6) player.setLoop([start, end]);
    player.seek(start);
    if (!player.playing) player.play(start);
  };

  const handleApply = async (mode: "new_take" | "overwrite") => {
    if (!activePreview || applying) return;
    setMoreOpen(false);
    setApplying(mode);
    setError(null);
    try {
      const save_mode =
        mode === "new_take"
          ? "new_take"
          : effectiveTarget === "source"
            ? "overwrite_source"
            : "overwrite_take";
      const res = await api.applyMidiRepair(pid, {
        target: effectiveTarget,
        take_id: effectiveTakeId,
        save_mode,
        options,
        manual_deletes: manual.deletes,
        manual_keeps: manual.keeps,
        manual_updates: manualUpdateList,
      });
      onProjectUpdated(res.project);
      if (res.created_take_id) {
        onSelectTake(res.created_take_id);
        setTarget("take");
        setTargetTakeId(res.created_take_id);
        resetManualEdits();
        flashToast("Đã lưu thành Take mới!", true);
      } else {
        resetManualEdits();
        const fresh = await api.previewMidiRepair(pid, { target: effectiveTarget, take_id: effectiveTakeId, options });
        setPreview(fresh);
        flashToast("Đã ghi đè trực tiếp (đã sao lưu bản gốc để khôi phục).");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(null);
    }
  };

  const handleRevert = async () => {
    if (!activePreview?.has_backup || applying) return;
    setMoreOpen(false);
    setApplying("revert");
    setError(null);
    try {
      const res = await api.revertMidiRepair(pid, { target: effectiveTarget, take_id: effectiveTakeId });
      onProjectUpdated(res.project);
      resetManualEdits();
      const fresh = await api.previewMidiRepair(pid, { target: effectiveTarget, take_id: effectiveTakeId, options });
      setPreview(fresh);
      flashToast("Đã khôi phục bản MIDI gốc.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(null);
    }
  };

  const before = activePreview?.before;
  const after = activePreview?.after;
  const diff = activePreview?.diff;
  const scoreDelta = before && after ? after.scores.overall - before.scores.overall : 0;
  const manualEditCount = manual.deletes.length + manual.keeps.length + manualUpdateList.length;

  if (!hasSource && doneTakes.length === 0) {
    return (
      <div className="smr-empty-state">
        <div className="smr-empty-badge" aria-hidden="true">✦</div>
        <h2>Chưa có MIDI để sửa</h2>
        <p>Hãy hoàn tất trích xuất MIDI hoặc tạo ít nhất một Take piano cover.</p>
        <button type="button" className="btn primary" onClick={onSwitchToPiano}>
          Quay lại Piano &amp; Takes
        </button>
      </div>
    );
  }

  return (
    <div className="smr-workspace">
      {toast && (
        <div className="smr-toast" role="status" aria-live="polite">
          <span>{toast.message}</span>
          {toast.showGoPiano && (
            <button type="button" className="smr-toast-link" onClick={onSwitchToPiano}>
              Mở trong Piano &amp; Takes →
            </button>
          )}
        </div>
      )}

      {/* Single Clean Action Bar */}
      <div className="smr-topbar" role="toolbar" aria-label="Điều khiển Smart MIDI Repair">
        <div className="smr-topbar-group">
          {sourceCount > 1 && (
            <select
              className="smr-target-select"
              aria-label="Chọn nguồn MIDI"
              value={effectiveTarget === "source" ? "__source__" : effectiveTakeId ?? ""}
              onChange={e => {
                const val = e.target.value;
                resetManualEdits();
                if (val === "__source__") setTarget("source");
                else {
                  setTarget("take");
                  setTargetTakeId(val);
                  onSelectTake(val);
                }
              }}
            >
              {doneTakes.map(tk => (
                <option key={tk.id} value={tk.id}>
                  {takeLabel(tk, t)}
                </option>
              ))}
              {hasSource && <option value="__source__">MIDI gốc</option>}
            </select>
          )}

          <div className="smr-ab-group" role="group" aria-label="Nghe thử A/B">
            <button
              type="button"
              className={`smr-seg-btn ${auditionMode === "before" ? "on" : ""}`}
              onClick={() => setAuditionMode("before")}
            >
              Nghe gốc
            </button>
            <button
              type="button"
              className={`smr-seg-btn ${auditionMode === "after" ? "on" : ""}`}
              onClick={() => setAuditionMode("after")}
            >
              Nghe sau sửa
            </button>
          </div>

          <div className="smr-diff-legend" aria-label="Chú giải màu">
            <span><i className="dot mod" /> Sửa ({diff?.modified_count ?? 0})</span>
            <span><i className="dot rem" /> Xóa ({diff?.removed_count ?? 0})</span>
          </div>

          {manualEditCount > 0 && (
            <button
              type="button"
              className="smr-link-btn"
              disabled={!manualHistory.length}
              onClick={undoManual}
            >
              ↶ Hoàn tác ({manualEditCount})
            </button>
          )}
          {loading && <span className="spinner small" aria-label="Đang cập nhật…" />}
        </div>

        <div className="smr-topbar-group">
          <button
            type="button"
            className="btn primary small"
            disabled={!activePreview || applying !== null}
            onClick={() => void handleApply("new_take")}
          >
            <Icon name="sparkle" size={14} />
            {applying === "new_take" ? "Đang lưu…" : "Lưu thành Take mới"}
          </button>

          <div className="studio-menu-wrap" ref={moreRef}>
            <button
              type="button"
              className="studio-menu-btn"
              aria-label="Tùy chọn lưu khác"
              aria-expanded={moreOpen}
              onClick={() => setMoreOpen(v => !v)}
            >
              ⋯
            </button>
            {moreOpen && (
              <div className="studio-menu-dropdown" role="menu">
                <button
                  type="button"
                  role="menuitem"
                  disabled={!activePreview || applying !== null}
                  onClick={() => void handleApply("overwrite")}
                >
                  Ghi đè trực tiếp (tự sao lưu gốc)
                </button>
                {activePreview?.has_backup && (
                  <button
                    type="button"
                    role="menuitem"
                    disabled={applying !== null}
                    onClick={() => void handleRevert()}
                  >
                    ↺ Khôi phục bản MIDI gốc
                  </button>
                )}
                {manualEditCount > 0 && (
                  <button type="button" role="menuitem" onClick={() => { setMoreOpen(false); resetManualEdits(); }}>
                    Bỏ toàn bộ sửa thủ công
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="alert" role="alert" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      {/* Clean 2-Column Split: Direct-Edit Full Piano Roll Left | Minimal 5-Toggle Panel Right */}
      <div className="smr-layout">
        <div className="smr-stage-col">
          <div className="smr-roll-card">
            <RepairPianoRoll
              player={player}
              preview={activePreview}
              auditionMode={auditionMode}
              focusTime={focusedIssueTime}
              selectedNoteIdx={selectedNoteIdx}
              onSelectNote={note => {
                if (!note) {
                  setSelectedNoteIdx(null);
                  return;
                }
                setSelectedNoteIdx(note.idx);
                previewTone(note.pitch, note.velocity);
              }}
            />

            {/* Minimal Selected-Note Bar (appears ONLY when a note is clicked) */}
            {selectedNote && (
              <div className="smr-note-editor" role="region" aria-label="Chỉnh nốt đang chọn">
                <div className="smr-note-id">
                  <strong>{noteName(selectedNote.pitch)}</strong>
                  <span className="mono">{formatTime(selectedNote.onset)}</span>
                </div>

                <div className="smr-note-controls">
                  <div className="smr-stepper">
                    <span>Cao độ</span>
                    <button
                      type="button"
                      onClick={() => handlePatchSelectedNote(selectedNote, { pitch: selectedNote.pitch - 1 }, true)}
                    >
                      −
                    </button>
                    <button
                      type="button"
                      onClick={() => handlePatchSelectedNote(selectedNote, { pitch: selectedNote.pitch + 1 }, true)}
                    >
                      +
                    </button>
                  </div>

                  <div className="smr-stepper">
                    <span>Độ dài</span>
                    <button
                      type="button"
                      onClick={() =>
                        handlePatchSelectedNote(selectedNote, { duration: Math.max(0.05, selectedNote.duration - 0.08) })
                      }
                    >
                      −
                    </button>
                    <span className="mono smr-stepper-val">{Math.round(selectedNote.duration * 1000)}ms</span>
                    <button
                      type="button"
                      onClick={() => handlePatchSelectedNote(selectedNote, { duration: selectedNote.duration + 0.08 })}
                    >
                      +
                    </button>
                  </div>
                </div>

                <div className="smr-note-actions">
                  <button
                    type="button"
                    className={`btn small ${
                      selectedNote.status === "removed" || selectedNote.status === "manual_deleted"
                        ? "primary"
                        : "danger"
                    }`}
                    onClick={() => handleToggleNoteDelete(selectedNote)}
                  >
                    {selectedNote.status === "removed" || selectedNote.status === "manual_deleted"
                      ? "↺ Khôi phục"
                      : "× Xóa nốt"}
                  </button>
                  <button
                    type="button"
                    className="smr-close-note-btn"
                    aria-label="Đóng"
                    onClick={() => setSelectedNoteIdx(null)}
                  >
                    ×
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar: 1-Line Score + 4 Presets + 5 Clean ON/OFF Toggles */}
        <aside className="smr-panel-col" aria-label="Tùy chỉnh sửa MIDI">
          <div className="smr-health-strip">
            <div className="smr-score-headline">
              <span className="smr-score-before">{before?.scores.overall ?? "—"}</span>
              <span className="smr-score-arrow" aria-hidden="true">→</span>
              <strong className="smr-score-after">{after?.scores.overall ?? "—"}</strong>
              <span className="smr-score-max">/100</span>
              {scoreDelta > 0 && <span className="smr-score-delta">+{scoreDelta}</span>}
            </div>
            <div className="smr-note-summary">
              <strong>{after?.note_count ?? 0}</strong> nốt
            </div>
          </div>

          <div className="smr-side-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={sideTab === "controls"}
              className={sideTab === "controls" ? "active" : ""}
              onClick={() => setSideTab("controls")}
            >
              Sửa tự động
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={sideTab === "issues"}
              className={sideTab === "issues" ? "active" : ""}
              onClick={() => setSideTab("issues")}
            >
              <span>Lỗi</span>
              <span className="smr-tab-count">{before?.total_issues ?? 0}</span>
            </button>
          </div>

          {sideTab === "controls" ? (
            <div className="smr-tab-body">
              <div className="smr-preset-row" role="group" aria-label="Cấu hình nhanh">
                {(Object.keys(PRESETS) as Exclude<PresetId, "custom">[]).map(id => (
                  <button
                    key={id}
                    type="button"
                    className={`smr-preset-pill ${presetId === id ? "on" : ""}`}
                    title={PRESETS[id].tip}
                    onClick={() => applyPreset(id)}
                  >
                    {PRESETS[id].label}
                  </button>
                ))}
              </div>

              <div className="smr-modules-list">
                <ToggleCard
                  title="Dọn nốt rác & trùng phím"
                  count={diff ? diff.actions.ghosts_removed + diff.actions.overlaps_fixed : 0}
                  enabled={options.remove_ghost_notes || options.fix_overlaps}
                  expanded={expandedCard === "ghost"}
                  onToggleExpand={() => setExpandedCard(c => (c === "ghost" ? null : "ghost"))}
                  onToggle={v => patchOptions({ remove_ghost_notes: v, fix_overlaps: v })}
                >
                  <ComfortableSlider
                    label="Bỏ nốt ngắn dưới"
                    value={options.min_duration_ms}
                    min={15}
                    max={95}
                    step={5}
                    display={`${options.min_duration_ms} ms`}
                    onChange={v => patchOptions({ min_duration_ms: v, remove_ghost_notes: true })}
                  />
                  <ComfortableSlider
                    label="Bỏ nốt nhẹ dưới"
                    value={options.min_velocity}
                    min={5}
                    max={45}
                    step={1}
                    display={`vel ${options.min_velocity}`}
                    onChange={v => patchOptions({ min_velocity: v, remove_ghost_notes: true })}
                  />
                </ToggleCard>

                <ToggleCard
                  title="Tối ưu tầm tay Piano"
                  count={diff?.actions.hand_spans_fixed ?? 0}
                  enabled={options.fix_hand_span}
                  expanded={expandedCard === "hand"}
                  onToggleExpand={() => setExpandedCard(c => (c === "hand" ? null : "hand"))}
                  onToggle={v => patchOptions({ fix_hand_span: v })}
                >
                  <ComfortableSlider
                    label="Độ giãn tay tối đa"
                    value={options.max_hand_span}
                    min={12}
                    max={18}
                    step={1}
                    display={spanLabel(options.max_hand_span)}
                    onChange={v => patchOptions({ max_hand_span: v })}
                  />
                  <ComfortableSlider
                    label="Số ngón tối đa"
                    value={options.max_polyphony}
                    min={6}
                    max={12}
                    step={1}
                    display={`${options.max_polyphony} ngón`}
                    onChange={v => patchOptions({ max_polyphony: v })}
                  />
                </ToggleCard>

                <ToggleCard
                  title="Căn nhịp thông minh"
                  count={diff?.actions.notes_quantized ?? 0}
                  enabled={options.smart_quantize}
                  expanded={expandedCard === "quantize"}
                  onToggleExpand={() => setExpandedCard(c => (c === "quantize" ? null : "quantize"))}
                  onToggle={v => patchOptions({ smart_quantize: v })}
                >
                  <ComfortableSlider
                    label="Độ bám lưới nhịp"
                    value={Math.round(options.quantize_strength * 100)}
                    min={10}
                    max={100}
                    step={5}
                    display={`${Math.round(options.quantize_strength * 100)}%`}
                    onChange={v => patchOptions({ quantize_strength: v / 100 })}
                  />
                  <div className="smr-grid-row">
                    <span>Lưới phách</span>
                    <div className="smr-mini-seg">
                      {(
                        [
                          ["16th", "1/16"],
                          ["8th", "1/8"],
                          ["triplet", "Chùm 3"],
                        ] as const
                      ).map(([val, lbl]) => (
                        <button
                          key={val}
                          type="button"
                          className={options.grid_subdivision === val ? "on" : ""}
                          onClick={() => patchOptions({ grid_subdivision: val })}
                        >
                          {lbl}
                        </button>
                      ))}
                    </div>
                  </div>
                </ToggleCard>

                <ToggleCard
                  title="Cân bằng sắc thái & giai điệu"
                  count={diff?.actions.velocities_balanced ?? 0}
                  enabled={options.normalize_velocity}
                  expanded={expandedCard === "vel"}
                  onToggleExpand={() => setExpandedCard(c => (c === "vel" ? null : "vel"))}
                  onToggle={v => patchOptions({ normalize_velocity: v })}
                >
                  <ComfortableSlider
                    label="Nâng giai điệu tay phải"
                    value={options.melody_boost}
                    min={0}
                    max={20}
                    step={1}
                    display={`+${options.melody_boost}`}
                    onChange={v => patchOptions({ melody_boost: v })}
                  />
                </ToggleCard>

                <ToggleCard
                  title="Làm sạch Sustain Pedal"
                  count={diff ? diff.actions.pedals_cleaned + diff.actions.range_clamped : 0}
                  enabled={options.clean_pedals || options.clamp_piano_range}
                  expanded={expandedCard === "pedal"}
                  onToggleExpand={() => setExpandedCard(c => (c === "pedal" ? null : "pedal"))}
                  onToggle={v => patchOptions({ clean_pedals: v, clamp_piano_range: v })}
                >
                  <ComfortableSlider
                    label="Giữ pedal tối đa"
                    value={options.max_pedal_sec}
                    min={1.5}
                    max={7.5}
                    step={0.5}
                    display={`${options.max_pedal_sec.toFixed(1)}s`}
                    onChange={v => patchOptions({ max_pedal_sec: v, clean_pedals: true })}
                  />
                  <label className="smr-check-row">
                    <input
                      type="checkbox"
                      checked={options.repedal_on_chord}
                      onChange={e => patchOptions({ repedal_on_chord: e.target.checked })}
                    />
                    <span>Tự nhả khi đổi hợp âm</span>
                  </label>
                </ToggleCard>
              </div>
            </div>
          ) : (
            <div className="smr-tab-body">
              <div className="smr-issues-list">
                {(before?.issues ?? []).length === 0 ? (
                  <div className="smr-issues-clean">✓ Không phát hiện lỗi đáng kể!</div>
                ) : (
                  (before?.issues ?? []).slice(0, 100).map(issue => (
                    <button
                      key={issue.id}
                      type="button"
                      className={`smr-issue-row ${
                        focusedIssueTime !== null && Math.abs(focusedIssueTime - issue.time) < 0.02 ? "active" : ""
                      }`}
                      onClick={() => handleJumpToIssue(issue)}
                    >
                      <span className={`smr-sev-dot sev-${issue.severity}`} aria-hidden="true" />
                      <div className="smr-issue-body">
                        <div className="smr-issue-top">
                          <span className="smr-issue-time mono">{formatTime(issue.time)}</span>
                          <span className="smr-issue-type">{ISSUE_SHORT[issue.type]}</span>
                        </div>
                        <span className="smr-issue-label">{issue.label}</span>
                      </div>
                      <span className="smr-issue-jump">▶</span>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function ToggleCard({
  title,
  count,
  enabled,
  expanded,
  onToggleExpand,
  onToggle,
  children,
}: {
  title: string;
  count: number;
  enabled: boolean;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggle: (next: boolean) => void;
  children?: React.ReactNode;
}) {
  return (
    <section className={`smr-card ${enabled ? "is-on" : "is-off"}`}>
      <div className="smr-card-head">
        <button
          type="button"
          className="smr-card-expand-btn"
          aria-expanded={expanded}
          onClick={onToggleExpand}
        >
          <strong>{title}</strong>
          {count > 0 && <span className="smr-card-badge">{count}</span>}
          <span className="smr-card-caret">{expanded ? "▴" : "▾"}</span>
        </button>
        <label className="switch">
          <input type="checkbox" checked={enabled} onChange={e => onToggle(e.target.checked)} />
          <span />
        </label>
      </div>
      {enabled && expanded && children && <div className="smr-card-body">{children}</div>}
    </section>
  );
}

function ComfortableSlider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <div className="smr-slider-row">
      <div className="smr-slider-head">
        <span>{label}</span>
        <strong className="mono">{display}</strong>
      </div>
      <input
        className="smr-big-range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--fill": `${pct}%` } as CSSProperties}
        onChange={e => onChange(Number(e.target.value))}
      />
    </div>
  );
}

/** Direct-Manipulation Piano Roll: click note = select/edit, click empty = seek, drag empty = loop */
function RepairPianoRoll({
  player,
  preview,
  auditionMode,
  focusTime,
  selectedNoteIdx,
  onSelectNote,
}: {
  player: Player;
  preview: MidiRepairPreviewResponse | null;
  auditionMode: AuditionMode;
  focusTime: number | null;
  selectedNoteIdx: number | null;
  onSelectNote: (note: MidiRepairDiffNote | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const keyboardRef = useRef<HTMLCanvasElement>(null);
  const view = useRef({ start: 0, seconds: 8 });
  const drag = useRef<{ x: number; t: number; moved: boolean } | null>(null);
  const geomRef = useRef({ lo: 48, hi: 76, top: CHORD_H, rowH: 16 });
  const heldPointers = useRef(new Map<number, number>());

  const [zoom, setZoom] = useState(10);
  const [pitchSpan, setPitchSpan] = useState(88);
  const [pitchOffset, setPitchOffset] = useState(0);
  const [hoveredIdx, setHoveredIdx] = useState<number>(-1);

  const requestedZoom = useRef(zoom);
  const requestedSpan = useRef(pitchSpan);
  const dataRef = useRef({ preview, auditionMode, focusTime, hoveredIdx, selectedNoteIdx });

  useEffect(() => {
    requestedZoom.current = zoom;
  }, [zoom]);
  useEffect(() => {
    requestedSpan.current = pitchSpan;
  }, [pitchSpan]);
  useEffect(() => {
    dataRef.current = { preview, auditionMode, focusTime, hoveredIdx, selectedNoteIdx };
  }, [preview, auditionMode, focusTime, hoveredIdx, selectedNoteIdx]);

  useEffect(() => {
    if (focusTime == null) return;
    const dur = preview?.after.duration || player.duration || 30;
    const sec = view.current.seconds;
    view.current.start = Math.max(0, Math.min(focusTime - sec * 0.3, Math.max(0, dur - sec)));
  }, [focusTime, preview?.after.duration, player.duration]);

  const computeRange = useCallback((p: MidiRepairPreviewResponse | null, spanReq: number, offset: number) => {
    let lo = MAX_PITCH;
    let hi = MIN_PITCH;
    for (const n of p?.diff.notes ?? []) {
      if (n.pitch >= MIN_PITCH && n.pitch <= MAX_PITCH) {
        lo = Math.min(lo, n.pitch);
        hi = Math.max(hi, n.pitch);
      }
    }
    if (hi < lo) {
      lo = 48;
      hi = 76;
    }
    const span = Math.max(16, Math.min(88, spanReq));
    const center = Math.max(MIN_PITCH + (span - 1) / 2, Math.min(MAX_PITCH - (span - 1) / 2, (lo + hi) / 2 + offset));
    const rLo = Math.max(MIN_PITCH, Math.floor(center - (span - 1) / 2));
    const rHi = Math.min(MAX_PITCH, rLo + span - 1);
    return { lo: Math.max(MIN_PITCH, rHi - span + 1), hi: rHi };
  }, []);

  const timeAt = useCallback(
    (clientX: number) => {
      const rect = canvasRef.current!.getBoundingClientRect();
      const dur = dataRef.current.preview?.after.duration || player.duration || 1;
      return Math.max(0, Math.min(dur, view.current.start + ((clientX - rect.left) / rect.width) * view.current.seconds));
    },
    [player],
  );

  const findNoteAt = useCallback((clientX: number, clientY: number): MidiRepairDiffNote | null => {
    const { preview: p } = dataRef.current;
    const canvas = canvasRef.current;
    if (!p || !canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const { lo, hi, top, rowH } = geomRef.current;
    const pitch = hi - Math.floor((y - top) / rowH);
    if (pitch < lo || pitch > hi) return null;
    const tSec = view.current.start + (x / rect.width) * view.current.seconds;
    const tol = Math.max(0.035, (view.current.seconds / rect.width) * 6);

    let best: MidiRepairDiffNote | null = null;
    for (const n of p.diff.notes) {
      if (n.pitch !== pitch) continue;
      if (tSec >= n.onset - tol && tSec <= n.onset + Math.max(0.08, n.duration) + tol) {
        best = n;
        if (n.status !== "kept") return n;
      }
    }
    return best;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const keyboard = keyboardRef.current!;
    const keyCtx = keyboard.getContext("2d")!;
    let raf = 0;

    const readTheme = () => {
      const s = getComputedStyle(canvas);
      const keys = [
        "--roll-bg",
        "--roll-black-key",
        "--roll-grid",
        "--roll-text",
        "--roll-key-light",
        "--roll-key-dark",
        "--accent",
        "--accent-2",
        "--accent-soft",
        "--warning",
        "--danger",
        "--playhead",
        "--loop",
      ];
      return Object.fromEntries(keys.map(k => [k, s.getPropertyValue(k).trim()]));
    };
    let palette = readTheme();

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { preview: p, auditionMode: am, focusTime: ft, hoveredIdx: hIdx, selectedNoteIdx: sIdx } = dataRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const kw = keyboard.clientWidth;
      if (!w || !h || !kw) return;

      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      if (keyboard.width !== Math.round(kw * dpr) || keyboard.height !== Math.round(h * dpr)) {
        keyboard.width = Math.round(kw * dpr);
        keyboard.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      keyCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const dur = p?.after.duration || player.duration || 16;
      const pos = player.position();
      const v = view.current;
      v.seconds = dur > 0 ? Math.min(requestedZoom.current, Math.max(3, dur)) : requestedZoom.current;
      if (player.playing && (pos > v.start + v.seconds * 0.86 || pos < v.start)) {
        v.start = pos - v.seconds * 0.12;
      }
      v.start = Math.max(0, Math.min(v.start, Math.max(0, dur - v.seconds)));
      const t0 = v.start;
      const t1 = v.start + v.seconds;
      const xOf = (sec: number) => ((sec - t0) / v.seconds) * w;

      const { lo, hi } = computeRange(p, requestedSpan.current, pitchOffset);
      const top = CHORD_H;
      const bottom = h - PEDAL_H;
      const rowH = (bottom - top) / (hi - lo + 1);
      const yOf = (pitch: number) => top + (hi - pitch) * rowH;
      geomRef.current = { lo, hi, top, rowH };

      keyCtx.fillStyle = palette["--roll-bg"];
      keyCtx.fillRect(0, 0, kw, h);
      const activePitches = new Set<number>();
      for (const n of player.notes) {
        if (n[0] > pos) break;
        if (pos < n[4]) activePitches.add(n[2]);
      }
      for (const pitch of heldPointers.current.values()) activePitches.add(pitch);

      for (let pitch = lo; pitch <= hi; pitch++) {
        if (BLACK_PITCH_CLASSES.includes(pitch % 12)) continue;
        const y = yOf(pitch);
        keyCtx.fillStyle = activePitches.has(pitch) ? palette["--accent-soft"] : palette["--roll-key-light"];
        keyCtx.fillRect(0, y, kw, rowH);
        keyCtx.strokeStyle = palette["--roll-grid"];
        keyCtx.strokeRect(0.5, y + 0.5, kw - 1, Math.max(0, rowH - 1));
        if ((pitch % 12 === 0 && rowH >= 3) || rowH >= 15) {
          keyCtx.fillStyle = activePitches.has(pitch) ? palette["--accent"] : palette["--roll-text"];
          keyCtx.font = `600 ${rowH >= 8 ? 10 : 9}px ${CANVAS_FONT}`;
          keyCtx.textBaseline = "middle";
          keyCtx.fillText(noteName(pitch), 6, y + rowH / 2);
        }
      }
      for (let pitch = lo; pitch <= hi; pitch++) {
        if (!BLACK_PITCH_CLASSES.includes(pitch % 12)) continue;
        const y = yOf(pitch);
        keyCtx.fillStyle = activePitches.has(pitch) ? palette["--accent"] : palette["--roll-key-dark"];
        keyCtx.fillRect(0, y, kw * 0.62, Math.max(1, rowH));
      }

      ctx.fillStyle = palette["--roll-bg"];
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = palette["--roll-black-key"];
      for (let pitch = lo; pitch <= hi; pitch++) {
        if (BLACK_PITCH_CLASSES.includes(pitch % 12)) ctx.fillRect(0, yOf(pitch), w, rowH);
      }

      ctx.fillStyle = palette["--roll-grid"];
      for (let pitch = lo; pitch <= hi; pitch++) {
        if (pitch % 12 === 0) ctx.fillRect(0, Math.round(yOf(pitch) + rowH), w, 1);
      }
      if (p?.beats?.length) {
        for (const b of p.beats) {
          if (b >= t0 && b <= t1) ctx.fillRect(Math.round(xOf(b)), top, 1, bottom - top);
        }
      }

      if (player.loop) {
        const [a, b] = player.loop;
        ctx.globalAlpha = player.loopOn ? 1 : 0.4;
        ctx.fillStyle = palette["--loop"];
        ctx.fillRect(xOf(a), 0, xOf(b) - xOf(a), h);
        ctx.globalAlpha = 1;
      }

      if (p) {
        for (const n of p.diff.notes) {
          const isRemoved = n.status === "removed" || n.status === "manual_deleted";
          if (n.onset > t1 || n.onset + n.duration < t0) continue;
          if (n.pitch < lo || n.pitch > hi) continue;

          const x = xOf(n.onset);
          const nw = Math.max(6, (n.duration / v.seconds) * w);
          const y = yOf(n.pitch) + 1;
          const nh = Math.max(4, rowH - 2);
          const isSelected = n.idx === sIdx;
          const isHovered = n.idx === hIdx;

          if (isRemoved) {
            ctx.save();
            ctx.globalAlpha = isSelected || isHovered ? 0.9 : 0.45;
            ctx.fillStyle = palette["--danger"];
            ctx.beginPath();
            ctx.roundRect(x, y, nw, nh, 3);
            ctx.fill();
            ctx.strokeStyle = palette["--danger"];
            ctx.setLineDash([4, 2]);
            ctx.lineWidth = 1.4;
            ctx.stroke();
            ctx.restore();
          } else if (n.status === "modified") {
            ctx.globalAlpha = isSelected || isHovered ? 1 : 0.55 + 0.45 * (n.velocity / 127);
            ctx.fillStyle = palette["--warning"];
            ctx.beginPath();
            ctx.roundRect(x, y, nw, nh, 3);
            ctx.fill();
            ctx.globalAlpha = 1;
          } else {
            ctx.globalAlpha = isSelected || isHovered ? 1 : 0.45 + 0.55 * (n.velocity / 127);
            ctx.fillStyle = am === "after" ? palette["--accent"] : palette["--accent-2"];
            ctx.beginPath();
            ctx.roundRect(x, y, nw, nh, 3);
            ctx.fill();
            ctx.globalAlpha = 1;
          }

          if (isSelected) {
            ctx.strokeStyle = palette["--playhead"];
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.roundRect(x - 1, y - 1, nw + 2, nh + 2, 4);
            ctx.stroke();
          } else if (isHovered) {
            ctx.strokeStyle = palette["--roll-text"];
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.roundRect(x - 0.5, y - 0.5, nw + 1, nh + 1, 3);
            ctx.stroke();
          }
        }

        const pedals = am === "before" ? p.before.pedals : p.after.pedals;
        ctx.fillStyle = palette["--accent"];
        ctx.globalAlpha = 0.45;
        for (const [a, b] of pedals) {
          if (a > t1 || b < t0) continue;
          ctx.fillRect(xOf(a), h - PEDAL_H + 3, Math.max(2, xOf(b) - xOf(a) - 1), PEDAL_H - 5);
        }
        ctx.globalAlpha = 1;
      }

      ctx.fillStyle = palette["--roll-bg"];
      ctx.fillRect(0, 0, w, CHORD_H);
      if (p?.chords?.length) {
        ctx.font = `600 11px ${CANVAS_FONT}`;
        ctx.fillStyle = palette["--roll-text"];
        ctx.textBaseline = "middle";
        let lastX = -Infinity;
        for (const [ct, name] of p.chords) {
          if (ct > t1) break;
          const x = xOf(ct);
          if (x < -30 || x - lastX < 36) continue;
          ctx.fillText(name, x + 4, CHORD_H / 2);
          lastX = x;
        }
      }

      if (ft != null && ft >= t0 && ft <= t1) {
        const fx = xOf(ft);
        ctx.save();
        ctx.strokeStyle = palette["--warning"];
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(fx, top);
        ctx.lineTo(fx, bottom);
        ctx.stroke();
        ctx.restore();
      }

      const px = xOf(pos);
      if (px >= 0 && px <= w) {
        ctx.fillStyle = palette["--playhead"];
        ctx.fillRect(px - 1, 0, 2, h);
      }

      ctx.font = `10px ${CANVAS_FONT}`;
      ctx.fillStyle = palette["--roll-text"];
      ctx.textBaseline = "bottom";
      const step = v.seconds > 24 ? 5 : v.seconds > 10 ? 2 : 1;
      for (let s = Math.ceil(t0 / step) * step; s <= t1; s += step) {
        ctx.fillText(formatTime(s), xOf(s) + 3, bottom - 2);
      }
    };

    const onThemeChange = () => {
      palette = readTheme();
    };
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", onThemeChange);
    draw();
    return () => {
      cancelAnimationFrame(raf);
      mq.removeEventListener("change", onThemeChange);
    };
  }, [player, pitchOffset, computeRange]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const host = canvas.parentElement!;
    const onWheel = (e: WheelEvent) => {
      const dur = dataRef.current.preview?.after.duration || player.duration || 30;
      const v = view.current;
      if (e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setPitchOffset(o => Math.max(-40, Math.min(40, o - Math.sign(e.deltaY) * 3)));
      } else if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const t = timeAt(e.clientX);
        const next = Math.min(Math.max(4, dur), Math.max(3, v.seconds * Math.exp(e.deltaY * 0.002)));
        v.start = Math.max(0, Math.min(t - ((t - v.start) * next) / v.seconds, Math.max(0, dur - next)));
        v.seconds = next;
        setZoom(Math.round(next));
      } else {
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        if (delta) e.preventDefault();
        v.start = Math.max(0, Math.min(v.start + (delta / 800) * v.seconds, Math.max(0, dur - v.seconds)));
      }
    };
    host.addEventListener("wheel", onWheel, { passive: false });
    return () => host.removeEventListener("wheel", onWheel);
  }, [player.duration, timeAt]);

  return (
    <div className="smr-roll-wrap">
      <div className="smr-roll-content">
        <canvas
          ref={keyboardRef}
          className="smr-roll-keys"
          onPointerDown={e => {
            const rect = e.currentTarget.getBoundingClientRect();
            const { lo, hi, top, rowH } = geomRef.current;
            const pitch = hi - Math.floor((e.clientY - rect.top - top) / rowH);
            if (pitch >= lo && pitch <= hi && player.startManualNote(-pitch, pitch)) {
              heldPointers.current.set(e.pointerId, pitch);
              e.currentTarget.setPointerCapture(e.pointerId);
            }
          }}
          onPointerUp={e => {
            const p = heldPointers.current.get(e.pointerId);
            if (p != null) {
              heldPointers.current.delete(e.pointerId);
              player.stopManualNote(-p);
            }
          }}
          onPointerCancel={e => {
            const p = heldPointers.current.get(e.pointerId);
            if (p != null) {
              heldPointers.current.delete(e.pointerId);
              player.stopManualNote(-p);
            }
          }}
        />
        <canvas
          ref={canvasRef}
          className="smr-roll-canvas"
          onPointerDown={e => {
            const hit = findNoteAt(e.clientX, e.clientY);
            if (hit) {
              onSelectNote(hit);
              return;
            }
            onSelectNote(null);
            e.currentTarget.setPointerCapture(e.pointerId);
            drag.current = { x: e.clientX, t: timeAt(e.clientX), moved: false };
          }}
          onPointerMove={e => {
            const d = drag.current;
            if (d) {
              if (Math.abs(e.clientX - d.x) > 5) d.moved = true;
              if (d.moved) {
                const tSec = timeAt(e.clientX);
                player.setLoop([Math.min(d.t, tSec), Math.max(d.t, tSec)]);
              }
              return;
            }
            setHoveredIdx(findNoteAt(e.clientX, e.clientY)?.idx ?? -1);
          }}
          onPointerUp={e => {
            const d = drag.current;
            drag.current = null;
            if (!d) return;
            if (!d.moved) {
              player.seek(timeAt(e.clientX));
            } else if (player.loop && player.loop[1] - player.loop[0] < 0.4) {
              player.setLoop(null);
            } else if (player.loop) {
              player.seek(player.loop[0]);
            }
          }}
          onPointerLeave={() => setHoveredIdx(-1)}
        />
      </div>

      <div className="smr-roll-foot">
        {player.loop ? (
          <button type="button" className="smr-link-btn" onClick={() => player.setLoop(null)}>
            Bỏ vòng lặp ({formatTime(player.loop[0])}–{formatTime(player.loop[1])})
          </button>
        ) : (
          <span className="smr-foot-tip">Nhấp nốt để sửa · Kéo nền trống để lặp</span>
        )}
        <div className="smr-zoom-btns">
          <button type="button" onClick={() => setZoom(z => Math.max(3, Math.round(z * 0.72)))} title="Phóng to">
            +
          </button>
          <span className="mono">{zoom}s</span>
          <button type="button" onClick={() => setZoom(z => Math.min(45, Math.round(z * 1.35)))} title="Thu nhỏ">
            −
          </button>
          <button
            type="button"
            className={pitchSpan === 88 ? "on" : ""}
            aria-pressed={pitchSpan === 88}
            onClick={() => {
              setPitchSpan(s => (s === 88 ? 28 : 88));
              setPitchOffset(0);
            }}
            title="Chuyển giữa toàn bộ 88 phím (A0–C8) và phóng to nốt để chỉnh tay"
          >
            88 phím
          </button>
        </div>
      </div>
    </div>
  );
}
