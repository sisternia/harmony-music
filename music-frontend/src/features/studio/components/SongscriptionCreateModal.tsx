import { useEffect, useRef, useState } from "react";
import { api, type ProjectSummary, type SongscriptionItem } from "../api";
import { LANGUAGE_OPTIONS, stripImportedLyricsFormat } from "../utils/lyricsUtils";
import {
  DEFAULT_SONGSCRIPTION_MODEL_ID,
  getSongscriptionModels,
  normalizeSongscriptionModelId,
} from "../utils/songscriptionModels";
import "../styles/songscription-create.css";

const MEDIA_ACCEPT = /\.(mp3|wav|flac|m4a|ogg|opus|aac|mp4|webm|mkv)$/i;
const LYRICS_FILE_ACCEPT = /\.(txt|lrc|srt)$/i;
const YOUTUBE_URL_RE = /^(https?:\/\/)?(www\.|m\.)?(youtube\.com|youtu\.be)\//i;

export type SongscriptionCreateModalProps = {
  open: boolean;
  projects: ProjectSummary[];
  models: { id: string; label: string }[];
  vocalAvailable: boolean;
  youtubeSupported: boolean;
  onClose: () => void;
  onCreated: (created: SongscriptionItem) => void;
};

export function SongscriptionCreateModal({
  open,
  projects,
  models,
  vocalAvailable,
  youtubeSupported,
  onClose,
  onCreated,
}: SongscriptionCreateModalProps) {
  const audioProjects = projects.filter(p => Boolean(p.audio));

  const [sourceTab, setSourceTab] = useState<"media" | "project">("media");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [processMode, setProcessMode] = useState<"extract" | "align">("extract");
  const [projectId, setProjectId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [youtube, setYoutube] = useState("");
  const [customTitle, setCustomTitle] = useState("");
  const [userLyricsText, setUserLyricsText] = useState("");
  const [language, setLanguage] = useState("auto");
  const [modelId, setModelId] = useState(DEFAULT_SONGSCRIPTION_MODEL_ID);
  const availableModels = getSongscriptionModels(models);
  const selectedModelId = normalizeSongscriptionModelId(modelId);
  const [isolateVocals, setIsolateVocals] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  const dialogRef = useRef<HTMLElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lyricsFileInputRef = useRef<HTMLInputElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const closeHandlerRef = useRef(onClose);
  const submittingRef = useRef(submitting);

  useEffect(() => {
    closeHandlerRef.current = onClose;
    submittingRef.current = submitting;
  }, [onClose, submitting]);

  // Focus trap, Escape-to-close, body scroll lock, and return focus on close
  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const getFocusable = () => {
      const root = dialogRef.current;
      if (!root) return [] as HTMLElement[];
      return Array.from(
        root.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter(el => el.offsetParent !== null && !el.matches(":disabled"));
    };

    const focusTimer = window.setTimeout(() => {
      const first = getFocusable()[0];
      first?.focus();
    }, 30);

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (!submittingRef.current) {
          e.preventDefault();
          closeHandlerRef.current();
        }
        return;
      }
      if (e.key === "Tab") {
        const focusables = getFocusable();
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first || !dialogRef.current?.contains(document.activeElement)) {
            e.preventDefault();
            last.focus();
          }
        } else if (document.activeElement === last || !dialogRef.current?.contains(document.activeElement)) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = prevOverflow;
      previousFocusRef.current?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  const selectedProject = audioProjects.find(p => p.id === projectId) ?? null;

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    if (!MEDIA_ACCEPT.test(candidate.name)) {
      setFormError("Chỉ hỗ trợ tệp âm thanh có giọng hát (MP3, WAV, FLAC, M4A, OGG, MP4, WebM). Không hỗ trợ MIDI.");
      return;
    }
    if (candidate.size > 100 * 1024 * 1024) {
      setFormError("Tệp âm thanh vượt quá giới hạn 100 MB. Vui lòng chọn tệp nhỏ hơn.");
      return;
    }
    setFile(candidate);
    setYoutube("");
    setCustomTitle(candidate.name.replace(/\.[^.]+$/, ""));
    setFormError("");
  };

  const importLyricsFile = async (candidate?: File) => {
    if (!candidate) return;
    if (!LYRICS_FILE_ACCEPT.test(candidate.name)) {
      setFormError("Chỉ hỗ trợ nhập lời từ tệp .TXT, .LRC hoặc .SRT.");
      return;
    }
    try {
      const raw = await candidate.text();
      setUserLyricsText(stripImportedLyricsFormat(raw));
      setFormError("");
    } catch {
      setFormError("Không thể đọc tệp lời bài hát. Vui lòng thử lại hoặc dán lời trực tiếp.");
    }
  };

  const handleSubmit = async () => {
    if (submitting) return;
    setFormError("");

    if (processMode === "align" && !userLyricsText.trim()) {
      setAdvancedOpen(true);
      setFormError("Vui lòng dán hoặc nhập lời bài hát gốc trước khi bắt đầu đồng bộ.");
      return;
    }

    const form = new FormData();
    if (sourceTab === "project") {
      if (!selectedProject) {
        setFormError(
          audioProjects.length
            ? "Vui lòng chọn một bài hát có tệp âm thanh gốc từ danh sách Projects."
            : "Chưa có Project nào chứa tệp âm thanh gốc. Hãy chuyển sang tab Âm thanh / YouTube.",
        );
        return;
      }
      form.append("project_id", selectedProject.id);
      form.append("title", selectedProject.title);
    } else if (file) {
      form.append("audio", file);
      form.append("title", customTitle.trim() || file.name.replace(/\.[^.]+$/, ""));
    } else {
      const trimmedUrl = youtube.trim();
      if (!trimmedUrl) {
        setFormError("Vui lòng chọn tệp âm thanh hoặc dán đường dẫn video YouTube hợp lệ.");
        return;
      }
      if (!YOUTUBE_URL_RE.test(trimmedUrl)) {
        setFormError("Đường dẫn chưa đúng định dạng YouTube (youtube.com hoặc youtu.be).");
        return;
      }
      if (!youtubeSupported) {
        setFormError("Backend hiện tại chưa bật tính năng nhập link YouTube. Vui lòng khởi động lại server backend mới nhất.");
        return;
      }
      form.append("youtube", trimmedUrl);
      if (customTitle.trim()) {
        form.append("title", customTitle.trim());
      }
    }

    form.append("mode", processMode);
    if (processMode === "align") {
      form.append("lyrics_text", userLyricsText);
    }
    form.append("language", language);
    form.append("model", selectedModelId);
    form.append("isolate_vocals", String(isolateVocals && vocalAvailable));

    setSubmitting(true);
    try {
      const created = await api.createSongscription(form);
      setFile(null);
      setYoutube("");
      setCustomTitle("");
      setUserLyricsText("");
      setProjectId("");
      onCreated(created);
    } catch (err) {
      setFormError(
        err instanceof Error
          ? err.message
          : "Không thể khởi tạo Songscription. Vui lòng kiểm tra kết nối máy chủ rồi thử lại.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="ss-create-backdrop"
      onMouseDown={e => {
        if (e.target === e.currentTarget && !submitting) onClose();
      }}
    >
      <section
        ref={dialogRef}
        className="ss-create-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ss-create-heading"
      >
        <button
          type="button"
          className="ss-create-dismiss"
          onClick={onClose}
          aria-label="Đóng hộp thoại tạo Songscription mới"
          disabled={submitting}
        >
          ×
        </button>
        <span className="ss-create-eyebrow">HARMONY AURALIS · SONGSCRIPTION AI</span>
        <h2 id="ss-create-heading">Tạo Songscription mới</h2>
        <p className="ss-create-intro">
          Chọn một bản thu hoặc video YouTube. AI sẽ trích xuất lời bài hát để bạn nghe và rà soát.
        </p>

        <form aria-busy={submitting} onSubmit={e => { e.preventDefault(); void handleSubmit(); }}>
          <fieldset className="ss-create-fields" disabled={submitting}>
            <div className="ss-create-source-tabs" role="tablist" aria-label="Nguồn âm thanh">
              <button
                id="ss-create-media-tab"
                type="button"
                role="tab"
                aria-controls="ss-create-media-panel"
                aria-selected={sourceTab === "media"}
                onClick={() => { setSourceTab("media"); setFormError(""); }}
              >
                Âm thanh / YouTube
              </button>
              <button
                id="ss-create-project-tab"
                type="button"
                role="tab"
                aria-controls="ss-create-project-panel"
                aria-selected={sourceTab === "project"}
                onClick={() => { setSourceTab("project"); setFormError(""); }}
              >
                Từ Projects ({audioProjects.length})
              </button>
            </div>

            {sourceTab === "media" ? (
              <div id="ss-create-media-panel" role="tabpanel" aria-labelledby="ss-create-media-tab">
                <label
                  className={`ss-create-upload-zone ${dragging ? "dragging" : ""}`}
                  onDragOver={e => { e.preventDefault(); if (!submitting) setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={e => {
                    e.preventDefault();
                    setDragging(false);
                    if (!submitting) chooseFile(e.dataTransfer.files[0]);
                  }}
                >
                  <svg className="ss-create-upload-icon" width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <path d="M14 34H10a8 8 0 0 1-1-16 14 14 0 0 1 27-2 9 9 0 0 1 2 18h-5M24 39V20m-7 7 7-7 7 7" />
                  </svg>
                  <strong>{file ? file.name : "Nhập file âm thanh hoặc bản nhạc"}</strong>
                  <span>Hoặc kéo và thả file vào đây</span>
                  <span>Âm thanh hoặc video · tối đa 100 MB</span>
                  <input
                    ref={fileInputRef}
                    className="ss-create-file-input"
                    type="file"
                    accept=".mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mp4,.webm,.mkv"
                    aria-label="Chọn file âm thanh hoặc video có giọng hát"
                    onChange={e => { chooseFile(e.target.files?.[0]); e.target.value = ""; }}
                  />
                </label>
                <label className="youtube-field ss-create-youtube" htmlFor="ss-youtube-input">
                  <svg viewBox="0 0 28 20" aria-hidden="true">
                    <rect width="28" height="20" rx="5" fill="currentColor" />
                    <path d="m11 5 8 5-8 5z" fill="var(--on-accent)" />
                  </svg>
                  <input
                    id="ss-youtube-input"
                    type="text"
                    inputMode="url"
                    autoCapitalize="none"
                    spellCheck={false}
                    value={youtube}
                    aria-label="Link YouTube"
                    placeholder="Dán link YouTube tại đây"
                    onChange={e => {
                      setYoutube(e.target.value);
                      if (e.target.value.trim()) {
                        if (file && customTitle === file.name.replace(/\.[^.]+$/, "")) setCustomTitle("");
                        setFile(null);
                      }
                      setFormError("");
                    }}
                  />
                </label>
                {!youtubeSupported && youtube.trim() && (
                  <p className="ss-create-warning">Backend hiện tại chưa hỗ trợ YouTube. Hãy khởi động lại backend mới nhất.</p>
                )}
              </div>
            ) : (
              <div className="ss-create-project-panel" id="ss-create-project-panel" role="tabpanel" aria-labelledby="ss-create-project-tab">
                <label className="ss-create-field" htmlFor="ss-project-select">
                  <span>Chọn bài hát từ Projects</span>
                  <select id="ss-project-select" value={projectId} disabled={!audioProjects.length} onChange={e => { setProjectId(e.target.value); setFormError(""); }}>
                    <option value="">{audioProjects.length ? "Chọn bài hát có âm thanh gốc" : "Chưa có Project chứa âm thanh gốc"}</option>
                    {audioProjects.map(project => <option key={project.id} value={project.id}>{project.title}</option>)}
                  </select>
                </label>
                <p>Tái sử dụng bản thu trong Project, không cần tải file lên lần nữa.</p>
              </div>
            )}

            <details className="ss-create-settings" open={advancedOpen} onToggle={e => setAdvancedOpen(e.currentTarget.open)}>
              <summary aria-disabled={submitting} onClick={e => { if (submitting) e.preventDefault(); }}>
                <span>Cài đặt lời hát</span>
                <small>{language === "auto" ? "Tự động" : language.toUpperCase()} · {processMode === "align" ? "Căn chỉnh lời" : "AI trích xuất"}</small>
                <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
              </summary>
              <div className="ss-create-settings-body">
                <div className="ss-create-mode-tabs" role="tablist" aria-label="Chế độ xử lý lời bài hát">
                  <button type="button" role="tab" aria-selected={processMode === "extract"} onClick={() => { setProcessMode("extract"); setFormError(""); }}>AI trích xuất lời</button>
                  <button type="button" role="tab" aria-selected={processMode === "align"} onClick={() => { setProcessMode("align"); setFormError(""); }}>Căn chỉnh lời có sẵn</button>
                </div>

                {processMode === "align" && (
                  <div className="ss-create-lyrics-panel">
                    <div className="ss-create-lyrics-head">
                      <label htmlFor="ss-align-lyrics">Lời bài hát gốc</label>
                      <button type="button" onClick={() => lyricsFileInputRef.current?.click()}>Nhập .TXT / .LRC / .SRT</button>
                      <input ref={lyricsFileInputRef} type="file" accept=".txt,.lrc,.srt" hidden onChange={e => { void importLyricsFile(e.target.files?.[0]); e.target.value = ""; }} />
                    </div>
                    <textarea id="ss-align-lyrics" rows={4} value={userLyricsText} onChange={e => { setUserLyricsText(e.target.value); setFormError(""); }} placeholder="Dán lời chuẩn vào đây, mỗi câu một dòng. AI giữ nguyên lời và căn thời gian theo giọng hát." />
                  </div>
                )}

                {sourceTab === "media" && (
                  <label className="ss-create-field" htmlFor="ss-create-title">
                    <span>Tên Songscription (tùy chọn)</span>
                    <input id="ss-create-title" type="text" value={customTitle} onChange={e => setCustomTitle(e.target.value)} placeholder="Để trống để dùng tên file hoặc tiêu đề YouTube" />
                  </label>
                )}
                <label className="ss-create-field" htmlFor="ss-lang-select">
                  <span>Ngôn ngữ bài hát</span>
                  <select id="ss-lang-select" value={language} onChange={e => setLanguage(e.target.value)}>
                    {LANGUAGE_OPTIONS.map(([code, label]) => <option key={code} value={code}>{label}</option>)}
                  </select>
                </label>
                <label className="ss-create-field" htmlFor="ss-model-select">
                  <span>Mô hình nhận diện AI</span>
                  <select id="ss-model-select" value={selectedModelId} onChange={e => setModelId(e.target.value)}>
                    {availableModels.map(model => <option key={model.id} value={model.id}>{model.label}</option>)}
                  </select>
                </label>
                <label className="ss-create-vocals">
                  <input type="checkbox" checked={isolateVocals && vocalAvailable} disabled={!vocalAvailable} onChange={e => setIsolateVocals(e.target.checked)} />
                  <span>Tách riêng giọng hát AI<small>{vocalAvailable ? "BS-RoFormer · Khuyên dùng" : "Sử dụng bộ lọc tần số FFmpeg"}</small></span>
                </label>
              </div>
            </details>
          </fieldset>

          <p className="ss-create-note">Dùng bản thu có giọng hát. Lời do AI nhận diện có thể cần rà soát lại một số từ.</p>
          {formError && <p className="ss-create-error" role="alert">{formError}</p>}
          <button className="ss-create-submit" type="submit" disabled={submitting}>
            {submitting ? "Đang khởi tạo tác vụ…" : processMode === "align" ? "Bắt đầu đồng bộ" : "Bắt đầu trích xuất"}
          </button>
        </form>
      </section>
    </div>
  );
}
