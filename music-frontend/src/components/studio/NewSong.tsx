// 新しい曲の追加 (音源だけなら tsumugi で採譜、MIDI も一緒なら採譜を飛ばす)。曲の一覧と、上の帯から開くダイアログで使う
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { API_VERSION, api, type Config, type ProjectSummary } from "../../features/studio/api";
import { useT } from "../../i18n";
import { coverBackgrounds } from "../../theme/colors";
import { Icon } from "../ui/StudioIcon";

const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac)$/i;
const MIDI = /\.(mid|midi)$/i;

export function NewSongForm({ config, onCreated }: { config: Config; onCreated: (pid: string) => void }) {
  const qc = useQueryClient();
  const t = useT();
  const [audio, setAudio] = useState<File | null>(null);
  const [midi, setMidi] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [youtube, setYoutube] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = (files: FileList | File[]) => {
    for (const f of Array.from(files)) {
      if (f.size > 100 * 1024 * 1024) {
        setError("Tệp cần có dung lượng không quá 100 MB.");
        continue;
      }
      if (AUDIO.test(f.name)) {
        setError(null);
        setAudio(f);
        setYoutube("");
        if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
      } else if (MIDI.test(f.name)) {
        setError(null);
        setMidi(f);
        setYoutube("");
        if (!title && !audio) setTitle(f.name.replace(/\.[^.]+$/, ""));
      } else setError(t.unsupportedFile(f.name));
    }
  };

  const create = async () => {
    if (!audio && !midi && !youtube.trim()) {
      setError("Hãy chọn tệp âm thanh hoặc MIDI, hoặc dán link YouTube.");
      return;
    }
    if (!audio && !midi && config.api_version !== API_VERSION) {
      setError("Phiên bản backend hiện tại chưa hỗ trợ link YouTube. Hãy khởi động lại backend rồi thử lại.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      if (audio) form.append("audio", audio);
      if (midi) form.append("midi", midi);
      if (audio || midi) form.append("title", title);
      else form.append("youtube", youtube.trim());
      const detail = await api.createProject(form);
      await qc.invalidateQueries({ queryKey: ["projects"] });
      onCreated(detail.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="new-song-form" onSubmit={(event) => { event.preventDefault(); void create(); }}>
      <section
        className={`upload-zone new-song-drop ${over ? "dragging" : ""}`}
        aria-label="Khu vực tải tệp âm thanh hoặc MIDI lên"
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
      >
        <svg className="upload-cloud" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M15 34H12a8 8 0 0 1-1-16 14 14 0 0 1 26 3 7 7 0 0 1-1 13h-3" /><path d="M24 39V24m-6 6 6-6 6 6" /></svg>
        <button type="button" className="new-song-file-button" onClick={() => input.current?.click()}>
          {audio || midi ? [audio?.name, midi?.name].filter(Boolean).join(" · ") : "Nhập file âm thanh hoặc bản nhạc"}
        </button>
        <span>Hoặc kéo và thả file vào đây</span>
        <small>Âm thanh hoặc MIDI · tối đa 100 MB</small>
        <input
          ref={input}
          type="file"
          multiple
          hidden
          accept=".mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mid,.midi"
          onChange={(e) => {
            if (e.target.files) take(e.target.files);
            e.target.value = "";
          }}
        />
      </section>
      {(audio || midi) && <input className="upload-title new-song-title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder={t.songTitle} aria-label={t.songTitle} />}
      <div className="upload-divider"><span>HOẶC</span></div>
      <label className="youtube-field new-song-youtube">
        <svg viewBox="0 0 28 20" aria-hidden="true"><rect x="1" y="1" width="26" height="18" rx="5" fill="currentColor" /><path d="m11 6 7 4-7 4z" fill="var(--on-accent)" /></svg>
        <input value={youtube} onChange={(event) => { setYoutube(event.target.value); if (event.target.value) { setAudio(null); setMidi(null); setTitle(""); } }} placeholder="Dán link YouTube tại đây" aria-label="Link YouTube" />
      </label>
      <p className="new-song-help">Tối đa 20 phút mỗi bài. Chuyển nhạc có thể mất vài phút; phổ AI có thể cần chỉnh lại một số nốt.</p>
      <button className="upload-submit" type="submit" disabled={busy}>
        {busy ? "Đang chuyển thành piano…" : "Chuyển thành piano"}
      </button>
      {error && (
        <div className="upload-error" role="alert" onClick={() => setError(null)}>
          {error}
        </div>
      )}
    </form>
  );
}

/** 上の帯の「新しい曲」から開くダイアログ */
export function NewSongDialog({
  config,
  onClose,
  onCreated,
}: {
  config: Config;
  onClose: () => void;
  onCreated: (pid: string) => void;
}) {
  const t = useT();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop new-song-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <section className="transcription-entry new-song-modal" role="dialog" aria-modal="true" aria-labelledby="new-song-heading">
        <button type="button" className="new-song-close" onClick={onClose} title={t.close} aria-label={t.close}>
          <Icon name="close" size={16} />
        </button>
        <span className="studio-eyebrow new-song-eyebrow">HARMONY AURALIS · PIANO AI</span>
        <h2 id="new-song-heading">Biến bản nhạc thành piano</h2>
        <p className="new-song-intro">Chọn một bản thu hoặc video YouTube. AI sẽ chuyển thành bản nhạc piano để bạn nghe và tập theo.</p>
        <NewSongForm config={config} onCreated={onCreated} />
      </section>
    </div>
  );
}

export function songStatus(p: ProjectSummary, t: ReturnType<typeof useT>): string {
  const state = p.transcribe?.state;
  if (!p.has_source) {
    if (state === "queued") return t.statusQueued;
    if (state === "running") return t.statusRunning;
    if (state === "error") return t.statusFailed;
    return t.statusNoSource;
  }
  const parts = [t.takesCount(p.takes)];
  if (p.favorites) parts.push(`★ ${p.favorites}`);
  if (p.busy) parts.push(t.generatingNow);
  return parts.join(" · ");
}

export function songArt(title: string): React.CSSProperties {
  let index = 0;
  for (const c of title) index = (index * 31 + c.charCodeAt(0)) % coverBackgrounds.length;
  return { background: coverBackgrounds[index] };
}
