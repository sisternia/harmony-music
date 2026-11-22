import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, type ProjectSummary } from "../api";
import AuthDialog from "../../auth/screen/AuthDialog";
import { songStatus } from "../../../components/studio/NewSong";
import { useT } from "../../../i18n";
import PianoKeyboard from "../../user/piano/screen/PianoKeyboard";
import { Songscription } from "./Songscription";
import Navbar from "../../../components/layout/Navbar";
import "../../../styles/library-home.css";

const logo = "/images/harmony-auralis-logo.png";
const notes = ["♪", "♫", "♬", "♩", "♭", "♯", "𝄞", "♫", "♪", "♬", "♭", "♩", "♫", "♯", "♬", "♪"];
const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac)$/i;
const MIDI = /\.(mid|midi)$/i;
type View = "new" | "projects" | "collections";

export function Library({ path, navigate, youtubeSupported }: { path: string; navigate: (to: string) => void; youtubeSupported: boolean }) {
  const qc = useQueryClient();
  const t = useT();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, refetchInterval: (q) => q.state.data?.some((p) => p.busy) ? 2000 : false });
  const view: View = path === "/projects" ? "projects" : path === "/collections" ? "collections" : "new";
  const [collapsed, setCollapsed] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [youtube, setYoutube] = useState("");
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    if (!AUDIO.test(candidate.name) && !MIDI.test(candidate.name)) { setError("Chỉ hỗ trợ file âm thanh hoặc MIDI."); return; }
    if (candidate.size > 100 * 1024 * 1024) { setError("File phải nhỏ hơn 100 MB."); return; }
    setFile(candidate);
    setYoutube("");
    setTitle(candidate.name.replace(/\.[^.]+$/, ""));
    setError("");
  };

  const createProject = async () => {
    if (!file && !youtube.trim()) {
      setError("Hãy chọn file âm thanh, MIDI hoặc dán link YouTube.");
      return;
    }
    if (!file && !youtubeSupported) {
      setError("Backend đang chạy phiên bản cũ. Hãy khởi động lại backend bằng lệnh python -m cover_studio rồi thử lại link YouTube.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      if (file) {
        form.append(MIDI.test(file.name) ? "midi" : "audio", file);
        form.append("title", title.trim() || file.name.replace(/\.[^.]+$/, ""));
      } else {
        form.append("youtube", youtube.trim());
      }
      const project = await api.createProject(form);
      await qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/p/${encodeURIComponent(project.id)}`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  const remove = async (project: ProjectSummary) => {
    if (!confirm(t.confirmDeleteSong(project.title, project.takes))) return;
    try { await api.deleteProject(project.id); await qc.invalidateQueries({ queryKey: ["projects"] }); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };

  const allProjects = projects.data ?? [];
  const visible = allProjects.filter((project) => project.title.toLowerCase().includes(search.toLowerCase()) && (view !== "collections" || project.favorites > 0));
  const openProject = (project: ProjectSummary) => navigate(`/p/${encodeURIComponent(project.id)}`);

  return <div className={`auralis-workspace cover-home ${collapsed ? "sidebar-collapsed" : ""}`}>
    <Navbar path={path} navigate={navigate} projects={allProjects} collapsed={collapsed} onToggle={() => setCollapsed(value => !value)} onAuth={() => setAuthOpen(true)} onSearch={() => { navigate("/projects"); requestAnimationFrame(() => searchInput.current?.focus()); }} />
    <main className="studio-main">
      {path === "/piano" ? (
        <PianoKeyboard workspace />
      ) : path === "/songscription" || path.startsWith("/songscription/") ? (
        <Songscription path={path} navigate={navigate} projects={allProjects} youtubeSupported={youtubeSupported} />
      ) : <>
      {view === "new" ? <div className="studio-new">
        <div className="studio-topline"><span>YOUR PERSONAL MUSIC STUDIO</span><span>Âm nhạc, theo cách của bạn <i>✦</i></span></div>
        <div className="studio-welcome"><img src={logo} alt="Harmony Auralis" /><span className="studio-eyebrow">FROM SOUND TO SOMETHING BEAUTIFUL</span><h1>Hôm nay, bạn muốn<br /><em>chơi bản nhạc nào?</em></h1><p>Một bản thu. Một đường link. Cả thế giới âm nhạc của bạn.</p></div>
        <div className="studio-import-stage">
          <svg className="studio-wave" viewBox="0 0 500 220" aria-hidden="true">{Array.from({ length: 56 }, (_, index) => { const height = 22 + (Math.sin(index * 1.73) + 1) * 45 + (56 - index) * 1.5; return <rect key={index} x={index * 9} y={(220 - height) / 2} width="5" height={height} rx="3" style={{ animationDelay: `${-index * 35}ms` }} />; })}</svg>
          <div className="studio-notes-decoration" aria-hidden="true">{notes.map((note, index) => <span key={index} style={{ top: `${5 + (index * 37) % 86}%`, fontSize: `${30 + (index % 5) * 8}px`, animationDelay: `${-index * 1.8}s` }}>{note}</span>)}</div>
          <div className="transcription-entry">
            <div className={`upload-zone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}>
              <svg className="upload-cloud" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M15 34H12a8 8 0 0 1-1-16 14 14 0 0 1 26 3 7 7 0 0 1-1 13h-3" /><path d="M24 39V24m-6 6 6-6 6 6" /></svg>
              <button type="button" onClick={() => fileInput.current?.click()}>{file ? file.name : "Nhập file âm thanh hoặc bản nhạc"}</button>
              <span>Hoặc kéo và thả file vào đây</span><small>Âm thanh hoặc MIDI · tối đa 100 MB</small>
              <input ref={fileInput} type="file" accept=".mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mid,.midi" onChange={(event) => { chooseFile(event.target.files?.[0]); event.target.value = ""; }} />
            </div>
            {file && <input className="upload-title" value={title} onChange={(event) => setTitle(event.target.value)} aria-label="Tên bản nhạc" placeholder="Tên bản nhạc" />}
            <div className="upload-divider"><span>HOẶC</span></div>
            <label className="youtube-field"><span aria-hidden="true">▶</span><input value={youtube} onChange={(event) => { setYoutube(event.target.value); if (event.target.value) { setFile(null); setTitle(""); } }} placeholder="Dán link YouTube tại đây" aria-label="Link YouTube" /></label>
            <p>Tối đa 20 phút mỗi bài. Chuyển nhạc có thể mất vài phút; phổ AI có thể cần chỉnh lại một số nốt.</p>
            <button className="upload-submit" type="button" disabled={busy} onClick={() => void createProject()}>{busy ? "Đang tải lên…" : "Chuyển thành piano"}</button>
            {error && <div className="upload-error" role="alert">{error}</div>}
          </div>
        </div>
        <div className="studio-process"><span><b>01</b> Nhập bản nhạc</span><i>⟶</i><span><b>02</b> Để AI lắng nghe</span><i>⟶</i><span><b>03</b> Chơi &amp; sáng tạo</span></div>
        <footer className="studio-page-footer"><span>CRAFTED FOR THE WAY YOU MAKE MUSIC.</span><span>Harmony Auralis Studio</span></footer>
      </div> : <section className="studio-projects"><div className="studio-projects-header"><span className="studio-eyebrow">YOUR MUSIC, IN ONE PLACE</span><h1>{view === "collections" ? "Bộ sưu tập của bạn" : "Những bản nhạc của bạn"}</h1><p>{view === "collections" ? "Những dự án có bản phối yêu thích." : "Các bản nhạc đã tải lên và tạo bằng Tsumugi."}</p></div><div className="studio-projects-actions"><input ref={searchInput} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm tên bản nhạc…" aria-label="Tìm dự án" /><button onClick={() => navigate("/")}>＋ New transcription</button></div>{error && <div className="upload-error" role="alert">{error}</div>}{projects.isError && <div className="upload-error" role="alert">{projects.error.message}</div>}{projects.isPending ? <p>Đang tải dự án…</p> : visible.length ? <div className="studio-project-grid">{visible.map((project) => <article key={project.id}><button className="studio-project-open" onClick={() => openProject(project)}><span className="studio-project-cover">♫</span><span><strong>{project.title}</strong><small>{songStatus(project, t)}</small></span></button><button className="studio-project-delete" onClick={() => void remove(project)} aria-label={`Xóa ${project.title}`}>×</button></article>)}</div> : <div className="studio-project-empty">{view === "collections" ? "Chưa có bản phối yêu thích." : "Chưa có bản nhạc. Hãy tải file để bắt đầu."}</div>}</section>}
      </>}
    </main>
    {authOpen && <AuthDialog open onClose={() => setAuthOpen(false)} logo={logo} />}
  </div>;
}
