import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useRef, useState } from "react";
import { api, type ProjectSummary } from "../api";
import AuthDialog from "../../auth/screen/AuthDialog";
import { songStatus } from "../../../components/studio/NewSong";
import { useT } from "../../../i18n";
import PianoKeyboard from "../../user/piano/screen/PianoKeyboard";
import { Songscription } from "./Songscription";
import Navbar from "../../../components/layout/Navbar";
import Icon from "../../../components/ui/Icon";
import { colors } from "../../../theme/colors";
import "../../../styles/library-home.css";
import "../styles/songscription-youtube.css";

const logo = "/images/harmony-auralis-logo.png";
const notes = ["♪", "♫", "♬", "♩", "♭", "♯", "𝄞", "♫", "♪", "♬", "♭", "♩", "♫", "♯", "♬", "♪"];
const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac)$/i;
const MIDI = /\.(mid|midi)$/i;
type View = "new" | "projects" | "collections";
type ProjectFilter = "all" | "busy" | "ready" | "favorites" | "draft";
type ProjectSort = "recent" | "oldest" | "title";

type DerivedProjectStatus = {
  key: Exclude<ProjectFilter, "all" | "favorites">;
  label: string;
  tone: "running" | "done" | "warn" | "error";
};

function deriveProjectStatus(project: ProjectSummary): DerivedProjectStatus {
  if (project.busy || project.transcribe?.state === "running" || project.transcribe?.state === "queued") {
    return { key: "busy", label: "Đang xử lý", tone: "running" };
  }
  if (project.transcribe?.state === "error") {
    return { key: "draft", label: "Thất bại", tone: "error" };
  }
  if (project.takes > 0) {
    return { key: "ready", label: "Đã có bản phối", tone: "done" };
  }
  return { key: "draft", label: project.has_source ? "Sẵn sàng phối" : "Chưa xử lý", tone: "warn" };
}

function formatProjectDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function Library({
  path,
  navigate,
  youtubeSupported,
  isVisible = true,
}: {
  path: string;
  navigate: (to: string) => void;
  youtubeSupported: boolean;
  isVisible?: boolean;
}) {
  const qc = useQueryClient();
  const t = useT();
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.projects,
    refetchInterval: q => (q.state.data?.some(p => p.busy) ? 2000 : false),
  });
  const view: View = path === "/projects" ? "projects" : path === "/collections" ? "collections" : "new";
  const [collapsed, setCollapsed] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [youtube, setYoutube] = useState("");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<ProjectFilter>("all");
  const [sortBy, setSortBy] = useState<ProjectSort>("recent");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ProjectSummary | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    if (!AUDIO.test(candidate.name) && !MIDI.test(candidate.name)) {
      setError("Chỉ hỗ trợ file âm thanh hoặc MIDI.");
      return;
    }
    if (candidate.size > 100 * 1024 * 1024) {
      setError("File phải nhỏ hơn 100 MB.");
      return;
    }
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
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const confirmDeleteProject = async () => {
    if (!deleteTarget || deleteBusy) return;
    setDeleteBusy(true);
    setError("");
    try {
      await api.deleteProject(deleteTarget.id);
      await qc.invalidateQueries({ queryKey: ["projects"] });
      setDeleteTarget(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setDeleteBusy(false);
    }
  };

  const allProjects = useMemo(() => projects.data ?? [], [projects.data]);
  const baseProjects = useMemo(
    () => (view === "collections" ? allProjects.filter(p => p.favorites > 0) : allProjects),
    [allProjects, view],
  );

  const projectCounts = useMemo(() => {
    const counts = { all: baseProjects.length, busy: 0, ready: 0, favorites: 0, draft: 0 };
    for (const p of baseProjects) {
      const st = deriveProjectStatus(p);
      counts[st.key] += 1;
      if (p.favorites > 0) counts.favorites += 1;
    }
    return counts;
  }, [baseProjects]);

  const visibleProjects = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = baseProjects.filter(project => {
      if (statusFilter === "favorites") {
        if (project.favorites <= 0) return false;
      } else if (statusFilter !== "all") {
        if (deriveProjectStatus(project).key !== statusFilter) return false;
      }
      if (!q) return true;
      return project.title.toLowerCase().includes(q);
    });
    const sorted = [...filtered];
    sorted.sort((a, b) => {
      if (sortBy === "title") return a.title.localeCompare(b.title, "vi");
      const ta = new Date(a.created || 0).getTime();
      const tb = new Date(b.created || 0).getTime();
      return sortBy === "oldest" ? ta - tb : tb - ta;
    });
    return sorted;
  }, [baseProjects, search, statusFilter, sortBy]);

  const openProject = (project: ProjectSummary) => navigate(`/p/${encodeURIComponent(project.id)}`);
  const isSongscriptionRoute = path === "/songscription" || path.startsWith("/songscription/");

  return (
    <div className={`auralis-workspace cover-home ${collapsed ? "sidebar-collapsed" : ""}`}>
      <Navbar
        path={path}
        navigate={navigate}
        projects={allProjects}
        collapsed={collapsed}
        onToggle={() => setCollapsed(value => !value)}
        onAuth={() => setAuthOpen(true)}
      />
      <main className="studio-main">
        {!isSongscriptionRoute ? (
          path === "/piano" ? (
            <PianoKeyboard workspace />
          ) : view === "new" ? (
          <div className="studio-new">
            <div className="studio-topline">
              <span>YOUR PERSONAL MUSIC STUDIO</span>
              <span>
                Âm nhạc, theo cách của bạn <i>✦</i>
              </span>
            </div>
            <div className="studio-welcome">
              <img src={logo} alt="Harmony Auralis" />
              <span className="studio-eyebrow">FROM SOUND TO SOMETHING BEAUTIFUL</span>
              <h1>
                Hôm nay, bạn muốn
                <br />
                <em>chơi bản nhạc nào?</em>
              </h1>
              <p>Một bản thu. Một đường link. Cả thế giới âm nhạc của bạn.</p>
            </div>
            <div className="studio-import-stage">
              <svg className="studio-wave" viewBox="0 0 500 220" aria-hidden="true">
                {Array.from({ length: 56 }, (_, index) => {
                  const height = 22 + (Math.sin(index * 1.73) + 1) * 45 + (56 - index) * 1.5;
                  return (
                    <rect
                      key={index}
                      x={index * 9}
                      y={(220 - height) / 2}
                      width="5"
                      height={height}
                      rx="3"
                      style={{ animationDelay: `${-index * 35}ms` }}
                    />
                  );
                })}
              </svg>
              <div className="studio-notes-decoration" aria-hidden="true">
                {notes.map((note, index) => (
                  <span
                    key={index}
                    style={{
                      top: `${5 + ((index * 37) % 86)}%`,
                      fontSize: `${30 + (index % 5) * 8}px`,
                      animationDelay: `${-index * 1.8}s`,
                    }}
                  >
                    {note}
                  </span>
                ))}
              </div>
              <div className="transcription-entry">
                <div
                  className={`upload-zone ${dragging ? "dragging" : ""}`}
                  onDragOver={event => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={event => {
                    event.preventDefault();
                    setDragging(false);
                    chooseFile(event.dataTransfer.files[0]);
                  }}
                >
                  <svg
                    className="upload-cloud"
                    viewBox="0 0 48 48"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M15 34H12a8 8 0 0 1-1-16 14 14 0 0 1 26 3 7 7 0 0 1-1 13h-3" />
                    <path d="M24 39V24m-6 6 6-6 6 6" />
                  </svg>
                  <button type="button" onClick={() => fileInput.current?.click()}>
                    {file ? file.name : "Nhập file âm thanh hoặc bản nhạc"}
                  </button>
                  <span>Hoặc kéo và thả file vào đây</span>
                  <small>Âm thanh hoặc MIDI · tối đa 100 MB</small>
                  <input
                    ref={fileInput}
                    type="file"
                    accept=".mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mid,.midi"
                    onChange={event => {
                      chooseFile(event.target.files?.[0]);
                      event.target.value = "";
                    }}
                  />
                </div>
                {file && (
                  <input
                    className="upload-title"
                    value={title}
                    onChange={event => setTitle(event.target.value)}
                    aria-label="Tên bản nhạc"
                    placeholder="Tên bản nhạc"
                  />
                )}
                <div className="upload-divider">
                  <span>HOẶC</span>
                </div>
                <label className="youtube-field">
                  <span aria-hidden="true">▶</span>
                  <input
                    value={youtube}
                    onChange={event => {
                      setYoutube(event.target.value);
                      if (event.target.value) {
                        setFile(null);
                        setTitle("");
                      }
                    }}
                    placeholder="Dán link YouTube tại đây"
                    aria-label="Link YouTube"
                  />
                </label>
                <p>Tối đa 20 phút mỗi bài. Chuyển nhạc có thể mất vài phút; phổ AI có thể cần chỉnh lại một số nốt.</p>
                <button className="upload-submit" type="button" disabled={busy} onClick={() => void createProject()}>
                  {busy ? "Đang tải lên…" : "Chuyển thành piano"}
                </button>
                {error && (
                  <div className="upload-error" role="alert">
                    {error}
                  </div>
                )}
              </div>
            </div>
            <div className="studio-process">
              <span>
                <b>01</b> Nhập bản nhạc
              </span>
              <i>⟶</i>
              <span>
                <b>02</b> Để AI lắng nghe
              </span>
              <i>⟶</i>
              <span>
                <b>03</b> Chơi &amp; sáng tạo
              </span>
            </div>
            <footer className="studio-page-footer">
              <span>CRAFTED FOR THE WAY YOU MAKE MUSIC.</span>
              <span>Harmony Auralis Studio</span>
            </footer>
          </div>
        ) : (
          /* Unified Library Layout — matches SongscriptionLibraryView 1:1 */
          <div className="songscription-studio ss-library-page">
            <header className="ss-lib-header">
              <div className="ss-lib-title-block">
                <span className="studio-eyebrow">
                  {view === "collections" ? "HARMONY AURALIS · COLLECTIONS" : "HARMONY AURALIS · PIANO LIBRARY"}
                </span>
                <h1>
                  {view === "collections" ? (
                    <>
                      Collections — <em>Bộ sưu tập yêu thích</em>
                    </>
                  ) : (
                    <>
                      Projects — <em>Những bản nhạc của bạn</em>
                    </>
                  )}
                </h1>
                <p className="ss-lib-subtitle">
                  {view === "collections"
                    ? "Tất cả các dự án có bản phối piano đã được đánh dấu yêu thích."
                    : "Quản lý, tìm kiếm và mở lại toàn bộ các bản nhạc đã tải lên và tạo bằng Tsumugi."}
                </p>
              </div>

              <button
                type="button"
                className="ss-primary-btn ss-lib-create-btn"
                onClick={() => navigate("/")}
              >
                ＋ Tạo bản nhạc mới
              </button>
            </header>

            {baseProjects.length > 0 && (
              <section className="ss-lib-toolbar" aria-label="Tìm kiếm và lọc thư viện bản nhạc">
                <div className="ss-lib-search-row">
                  <label className="ss-lib-search-wrap" htmlFor="projects-lib-search">
                    <span className="ss-lib-search-icon" aria-hidden="true">
                      ⌕
                    </span>
                    <input
                      id="projects-lib-search"
                      type="search"
                      value={search}
                      onChange={e => setSearch(e.target.value)}
                      placeholder="Tìm theo tên bản nhạc…"
                    />
                  </label>

                  <label className="ss-lib-sort-wrap" htmlFor="projects-lib-sort">
                    <span>Sắp xếp</span>
                    <select
                      id="projects-lib-sort"
                      value={sortBy}
                      onChange={e => setSortBy(e.target.value as ProjectSort)}
                    >
                      <option value="recent">Mới nhất</option>
                      <option value="oldest">Cũ nhất</option>
                      <option value="title">Theo tên (A–Z)</option>
                    </select>
                  </label>
                </div>

                {view !== "collections" && (
                  <div className="ss-lib-filter-chips" role="tablist" aria-label="Lọc theo trạng thái">
                    {(
                      [
                        ["all", "Tất cả", projectCounts.all],
                        ["busy", "Đang xử lý", projectCounts.busy],
                        ["ready", "Đã có bản phối", projectCounts.ready],
                        ["favorites", "Yêu thích", projectCounts.favorites],
                        ["draft", "Chưa phối", projectCounts.draft],
                      ] as [ProjectFilter, string, number][]
                    ).map(([key, label, count]) => (
                      <button
                        key={key}
                        type="button"
                        role="tab"
                        aria-selected={statusFilter === key}
                        className={`ss-filter-pill ${statusFilter === key ? "active" : ""}`}
                        onClick={() => setStatusFilter(key)}
                      >
                        <span>{label}</span>
                        <em>{count}</em>
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}

            {error && (
              <div className="upload-error" role="alert">
                {error}
              </div>
            )}

            <section className="ss-lib-content" aria-live="polite">
              {projects.isPending ? (
                <div className="ss-lib-state-card">
                  <span className="songscription-pulse" aria-hidden="true" />
                  <h2>Đang tải danh sách bản nhạc…</h2>
                  <p>Hệ thống đang đọc các dự án đã lưu trong thư viện.</p>
                </div>
              ) : projects.isError ? (
                <div className="ss-lib-state-card is-error" role="alert">
                  <h2>Không thể tải danh sách bản nhạc</h2>
                  <p>{projects.error.message}</p>
                  <button type="button" className="ss-secondary-btn" onClick={() => void projects.refetch()}>
                    Thử tải lại
                  </button>
                </div>
              ) : baseProjects.length === 0 ? (
                <div className="ss-lib-empty-card">
                  <div className="songscription-empty-badge" aria-hidden="true">
                    ♫
                  </div>
                  <h2>{view === "collections" ? "Chưa có bản phối yêu thích nào" : "Chưa có bản nhạc nào"}</h2>
                  <p>
                    {view === "collections"
                      ? "Đánh dấu sao trên bản phối piano bạn thích để lưu vào bộ sưu tập này."
                      : "Tải lên tệp âm thanh, MIDI hoặc dán đường dẫn YouTube để tạo bản nhạc piano đầu tiên."}
                  </p>
                  <button type="button" className="ss-primary-btn" onClick={() => navigate("/")}>
                    ＋ Tạo bản nhạc đầu tiên
                  </button>
                </div>
              ) : visibleProjects.length === 0 ? (
                <div className="ss-lib-state-card">
                  <h2>Không tìm thấy kết quả phù hợp</h2>
                  <p>Thử từ khóa khác hoặc bỏ bộ lọc trạng thái để xem toàn bộ {baseProjects.length} bản nhạc.</p>
                  <button
                    type="button"
                    className="ss-secondary-btn"
                    onClick={() => {
                      setSearch("");
                      setStatusFilter("all");
                    }}
                  >
                    Hiển thị tất cả
                  </button>
                </div>
              ) : (
                <div className="ss-lib-list">
                  {visibleProjects.map(project => {
                    const status = deriveProjectStatus(project);
                    const dateLabel = formatProjectDate(project.created);
                    const sourceLabel = project.audio ? "Âm thanh gốc" : "Tệp MIDI";

                    return (
                      <article key={project.id} className={`ss-lib-card status-${status.tone}`}>
                        <div className="ss-lib-card-main">
                          <div className="ss-lib-card-icon" aria-hidden="true">
                            {status.key === "busy" ? "◌" : status.tone === "error" ? "!" : "♫"}
                          </div>

                          <div className="ss-lib-card-body">
                            <div className="ss-lib-card-top">
                              <button
                                type="button"
                                className="ss-lib-card-title-btn"
                                onClick={() => openProject(project)}
                              >
                                <strong>{project.title}</strong>
                              </button>

                              <span className={`ss-status-badge tone-${status.tone}`}>{status.label}</span>
                            </div>

                            <div className="ss-lib-card-meta">
                              <span className="ss-lib-meta-primary">
                                {sourceLabel} · {songStatus(project, t)}
                              </span>
                              <span className="ss-meta-date">{dateLabel}</span>
                            </div>

                            <div className="ss-lib-card-stats">
                              <span>
                                <strong>{project.takes}</strong> bản phối
                                {project.favorites > 0 && ` · ★ ${project.favorites} yêu thích`}
                                {project.lyrics?.state === "done" && " · Có lời bài hát"}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="ss-lib-card-actions">
                          <button
                            type="button"
                            className="ss-secondary-btn ss-lib-open-btn"
                            onClick={() => openProject(project)}
                          >
                            {status.key === "busy" ? "Xem tiến trình" : "Mở & chỉnh sửa"}
                          </button>

                          <button
                            type="button"
                            className="ss-icon-menu-btn ss-lib-delete-btn"
                            disabled={project.busy}
                            title={project.busy ? "Đang xử lý, chưa thể xóa" : `Xóa ${project.title}`}
                            aria-label={`Xóa ${project.title}`}
                            onClick={() => setDeleteTarget(project)}
                          >
                            <Icon name="trash" color={colors.danger} size={17} />
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        )) : null}
        <div
          key="persistent-songscription-host"
          className={isSongscriptionRoute ? "ss-visible-host" : "ss-bg-dock"}
          aria-hidden={!isSongscriptionRoute || undefined}
        >
          <Songscription
            path={path}
            navigate={navigate}
            projects={allProjects}
            youtubeSupported={youtubeSupported}
            isActiveView={isVisible && isSongscriptionRoute}
          />
        </div>
      </main>
      {authOpen && <AuthDialog open onClose={() => setAuthOpen(false)} logo={logo} />}
      {deleteTarget && (
        <div
          className="ss-confirm-backdrop"
          role="presentation"
          onMouseDown={e => {
            if (e.target === e.currentTarget && !deleteBusy) setDeleteTarget(null);
          }}
        >
          <div
            className="ss-confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="proj-confirm-title"
            aria-describedby="proj-confirm-desc"
          >
            <div className="ss-confirm-head">
              <span className="ss-confirm-badge is-danger" aria-hidden="true">
                <Icon name="trash" color={colors.danger} size={18} />
              </span>
              <h3 id="proj-confirm-title">Xóa bản nhạc</h3>
            </div>
            <p id="proj-confirm-desc" className="ss-confirm-message">
              {t.confirmDeleteSong(deleteTarget.title, deleteTarget.takes)}
            </p>
            <div className="ss-confirm-actions">
              <button
                type="button"
                className="ss-secondary-btn"
                disabled={deleteBusy}
                onClick={() => setDeleteTarget(null)}
              >
                Hủy
              </button>
              <button
                type="button"
                className="ss-danger-btn"
                disabled={deleteBusy}
                onClick={() => void confirmDeleteProject()}
              >
                {deleteBusy ? "Đang xóa…" : "Xóa vĩnh viễn"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
