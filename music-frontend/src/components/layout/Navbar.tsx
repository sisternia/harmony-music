import { useEffect, useState } from "react";

import { clearAuthSession, loadAuthSession, subscribeAuthSession, type AuthUserSession } from "@/features/auth/api";
import type { ProjectSummary } from "@/features/studio/api";
import { playNavbarFeedback } from "./navbar-feedback";

type WorkspaceNavbarProps = {
  path: string;
  navigate: (to: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  onAuth: () => void;
  projects?: ProjectSummary[];
};

function PianoMark() {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><rect x="2" y="3" width="20" height="18" rx="2" /><path d="M7 3v18M12 3v18M17 3v18M5.5 3v9h3V3m2 0v9h3V3m2 0v9h3V3" /></svg>;
}

function SongscriptionMark() {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h5M8 16h4M16.5 11.5v5" /><circle cx="15.1" cy="16.8" r="1.5" fill="currentColor" stroke="none" /></svg>;
}

export default function Navbar({ path, navigate, collapsed, onToggle, onAuth, projects }: WorkspaceNavbarProps) {
  const projectList = Array.isArray(projects) ? projects : [];
  const isProject = path === "/projects" || path.startsWith("/p/");
  const isSongscription = path === "/songscription" || path.startsWith("/songscription/");
  const [authUser, setAuthUser] = useState<AuthUserSession | null>(() => loadAuthSession());

  useEffect(() => {
    return subscribeAuthSession(() => setAuthUser(loadAuthSession()));
  }, []);

  const handleProfileClick = () => {
    if (authUser) {
      if (confirm(`Đăng xuất khỏi tài khoản ${authUser.email}?`)) {
        clearAuthSession();
      }
      return;
    }
    onAuth();
  };

  const avatarChar = authUser?.email ? authUser.email.trim()[0].toUpperCase() : "A";

  return <aside className="studio-sidebar" onClickCapture={playNavbarFeedback}>
    <div className="studio-brand">
      <img src="/images/harmony-auralis-logo.png" alt="Harmony Auralis" />
      <button type="button" onClick={onToggle} aria-label={collapsed ? "Mở thanh bên" : "Thu gọn thanh bên"} aria-expanded={!collapsed}>◫</button>
    </div>
    <nav className="studio-navigation" aria-label="Không gian âm nhạc">
      <button className={path === "/" ? "active" : ""} aria-current={path === "/" ? "page" : undefined} onClick={() => navigate("/")}><b>＋</b><span>New transcription</span></button>
      <button className={path === "/piano" ? "active" : ""} aria-current={path === "/piano" ? "page" : undefined} onClick={() => navigate("/piano")}><b><PianoMark /></b><span>Piano</span></button>
      <button className={isSongscription ? "active" : ""} aria-current={isSongscription ? "page" : undefined} onClick={() => navigate("/songscription")}><b><SongscriptionMark /></b><span>Songscription</span></button>
      <div className="studio-nav-divider" />
      <button className={isProject ? "active" : ""} aria-current={isProject ? "page" : undefined} onClick={() => navigate("/projects")}><b>♫</b><span>Projects</span><em>{projectList.length}</em></button>
      <button className={path === "/collections" ? "active" : ""} aria-current={path === "/collections" ? "page" : undefined} onClick={() => navigate("/collections")}><b>▥</b><span>Collections</span></button>
    </nav>
    <div className="studio-recent">
      <div className="studio-section-label">RECENT PROJECTS</div>
      {projectList.slice(0, 5).map(project => <button key={project.id} onClick={() => navigate(`/p/${encodeURIComponent(project.id)}`)} title={project.title}><i>♫</i><span>{project.title}</span></button>)}
      {!projectList.length && <p>Bản nhạc của bạn sẽ xuất hiện ở đây.</p>}
    </div>
    <div className="studio-sidebar-bottom">
      <button className="studio-profile" onClick={handleProfileClick} title={authUser ? `${authUser.email} (Bấm để đăng xuất)` : "Đăng nhập / Đăng ký"}>
        <span className="studio-avatar">{avatarChar}</span>
        <span>
          {authUser ? authUser.email.split("@")[0] : "Đăng nhập"}
          <small>{authUser ? "Đã đăng nhập · Đăng xuất" : "Tạo tài khoản miễn phí"}</small>
        </span>
        <b>{authUser ? "⎋" : "⌄"}</b>
      </button>
    </div>
  </aside>;
}
