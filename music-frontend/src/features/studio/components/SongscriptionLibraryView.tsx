import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectSummary, SongscriptionItem } from "../api";

export type LibraryStatusFilter = "all" | "running" | "needs_review" | "reviewed" | "failed";
export type LibrarySort = "recent" | "oldest" | "title";

type DerivedStatus = {
  key: Exclude<LibraryStatusFilter, "all">;
  label: string;
  tone: "running" | "warn" | "done" | "error";
};

function deriveSessionStatus(item: SongscriptionItem): DerivedStatus {
  if (item.state === "queued" || item.state === "running") {
    return { key: "running", label: "Đang xử lý", tone: "running" };
  }
  if (item.state === "error" || item.state === "cancelled") {
    return { key: "failed", label: item.state === "cancelled" ? "Đã dừng" : "Thất bại", tone: "error" };
  }
  const needs = item.review_needs_count ?? 0;
  if (item.segment_count > 0 && needs > 0) {
    return { key: "needs_review", label: "Cần kiểm tra", tone: "warn" };
  }
  if (item.segment_count > 0 && needs === 0) {
    return { key: "reviewed", label: "Đã rà soát", tone: "done" };
  }
  return { key: "reviewed", label: "Đã hoàn thành", tone: "done" };
}
function formatSessionDate(iso?: string | null): string {
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

export type SongscriptionLibraryViewProps = {
  items: SongscriptionItem[];
  projects: ProjectSummary[];
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  onRetryFetch: () => void;
  onCreateNew: () => void;
  onOpenSession: (sid: string) => void;
  onRenameSession: (item: SongscriptionItem, newTitle: string) => Promise<void>;
  onRetrySession: (item: SongscriptionItem) => Promise<void>;
  onDeleteSession: (item: SongscriptionItem) => Promise<void>;
};

export function SongscriptionLibraryView({
  items,
  projects,
  isLoading,
  isError,
  errorMessage,
  onRetryFetch,
  onCreateNew,
  onOpenSession,
  onRenameSession,
  onRetrySession,
  onDeleteSession,
}: SongscriptionLibraryViewProps) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<LibraryStatusFilter>("all");
  const [sortBy, setSortBy] = useState<LibrarySort>("recent");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [busyActionId, setBusyActionId] = useState<string | null>(null);

  const menuContainerRef = useRef<HTMLDivElement>(null);

  const projectMap = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of projects) {
      m.set(p.id, p.title);
    }
    return m;
  }, [projects]);

  useEffect(() => {
    if (!openMenuId) return;
    const onPointerDown = (e: MouseEvent) => {
      if (menuContainerRef.current && !menuContainerRef.current.contains(e.target as Node)) {
        setOpenMenuId(null);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenMenuId(null);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [openMenuId]);

  const statusCounts = useMemo(() => {
    const counts = { all: items.length, running: 0, needs_review: 0, reviewed: 0, failed: 0 };
    for (const item of items) {
      const st = deriveSessionStatus(item);
      counts[st.key] += 1;
    }
    return counts;
  }, [items]);

  const visibleItems = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = items.filter(item => {
      const st = deriveSessionStatus(item);
      if (statusFilter !== "all" && st.key !== statusFilter) return false;
      if (!q) return true;
      const projTitle = (item.project_id && projectMap.get(item.project_id)) || item.project_id || "";
      return (
        item.title.toLowerCase().includes(q) ||
        projTitle.toLowerCase().includes(q) ||
        (item.language_name || "").toLowerCase().includes(q) ||
        (item.language || "").toLowerCase().includes(q)
      );
    });

    const sorted = [...filtered];
    sorted.sort((a, b) => {
      if (sortBy === "title") {
        return a.title.localeCompare(b.title, "vi");
      }
      const ta = new Date(a.updated || a.completed || a.created || 0).getTime();
      const tb = new Date(b.updated || b.completed || b.created || 0).getTime();
      return sortBy === "oldest" ? ta - tb : tb - ta;
    });
    return sorted;
  }, [items, search, statusFilter, sortBy, projectMap]);

  const startRename = (item: SongscriptionItem) => {
    setOpenMenuId(null);
    setRenamingId(item.id);
    setRenameValue(item.title);
  };

  const commitRename = async (item: SongscriptionItem) => {
    const trimmed = renameValue.trim();
    if (!trimmed || trimmed === item.title) {
      setRenamingId(null);
      return;
    }
    setBusyActionId(item.id);
    try {
      await onRenameSession(item, trimmed);
      setRenamingId(null);
    } finally {
      setBusyActionId(null);
    }
  };

  return (
    <div className="songscription-studio ss-library-page">
      {/* Library Header */}
      <header className="ss-lib-header">
        <div className="ss-lib-title-block">
          <span className="studio-eyebrow">HARMONY AURALIS · LYRICS LIBRARY</span>
          <h1>
            Songscription — <em>Thư viện lời bài hát AI</em>
          </h1>
          <p className="ss-lib-subtitle">
            Quản lý, tìm kiếm và rà soát toàn bộ các phiên trích xuất lời bài hát đã lưu.
          </p>
        </div>

        <button type="button" className="ss-primary-btn ss-lib-create-btn" onClick={onCreateNew}>
          ＋ Tạo Songscription
        </button>
      </header>

      {/* Toolbar: Search, Status Filter Chips, Sort */}
      {items.length > 0 && (
        <section className="ss-lib-toolbar" aria-label="Tìm kiếm và lọc thư viện Songscription">
          <div className="ss-lib-search-row">
            <label className="ss-lib-search-wrap" htmlFor="ss-lib-search">
              <span className="ss-lib-search-icon" aria-hidden="true">
                ⌕
              </span>
              <input
                id="ss-lib-search"
                type="search"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Tìm theo tên bài hát, Project liên kết hoặc ngôn ngữ…"
              />
            </label>

            <label className="ss-lib-sort-wrap" htmlFor="ss-lib-sort">
              <span>Sắp xếp</span>
              <select
                id="ss-lib-sort"
                value={sortBy}
                onChange={e => setSortBy(e.target.value as LibrarySort)}
              >
                <option value="recent">Mới nhất</option>
                <option value="oldest">Cũ nhất</option>
                <option value="title">Theo tên (A–Z)</option>
              </select>
            </label>
          </div>

          <div className="ss-lib-filter-chips" role="tablist" aria-label="Lọc theo trạng thái">
            {(
              [
                ["all", "Tất cả", statusCounts.all],
                ["running", "Đang xử lý", statusCounts.running],
                ["needs_review", "Cần kiểm tra", statusCounts.needs_review],
                ["reviewed", "Đã rà soát", statusCounts.reviewed],
                ["failed", "Thất bại", statusCounts.failed],
              ] as [LibraryStatusFilter, string, number][]
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
        </section>
      )}

      {/* Main Content: Loading / Error / Empty / Session Cards */}
      <section className="ss-lib-content" aria-live="polite">
        {isLoading ? (
          <div className="ss-lib-state-card">
            <span className="songscription-pulse" aria-hidden="true" />
            <h2>Đang tải thư viện Songscription…</h2>
            <p>Hệ thống đang đọc danh sách các phiên lời bài hát đã lưu.</p>
          </div>
        ) : isError ? (
          <div className="ss-lib-state-card is-error" role="alert">
            <h2>Không thể tải danh sách Songscription</h2>
            <p>{errorMessage || "Không thể kết nối đến máy chủ backend. Hãy kiểm tra server rồi thử lại."}</p>
            <button type="button" className="ss-secondary-btn" onClick={onRetryFetch}>
              Thử tải lại
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="ss-lib-empty-card">
            <div className="songscription-empty-badge" aria-hidden="true">
              ✦
            </div>
            <h2>Chưa có Songscription nào</h2>
            <p>Trích xuất lời bài hát từ dự án hoặc tải nhạc lên để bắt đầu.</p>
            <button type="button" className="ss-primary-btn" onClick={onCreateNew}>
              ＋ Tạo Songscription đầu tiên
            </button>
          </div>
        ) : visibleItems.length === 0 ? (
          <div className="ss-lib-state-card">
            <h2>Không tìm thấy kết quả phù hợp</h2>
            <p>Thử từ khóa khác hoặc bỏ bộ lọc trạng thái để xem toàn bộ {items.length} phiên đã lưu.</p>
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
          <div className="ss-lib-list" ref={menuContainerRef}>
            {visibleItems.map(item => {
              const status = deriveSessionStatus(item);
              const totalLines = item.segment_count || 0;
              const needsCount = item.review_needs_count ?? 0;
              const doneCount = item.review_done_count ?? Math.max(0, totalLines - needsCount);
              const pct = totalLines > 0 ? Math.round((doneCount / totalLines) * 100) : 0;
              const linkedProjectTitle =
                item.project_id ? projectMap.get(item.project_id) || item.project_id : null;
              const dateLabel = formatSessionDate(item.updated || item.completed || item.created);
              const isBusy = busyActionId === item.id;
              const isMenuOpen = openMenuId === item.id;
              const isRenaming = renamingId === item.id;

              return (
                <article
                  key={item.id}
                  className={`ss-lib-card status-${status.tone}`}
                >
                  <div className="ss-lib-card-main">
                    <div className="ss-lib-card-icon" aria-hidden="true">
                      {status.key === "running" ? "◌" : status.key === "failed" ? "!" : "♪"}
                    </div>

                    <div className="ss-lib-card-body">
                      <div className="ss-lib-card-top">
                        {isRenaming ? (
                          <form
                            className="ss-lib-rename-form"
                            onSubmit={e => {
                              e.preventDefault();
                              void commitRename(item);
                            }}
                          >
                            <input
                              type="text"
                              value={renameValue}
                              onChange={e => setRenameValue(e.target.value)}
                              aria-label="Tên mới của Songscription"
                              autoFocus
                              onKeyDown={e => {
                                if (e.key === "Escape") setRenamingId(null);
                              }}
                            />
                            <button type="submit" className="ss-mini-btn primary" disabled={isBusy}>
                              Lưu
                            </button>
                            <button
                              type="button"
                              className="ss-mini-btn"
                              onClick={() => setRenamingId(null)}
                            >
                              Hủy
                            </button>
                          </form>
                        ) : (
                          <button
                            type="button"
                            className="ss-lib-card-title-btn"
                            onClick={() => onOpenSession(item.id)}
                          >
                            <strong>{item.title}</strong>
                          </button>
                        )}

                        <span className={`ss-status-badge tone-${status.tone}`}>
                          {status.label}
                          {status.key === "running" && typeof item.progress === "number"
                            ? ` · ${Math.round(item.progress * 100)}%`
                            : ""}
                        </span>
                      </div>

                      <div className="ss-lib-card-meta">
                        <span
                          className="ss-lib-meta-primary"
                          title={item.language_name ? `Ngôn ngữ: ${item.language_name}` : undefined}
                        >
                          {linkedProjectTitle
                            ? `Project: ${linkedProjectTitle}`
                            : item.source_kind === "youtube"
                              ? "YouTube"
                              : "Tệp tải lên"}
                          {` · ${item.mode === "align" ? "Căn lời có sẵn" : "AI trích xuất"}`}
                          {item.language_name ? ` · ${item.language_name}` : ""}
                        </span>
                        <span className="ss-meta-date">{dateLabel}</span>
                      </div>

                      {item.state === "done" && totalLines > 0 && (
                        <div className="ss-lib-card-stats">
                          <span>
                            <strong>{totalLines}</strong> câu
                            {needsCount > 0 && (
                              <span className="is-warn"> · {needsCount} cần kiểm tra</span>
                            )}
                          </span>
                          <div
                            className="ss-lib-progress-wrap"
                            role="progressbar"
                            aria-label="Tiến độ rà soát"
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={pct}
                          >
                            <div className="ss-lib-progress-bar">
                              <span style={{ width: `${pct}%` }} />
                            </div>
                            <small>{pct}%</small>
                          </div>
                        </div>
                      )}

                      {item.state === "error" && item.error && (
                        <p className="ss-lib-card-error" role="alert">
                          {item.error}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="ss-lib-card-actions">
                    <button
                      type="button"
                      className="ss-secondary-btn ss-lib-open-btn"
                      onClick={() => onOpenSession(item.id)}
                    >
                      {status.key === "running"
                        ? "Xem tiến trình"
                        : status.key === "failed"
                          ? "Xem chi tiết"
                          : "Mở & chỉnh sửa"}
                    </button>

                    <div className="ss-dropdown-wrap">
                      <button
                        type="button"
                        className="ss-icon-menu-btn"
                        aria-haspopup="menu"
                        aria-expanded={isMenuOpen}
                        aria-label={`Tùy chọn khác cho ${item.title}`}
                        onClick={() => setOpenMenuId(cur => (cur === item.id ? null : item.id))}
                      >
                        ⋯
                      </button>
                      {isMenuOpen && (
                        <div className="ss-dropdown-menu" role="menu">
                          <button
                            type="button"
                            role="menuitem"
                            onClick={() => {
                              setOpenMenuId(null);
                              onOpenSession(item.id);
                            }}
                          >
                            Mở trình rà soát lời
                          </button>
                          <button
                            type="button"
                            role="menuitem"
                            onClick={() => startRename(item)}
                          >
                            Đổi tên phiên
                          </button>
                          {(item.state === "error" || item.state === "cancelled") && (
                            <button
                              type="button"
                              role="menuitem"
                              disabled={isBusy}
                              onClick={async () => {
                                setOpenMenuId(null);
                                setBusyActionId(item.id);
                                try {
                                  await onRetrySession(item);
                                } finally {
                                  setBusyActionId(null);
                                }
                              }}
                            >
                              Thử trích xuất lại
                            </button>
                          )}
                          <button
                            type="button"
                            role="menuitem"
                            className="danger"
                            disabled={isBusy || status.key === "running"}
                            onClick={async () => {
                              setOpenMenuId(null);
                              setBusyActionId(item.id);
                              try {
                                await onDeleteSession(item);
                              } finally {
                                setBusyActionId(null);
                              }
                            }}
                          >
                            Xóa Songscription
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

