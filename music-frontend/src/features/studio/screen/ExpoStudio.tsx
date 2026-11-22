import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter, type Href } from "expo-router";
import { StrictMode, useState } from "react";
import "../../../styles/theme.css";
import { API_VERSION, api, configureApiBase } from "../api";
import { useLang, useT } from "../../../i18n";
import { Icon } from "../../../components/ui/StudioIcon";
import { NewSongDialog } from "../../../components/studio/NewSong";
import { SongSwitcher } from "../../../components/studio/SongSwitcher";
import { shortModel } from "../../../components/studio/Takes";
import { Library } from "./Library";
import { Studio } from "./Studio";
import Navbar from "../../../components/layout/Navbar";
import "../../../styles/global.css";

function App({ path }: { path: string }) {
  const router = useRouter();
  const navigate = (to: string) => router.push(to as Href);
  const qc = useQueryClient();
  const t = useT();
  const [lang, setLang] = useLang();
  const [newSong, setNewSong] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const config = useQuery({ queryKey: ["config"], queryFn: api.config, refetchInterval: 5000 });
  const match = path.match(/^\/p\/([^/]+)\/?$/);
  const pid = match ? decodeURIComponent(match[1]) : null;
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, enabled: Boolean(pid), refetchInterval: query => query.state.data?.some(project => project.busy) ? 2000 : false });
  const loaded = config.data?.models.find((m) => m.id === config.data?.loaded);

  return (
    <div className={`app ${pid ? `auralis-workspace cover-home project-app ${sidebarCollapsed ? "sidebar-collapsed" : ""}` : ""}`}>
      {pid && <Navbar path={path} navigate={navigate} projects={projects.data ?? []} collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(value => !value)} onAuth={() => navigate("/auth")} />}
      <div className="app-content">
      {pid && <header className="topbar">
        {pid && (
          <>
            <SongSwitcher
              pid={pid}
              onSelect={(id) => navigate(`/p/${encodeURIComponent(id)}`)}
              onNew={() => setNewSong(true)}
              onAll={() => navigate("/")}
            />
            <button className="btn small new-song-btn" onClick={() => setNewSong(true)}>
              <Icon name="plus" size={14} /> {t.newSong}
            </button>
          </>
        )}
        <div className="grow" />
        {config.data && (
          <div className="engine" title={`${t.device}: ${config.data.device}`}>
            <Icon name="chip" size={15} />
            <span>{config.data.device}</span>
            <span className="muted">·</span>
            <span className={loaded ? "" : "muted"}>{loaded ? shortModel(loaded.label) : t.noModelLoaded}</span>
            {loaded && (
              <button
                className="link"
                title={t.freeVramHelp}
                onClick={async () => {
                  try {
                    await api.unload();
                  } finally {
                    void qc.invalidateQueries({ queryKey: ["config"] });
                  }
                }}
              >
                {t.freeVram}
              </button>
            )}
          </div>
        )}
        <button
          className="lang-btn"
          onClick={() => setLang(lang === "ja" ? "en" : "ja")}
          title={t.languageHelp}
          aria-label={t.languageHelp}
        >
          {t.language}
        </button>
      </header>}
      {config.data && config.data.api_version !== API_VERSION && <div className="stale-banner">{t.staleServer}</div>}
      <main className="content">
        {pid && config.isError ? (
          <div className="empty-state">
            {t.noServer}
          </div>
        ) : pid && !config.data ? (
          <div className="empty-state">{t.loading}</div>
        ) : pid ? (
          <Studio key={pid} pid={pid} config={config.data!} navigate={navigate} />
        ) : (
          <Library path={path} navigate={navigate} youtubeSupported={config.data?.api_version === API_VERSION} />
        )}
      </main>
      {newSong && config.data && (
        <NewSongDialog
          config={config.data}
          onClose={() => setNewSong(false)}
          onCreated={(id) => {
            setNewSong(false);
            navigate(`/p/${encodeURIComponent(id)}`);
          }}
        />
      )}
      </div>
    </div>
  );
}

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

export default function ExpoStudio({ apiBase = "", path }: { apiBase?: string; path: string }) {
  configureApiBase(apiBase);
  return (
    <StrictMode>
      <QueryClientProvider client={client}>
        <App path={path} />
      </QueryClientProvider>
    </StrictMode>
  );
}
