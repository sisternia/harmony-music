import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useRouter, type Href } from "expo-router";
import { StrictMode, useState } from "react";
import "../../../styles/theme.css";
import { API_VERSION, api, configureApiBase } from "../api";
import { useT } from "../../../i18n";
import { NewSongDialog } from "../../../components/studio/NewSong";
import { Library } from "./Library";
import { Studio } from "./Studio";
import Navbar from "../../../components/layout/Navbar";
import "../../../styles/global.css";

function App({ path }: { path: string }) {
  const router = useRouter();
  const navigate = (to: string) => router.push(to as Href);
  const t = useT();
  const [newSong, setNewSong] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const config = useQuery({ queryKey: ["config"], queryFn: api.config, refetchInterval: 5000 });
  const match = path.match(/^\/p\/([^/]+)\/?$/);
  const pid = match ? decodeURIComponent(match[1]) : null;
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, enabled: Boolean(pid), refetchInterval: query => query.state.data?.some(project => project.busy) ? 2000 : false });

  return (
    <div className={`app ${pid ? `auralis-workspace cover-home project-app ${sidebarCollapsed ? "sidebar-collapsed" : ""}` : ""}`}>
      {pid && <Navbar path={path} navigate={navigate} projects={projects.data ?? []} collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(value => !value)} onAuth={() => navigate("/auth")} />}
      <div className="app-content">
      {config.data && config.data.api_version !== API_VERSION && <div className="stale-banner">{t.staleServer}</div>}
      <main className="content">
        {pid ? (
          config.isError ? (
            <div className="empty-state">
              {t.noServer}
            </div>
          ) : !config.data ? (
            <div className="empty-state">{t.loading}</div>
          ) : (
            <Studio key={pid} pid={pid} config={config.data!} navigate={navigate} />
          )
        ) : null}
        <div
          key="persistent-library-host"
          className={pid ? "ss-bg-dock" : "ss-visible-host"}
          aria-hidden={Boolean(pid) || undefined}
        >
          <Library
            path={path}
            navigate={navigate}
            youtubeSupported={config.data?.api_version === API_VERSION}
            isVisible={!pid}
          />
        </div>
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
