// 1 曲の作業画面: 左で設定して生成し、真ん中でテイクを聴き比べ、右で選んだテイクを見る
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, urls, type Config, type ContinueFrom, type ProjectDetail, type Take } from "../api";
import { CreatePanel, takeLabel, type Draft } from "../../../components/studio/CreatePanel";
import { Icon } from "../../../components/ui/StudioIcon";
import { LiveTranscription } from "../../../components/piano/LiveTranscription";
import { LyricsPanel } from "../../../components/lyrics/LyricsPanel";
import { PianoRoll } from "../../../components/piano/PianoRoll";
import { PlayerBar, usePlayerState } from "../../../components/piano/PlayerBar";
import { ScoreViewer } from "../../../components/piano/ScoreViewer";
import { TakeCard, TakeDetail } from "../../../components/studio/Takes";
import { SmartMidiRepair } from "../components/SmartMidiRepair";
import { progressText, useT } from "../../../i18n";
import { formatTime } from "../utils/params";
import { downloadCoverWav, getPlayer, type Player } from "../utils/player";
import "../styles/studio-custom.css";

// 「作る」欄の設定は曲をまたいで同じものを使う (別の曲へ移っても、変えた設定のまま続けられる)
const DRAFT_KEY = "cover-studio:draft";
// 楽譜化のモデルの選択も曲をまたいで覚えておく
const SCORE_MODEL_KEY = "cover-studio:score-model";

function loadScoreModel(config: Config): string | null {
  try {
    const id = localStorage.getItem(SCORE_MODEL_KEY);
    return id && (config.score_models ?? []).some((m) => m.id === id) ? id : null;
  } catch {
    return null;
  }
}

// 携帯では 3 つの欄をタブで切り替える。
// 幅のしきい値は styles.css の @media (max-width: 820px) と揃えること
type Tab = "create" | "play" | "detail";
const PHONE_MAX_WIDTH = 820;
const isPhone = () => window.matchMedia(`(max-width: ${PHONE_MAX_WIDTH}px)`).matches;

function loadDraft(pid: string, config: Config): Draft {
  const fallback: Draft = { params: { ...config.defaults }, model: null, count: 2, seedLocked: false, seed: null };
  try {
    // 以前は曲ごとに覚えていたので、共通のものがまだなければその曲のものを使う
    const raw = localStorage.getItem(DRAFT_KEY) ?? localStorage.getItem(`cover-studio:draft:${pid}`);
    const saved = JSON.parse(raw ?? "null");
    if (saved) {
      const draft: Draft = { ...fallback, ...saved, params: { ...config.defaults, ...saved.params } };
      // 覚えていたモデルがもうなければ (消した・別の環境) 既定のモデルにする
      if (draft.model && !config.models.some((m) => m.id === draft.model)) draft.model = null;
      return draft;
    }
  } catch {
    // 保存できない環境 (プライベートモードなど) では毎回既定値から
  }
  return fallback;
}

function saveDraft(draft: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // 覚えておけなくても使える
  }
}

export function Studio({ pid, config, navigate }: { pid: string; config: Config; navigate: (to: string) => void }) {
  const qc = useQueryClient();
  const t = useT();
  const player = usePlayerState(getPlayer());
  const project = useQuery({
    queryKey: ["project", pid],
    queryFn: () => api.project(pid),
    refetchInterval: (q) => (q.state.data?.busy ? 1000 : false),
  });
  const data = project.data;
  useEffect(() => {
    if (data?.title) document.title = `${data.title} - Harmony Auralis`;
  }, [data?.title]);
  const sourceView = useQuery({
    queryKey: ["source-view", pid, data?.source?.created],
    queryFn: () => api.sourceView(pid),
    enabled: !!data?.source,
    staleTime: Infinity,
  });

  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "favorite">("all");
  const [showSource, setShowSource] = useState(false);
  const [centerView, setCenterView] = useState<"piano" | "repair" | "lyrics">("piano");
  const [createOpen, setCreateOpen] = useState<boolean | null>(null);
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const moreMenuRef = useRef<HTMLDivElement>(null);
  const midiUploadRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<Tab>("play");
  // 携帯で巻物をたたむかどうか (広い画面では styles.css がボタンごと隠す)
  const [rollOpen, setRollOpen] = useState(true);
  const [continueFrom, setContinueFrom] = useState<ContinueFrom | null>(null);
  const [draft, setDraftState] = useState<Draft>(() => loadDraft(pid, config));
  const setDraft = (d: Draft) => {
    setDraftState(d);
    saveDraft(d);
  };
  const [error, setError] = useState<string | null>(null);
  const [scoreModel, setScoreModelState] = useState<string | null>(() => loadScoreModel(config));
  const setScoreModel = (id: string) => {
    setScoreModelState(id);
    try {
      localStorage.setItem(SCORE_MODEL_KEY, id);
    } catch {
      // 覚えておけなくても使える
    }
  };
  // 楽譜を開いているテイク
  const [scoreOpen, setScoreOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!moreMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (moreMenuRef.current && !moreMenuRef.current.contains(e.target as Node)) setMoreMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [moreMenuOpen]);

  const takes = useMemo(() => data?.takes ?? [], [data?.takes]);
  const doneTakes = useMemo(() => takes.filter((t) => t.state === "done"), [takes]);
  const showCreatePanel = createOpen ?? takes.length === 0;
  const shown = filter === "favorite" ? takes.filter((t) => t.favorite) : takes;
  const effectiveSelected =
    selected && takes.some((t) => t.id === selected) ? selected : (doneTakes[0]?.id ?? null);
  const selectedTake = takes.find((t) => t.id === effectiveSelected) ?? null;
  const selectedState = selectedTake?.state;

  const takeView = useQuery({
    queryKey: ["take-view", pid, effectiveSelected],
    queryFn: () => api.takeView(pid, effectiveSelected!),
    enabled: selectedState === "done",
    staleTime: Infinity,
  });

  useEffect(() => {
    if (centerView === "repair") return;
    if (takeView.data) player.setNotes(takeView.data.notes);
    else if (selectedState && selectedState !== "done") player.setNotes([]);
  }, [centerView, takeView.data, selectedState, player]);

  useEffect(() => {
    void player.setOriginal(data?.audio ? urls.audio(pid) : null);
  }, [pid, data?.audio, player]);

  useEffect(() => {
    player.setExtraDuration(sourceView.data?.duration ?? 0);
  }, [sourceView.data, player]);

  // 曲を離れたら止めて片付ける
  useEffect(
    () => () => {
      player.pause();
      player.setNotes([]);
      player.setLoop(null);
      player.seek(0);
      void player.setOriginal(null);
    },
    [pid, player],
  );

  const refresh = (detail?: ProjectDetail) => {
    if (detail) qc.setQueryData(["project", pid], detail);
    else void qc.invalidateQueries({ queryKey: ["project", pid] });
  };
  const run = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const generate = useMutation({
    mutationFn: () =>
      api.generate(pid, {
        params: draft.params,
        model: draft.model,
        count: draft.count,
        seed: draft.seedLocked ? draft.seed : null,
        continue_from: continueFrom,
      }),
    onSuccess: (detail) => {
      refresh(detail);
      setContinueFrom(null);
      setError(null);
      // 携帯では、作り始めたら演奏の面へ移して進み具合を見せる
      if (isPhone()) setTab("play");
      void qc.invalidateQueries({ queryKey: ["config"] });
    },
    onError: (e) => setError(e instanceof Error ? e.message : String(e)),
  });

  const updateTake = (take: Take, body: Partial<Pick<Take, "name" | "favorite" | "memo">>) =>
    run(async () => {
      await api.updateTake(pid, take.id, body);
      refresh();
      // お気に入りの数は好みの学習の欄に出る
      if (body.favorite !== undefined) void qc.invalidateQueries({ queryKey: ["styles"] });
    });
  const deleteTake = (take: Take) =>
    run(async () => {
      if (!confirm(t.confirmDeleteTake(takeLabel(take, t)))) return;
      await api.deleteTake(pid, take.id);
      if (selected === take.id) setSelected(null);
      refresh();
      if (take.favorite) void qc.invalidateQueries({ queryKey: ["styles"] });
    });
  const reuse = (take: Take) => {
    setDraft({ ...draft, params: { ...config.defaults, ...take.params }, model: take.model.id, seed: take.seed });
    setCreateOpen(true);
    if (isPhone()) setTab("create");
  };
  const playTake = (take: Take) => {
    if (effectiveSelected === take.id) player.toggle();
    else {
      setSelected(take.id);
      if (!player.playing) player.play();
    }
  };

  // キー操作: Space 再生、← → 5 秒、↑ ↓ テイクを切り替え (位置はそのまま)、L ループ、F お気に入り
  const keys = useRef({ doneTakes, selectedTake });
  useEffect(() => {
    keys.current = { doneTakes, selectedTake };
  }, [doneTakes, selectedTake]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && (target as HTMLInputElement).type !== "range")
        return;
      const { doneTakes, selectedTake } = keys.current;
      if (e.code === "Space") {
        e.preventDefault();
        player.toggle();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        player.seek(player.position() - 5);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        player.seek(player.position() + 5);
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const i = doneTakes.findIndex((t) => t.id === selectedTake?.id);
        const next = doneTakes[Math.max(0, Math.min(doneTakes.length - 1, i + (e.key === "ArrowUp" ? -1 : 1)))];
        if (next) setSelected(next.id);
      } else if (e.key === "l" || e.key === "L") {
        player.toggleLoop();
      } else if ((e.key === "f" || e.key === "F") && selectedTake) {
        void updateTake(selectedTake, { favorite: !selectedTake.favorite });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player, pid]);

  const continueTake = continueFrom ? (takes.find((t) => t.id === continueFrom.take) ?? null) : null;
  const disabledReason = !data?.source
    ? t.waitForSource
    : !config.models.length
      ? t.noModels
      : null;
  const takeById = useMemo(() => new Map(takes.map((t) => [t.id, t])), [takes]);

  if (project.isError) {
    return (
      <div className="empty-state">
        <p>{(project.error as Error).message}</p>
        <button className="btn" onClick={() => navigate("/")}>
          {t.songs}
        </button>
      </div>
    );
  }
  if (!data) return <div className="empty-state">{t.loading}</div>;

  return (
    <>
      {/* 携帯だけに出る切り替え (広い画面では styles.css が .tabs を隠す) */}
      <nav className="tabs" role="tablist" aria-label={t.viewSwitch}>
        {(
          [
            ["create", t.tabCreate],
            ["play", t.tabPlay],
            ["detail", t.tabDetail],
          ] as [Tab, string][]
        ).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={tab === id ? "on" : ""}
            onClick={() => setTab(id)}
          >
            {label}
            {id === "play" && takes.length > 0 && <span className="count">{takes.length}</span>}
          </button>
        ))}
      </nav>
      <div
        className={`studio studio-custom ${centerView !== "piano" ? "is-expanded" : ""} ${
          !showCreatePanel ? "hide-left" : ""
        }`}
        data-tab={tab}
      >
        <CreatePanel
          config={config}
          draft={draft}
          onChange={setDraft}
          continueFrom={continueFrom}
          continueTake={continueTake}
          onClearContinue={() => setContinueFrom(null)}
          onGenerate={() => generate.mutate()}
          onClose={takes.length > 0 ? () => setCreateOpen(false) : undefined}
          generating={generate.isPending}
          disabled={disabledReason}
        />

        {/* has-roll: 巻物があるときだけ、巻物を固定して一覧だけをスクロールさせる */}
        <section
          className={`center ${
            centerView === "repair"
              ? "is-repair-mode"
              : data.source || centerView === "lyrics"
                ? "has-roll"
                : ""
          }`}
        >
          <div className="song-head studio-header-card">
            <div className="studio-header-left">
              <button
                type="button"
                className="studio-back-btn"
                onClick={() => navigate("/projects")}
                title="Quay lại danh sách Projects"
              >
                ← Projects
              </button>
              <h1>{data.title}</h1>
            </div>
            <div className="song-actions">
              <div className="studio-mode-tabs" role="tablist" aria-label="Chế độ làm việc Studio">
                <button
                  type="button"
                  role="tab"
                  aria-selected={centerView === "piano"}
                  className={centerView === "piano" ? "on" : ""}
                  onClick={() => setCenterView("piano")}
                >
                  🎹 Piano &amp; Takes
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={centerView === "repair"}
                  className={centerView === "repair" ? "on" : ""}
                  disabled={!data.source && doneTakes.length === 0}
                  onClick={() => setCenterView("repair")}
                >
                  ✦ Smart MIDI Repair
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={centerView === "lyrics"}
                  className={centerView === "lyrics" ? "on" : ""}
                  onClick={() => setCenterView("lyrics")}
                >
                  ♪ AI Lyrics
                  {data.lyrics?.segment_count ? (
                    <span className="studio-tab-badge">{data.lyrics.segment_count}</span>
                  ) : null}
                </button>
              </div>

              {centerView === "piano" && (
                <>
                  <button
                    type="button"
                    className={`btn small ${showCreatePanel ? "" : "primary"}`}
                    onClick={() => setCreateOpen(!showCreatePanel)}
                  >
                    <Icon name="plus" size={14} /> {showCreatePanel ? "Đóng tạo phối" : "Tạo phối mới"}
                  </button>

                  <div className="studio-menu-wrap" ref={moreMenuRef}>
                    <button
                      type="button"
                      className="studio-menu-btn"
                      aria-label="Tác vụ MIDI gốc"
                      aria-expanded={moreMenuOpen}
                      onClick={() => setMoreMenuOpen((v) => !v)}
                    >
                      ⋯
                    </button>
                    {moreMenuOpen && (
                      <div className="studio-menu-dropdown" role="menu">
                        {data.source && (
                          <a
                            role="menuitem"
                            href={urls.sourceMidi(pid)}
                            download
                            onClick={() => setMoreMenuOpen(false)}
                          >
                            <Icon name="download" size={15} /> Tải MIDI gốc
                          </a>
                        )}
                        <button
                          type="button"
                          role="menuitem"
                          onClick={() => {
                            setMoreMenuOpen(false);
                            midiUploadRef.current?.click();
                          }}
                        >
                          <Icon name="layers" size={15} /> Thay tệp MIDI gốc…
                        </button>
                      </div>
                    )}
                    <input
                      ref={midiUploadRef}
                      type="file"
                      accept=".mid,.midi"
                      hidden
                      onChange={async (e) => {
                        const file = e.target.files?.[0];
                        e.target.value = "";
                        if (!file) return;
                        try {
                          refresh(await api.uploadSource(pid, file));
                        } catch (err) {
                          setError(err instanceof Error ? err.message : String(err));
                        }
                      }}
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {error && (
            <div className="alert" onClick={() => setError(null)}>
              {error}
            </div>
          )}

          {centerView === "repair" ? (
            <SmartMidiRepair
              pid={pid}
              project={data}
              selectedTake={selectedTake}
              player={player}
              onSelectTake={tid => setSelected(tid)}
              onProjectUpdated={detail => {
                refresh(detail);
                void qc.invalidateQueries({ queryKey: ["take-view", pid] });
                void qc.invalidateQueries({ queryKey: ["source-view", pid] });
              }}
              onSwitchToPiano={() => setCenterView("piano")}
            />
          ) : centerView === "lyrics" ? (
            <LyricsPanel pid={pid} project={data} player={player} />
          ) : (
            <>
              {!data.source ? (
                <TranscribePanel project={data} pid={pid} player={player} onChange={refresh} onError={setError} />
              ) : (
                <div className={`roll-card ${rollOpen ? "" : "collapsed"}`}>
                  <div className="roll-head">
                    <span className="roll-title">
                      <span className="roll-title-text">
                        {selectedTake ? takeLabel(selectedTake, t) : t.sourceMidiTitle}
                      </span>
                      <span className="spinner small" style={{ visibility: takeView.isFetching ? "visible" : "hidden" }} />
                    </span>
                    {/* 携帯だけに出る巻物の開閉 (広い画面では styles.css が隠す) */}
                    <button
                      className={`roll-toggle ${rollOpen ? "open" : ""}`}
                      onClick={() => setRollOpen(!rollOpen)}
                      aria-expanded={rollOpen}
                      title={rollOpen ? t.rollCollapse : t.rollExpand}
                    >
                      <Icon name="chevron" size={14} />
                      {rollOpen ? t.rollCollapse : t.rollExpand}
                    </button>
                    <label className="check">
                      <input type="checkbox" checked={showSource} onChange={(e) => setShowSource(e.target.checked)} />
                      Hiện MIDI gốc
                    </label>
                  </div>
                  <PianoRoll
                    player={player}
                    cover={selectedTake?.state === "done" ? (takeView.data ?? null) : null}
                    source={sourceView.data ?? null}
                    showSource={showSource}
                    continueAt={continueFrom?.seconds ?? null}
                  />
                  {player.loop && (
                    <div className="roll-hint muted small">
                      <button className="link" onClick={() => player.setLoop(null)}>
                        Bỏ vòng lặp ({formatTime(player.loop[0])}–{formatTime(player.loop[1])})
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* 巻物 (ピアノロール) は固定したまま、この下のテイク一覧だけを独立スクロールさせる */}
              <div className="takes-scroll">
                <div className="takes-head">
                  <h2>{t.takes}</h2>
                  <div className="segmented small">
                    <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>
                      {t.all} {takes.length}
                    </button>
                    <button className={filter === "favorite" ? "on" : ""} onClick={() => setFilter("favorite")}>
                      ★ {takes.filter((x) => x.favorite).length}
                    </button>
                  </div>
                </div>
                <div className="takes">
                  {shown.length === 0 && (
                    <div className="empty-takes muted">
                      {data.source
                        ? filter === "favorite"
                          ? t.noFavorites
                          : t.emptyTakes
                        : t.waitTranscribe}
                    </div>
                  )}
                  {shown.map((take) => {
                    const cf = take.continue_from;
                    const base = cf ? takeById.get(cf.take) : undefined;
                    return (
                      <TakeCard
                        key={take.id}
                        take={take}
                        defaults={config.defaults}
                        selected={take.id === effectiveSelected}
                        playing={take.id === effectiveSelected && player.playing}
                        continueLabel={cf ? t.continueChip(base ? takeLabel(base, t) : cf.take, formatTime(cf.seconds)) : null}
                        onSelect={() => {
                          setSelected(take.id);
                          // 携帯では選んだら詳細へ (広い画面ではタブ自体が出ないので何も変わらない)
                          if (isPhone()) setTab("detail");
                        }}
                        onPlay={() => playTake(take)}
                        onFavorite={() => updateTake(take, { favorite: !take.favorite })}
                        onReuse={() => reuse(take)}
                        onCancel={() => take.job && run(async () => (await api.cancelJob(take.job!), refresh()))}
                        onDelete={() => deleteTake(take)}
                      />
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </section>

        <aside className="right">
          {selectedTake ? (
            <TakeDetail
              take={selectedTake}
              config={config}
              position={() => (player.loopOn && player.loop ? player.loop[0] : player.position())}
              midiUrl={urls.takeMidi(pid, selectedTake.id)}
              onRename={(name) => updateTake(selectedTake, { name })}
              onMemo={(memo) => updateTake(selectedTake, { memo })}
              onFavorite={() => updateTake(selectedTake, { favorite: !selectedTake.favorite })}
              onReuse={() => reuse(selectedTake)}
              onContinue={(seconds) => {
                setContinueFrom({ take: selectedTake.id, seconds: Math.round(seconds * 100) / 100 });
                reuse(selectedTake);
              }}
              onWav={() =>
                run(async () => {
                  const view = takeView.data ?? (await api.takeView(pid, selectedTake.id));
                  await downloadCoverWav(view.notes, `${data.title}_${takeLabel(selectedTake, t)}.wav`);
                })
              }
              onDelete={() => deleteTake(selectedTake)}
              scoreModel={scoreModel}
              scoreBlocked={data.audio ? null : t.scoreNeedsAudio}
              scoreUrl={urls.takeScore(pid, selectedTake.id)}
              onScoreModel={setScoreModel}
              onScore={() => run(async () => refresh(await api.score(pid, selectedTake.id, scoreModel)))}
              onCancelScore={() =>
                selectedTake.score?.job && run(async () => (await api.cancelJob(selectedTake.score!.job!), refresh()))
              }
              onOpenScore={() => setScoreOpen(selectedTake.id)}
              onRepair={() => setCenterView("repair")}
            />
          ) : (
            <div className="detail empty muted">{t.selectTakeHint}</div>
          )}
        </aside>
      </div>
      {scoreOpen && takeById.get(scoreOpen)?.score?.state === "done" && (
        <ScoreViewer
          pid={pid}
          tid={scoreOpen}
          version={takeById.get(scoreOpen)!.score!.created}
          title={`${data.title} — ${takeLabel(takeById.get(scoreOpen)!, t)}`}
          onClose={() => setScoreOpen(null)}
        />
      )}
      <PlayerBar
        player={player}
        title={selectedTake ? `${takeLabel(selectedTake, t)} — ${data.title}` : data.title}
        subtitle={
          player.originalLoading
            ? t.loadingAudio
            : selectedTake
              ? `seed ${selectedTake.seed}`
              : data.audio
                ? t.original
                : t.chooseTake
        }
      />
    </>
  );
}

function TranscribePanel({
  project,
  pid,
  player,
  onChange,
  onError,
}: {
  project: ProjectDetail;
  pid: string;
  player: Player;
  onChange: (d?: ProjectDetail) => void;
  onError: (m: string) => void;
}) {
  const t = useT();
  const tr = project.transcribe;
  const active = tr?.state === "queued" || tr?.state === "running";
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [tr?.log?.length]);
  return (
    <div className="transcribe">
      <div className="transcribe-head">
        {active ? <span className="spinner" /> : <Icon name="music" size={22} />}
        <div>
          <b>
            {active
              ? tr?.state === "queued"
                ? t.transcribeQueued
                : t.transcribeRunning
              : tr?.state === "error"
                ? t.transcribeFailed
                : tr?.state === "cancelled"
                  ? t.transcribeCancelled
                  : t.transcribeNone}
          </b>
          <div className="muted small">
            {active
              ? t.transcribeSteps
              : (tr?.error ?? t.transcribeHint)}
          </div>
        </div>
        <div className="grow" />
        {active && tr?.job ? (
          <button className="btn" onClick={async () => (await api.cancelJob(tr.job!), onChange())}>
            <Icon name="stop" size={14} /> {t.stop}
          </button>
        ) : (
          project.audio && (
            <button
              className="btn primary"
              onClick={async () => {
                try {
                  onChange(await api.transcribe(pid));
                } catch (e) {
                  onError(e instanceof Error ? e.message : String(e));
                }
              }}
            >
              {t.transcribe}
            </button>
          )
        )}
      </div>
      {tr?.message && active && !tr.message.startsWith("@@") && <div className="transcribe-msg mono small">{progressText(t, tr.message)}</div>}
      {tr?.job && <LiveTranscription pid={pid} player={player} active={active} />}
      {tr?.log && tr.log.length > 0 && (
        <details className="log-box" open={tr.state === "error"}>
          <summary className="muted small">{t.showLog}</summary>
          <pre className="log" ref={logRef}>
            {tr.log.filter((line) => !line.startsWith("@@")).join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}
