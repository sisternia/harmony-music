// API (cover_studio/app.py) の型と呼び出し

export type Params = {
  channel: number;
  temperature: number;
  top_p: number;
  source_cfg: number;
  channel_cfg: number;
  onset_bias: number;
  onset_bias_width: number;
  dynamics: number;
  density: number;
  fill: number;
  above: number;
  span: number;
  arrangement: boolean;
  seconds: number | null;
  // 好みのベクトル (Style) の id。あれば演奏者の番号 (channel) の代わりに使う
  style: string | null;
};

export type ModelEntry = {
  id: string;
  label: string;
  kind: "export" | "checkpoint" | "hub";
  spec: string;
  planner: string | null;
  num_channels: number | null;
};

// cover_studio/app.py の API_VERSION と同じにする (違えばサーバが古い)
export const API_VERSION = 5;

export type Config = {
  models: ModelEntry[];
  // 楽譜化のモデル (先頭が既定)
  score_models: ModelEntry[];
  loaded: string | null;
  device: string;
  tsumugi: boolean;
  defaults: Params;
  busy: boolean;
  api_version?: number;
};

export type TakeState = "queued" | "running" | "done" | "error" | "cancelled";

export type ContinueFrom = { take: string; seconds: number };

export type TakeScore = {
  state: "queued" | "running" | "done" | "error" | "cancelled";
  error: string | null;
  model: { id: string; label: string };
  measures?: number | null;
  created?: string | null;
  job?: string;
  progress?: number | null;
  message?: string;
};

export type Take = {
  id: string;
  number: number;
  name: string;
  created: string;
  state: TakeState;
  error: string | null;
  batch: number;
  batch_index: number;
  favorite: boolean;
  memo: string;
  duration: number | null;
  notes: number | null;
  params: Params;
  model: { id: string; label: string };
  seed: number;
  count: number;
  continue_from: ContinueFrom | null;
  score?: TakeScore | null;
  job?: string;
  progress?: number | null;
  message?: string;
  repaired_from?: string | null;
  repair_score?: number | null;
};

export type Transcribe = {
  state: "queued" | "running" | "done" | "error" | "cancelled";
  error: string | null;
  job?: string;
  message?: string;
  log?: string[];
};

export type ProjectLyricsSummary = {
  state: "idle" | "queued" | "running" | "done" | "error" | "cancelled";
  mode?: "extract" | "align";
  language?: string | null;
  language_name?: string | null;
  segment_count?: number;
  error?: string | null;
  job?: string;
  progress?: number | null;
  message?: string;
  updated?: string | null;
};

export type ProjectDetail = {
  id: string;
  title: string;
  created: string;
  audio: string | null;
  source: { origin: string; created: string } | null;
  transcribe: Transcribe | null;
  lyrics?: ProjectLyricsSummary | null;
  takes: Take[];
  busy: boolean;
};

export type ProjectSummary = {
  id: string;
  title: string;
  created: string;
  audio: string | null;
  has_source: boolean;
  transcribe: Transcribe | null;
  lyrics?: ProjectLyricsSummary | null;
  takes: number;
  favorites: number;
  busy: boolean;
};

// [onset 秒, 長さ 秒, pitch, velocity, 種類 (groups の番号)]
export type SourceNote = [number, number, number, number, number];
export type SourceView = {
  duration: number;
  groups: string[];
  notes: SourceNote[];
  beats: number[];
  downbeats: number[];
  chords: [number, string][];
};

// [onset 秒, 長さ 秒, pitch, velocity, 鳴り終わり 秒 (ペダルで延びた所まで)]
export type CoverNote = [number, number, number, number, number];
export type CoverView = { duration: number; notes: CoverNote[]; pedals: [number, number][] };

export type MidiRepairOptions = {
  remove_ghost_notes: boolean;
  min_duration_ms: number;
  min_velocity: number;
  fix_overlaps: boolean;
  duplicate_window_ms: number;
  rearticulate_gap_ms: number;
  clamp_piano_range: boolean;
  fix_hand_span: boolean;
  split_pitch: number;
  max_hand_span: number;
  max_polyphony: number;
  smart_quantize: boolean;
  quantize_strength: number;
  grid_subdivision: "8th" | "16th" | "triplet";
  normalize_velocity: boolean;
  vel_min: number;
  vel_max: number;
  melody_boost: number;
  clean_pedals: boolean;
  max_pedal_sec: number;
  repedal_on_chord: boolean;
};

export type MidiRepairIssueType =
  | "ghost_notes"
  | "overlapping_notes"
  | "out_of_range"
  | "hand_span"
  | "timing_jitter"
  | "velocity_spikes"
  | "pedal_glitches";

export type MidiRepairIssue = {
  id: string;
  type: MidiRepairIssueType;
  severity: "high" | "medium" | "low";
  time: number;
  pitch: number;
  note_idx: number | null;
  label: string;
};

export type MidiRepairScores = {
  overall: number;
  cleanliness: number;
  playability: number;
  timing: number;
  expression: number;
};

export type MidiRepairSnapshot = {
  duration: number;
  note_count: number;
  pedal_count: number;
  notes: CoverNote[];
  pedals: [number, number][];
  issue_counts: Record<MidiRepairIssueType, number>;
  total_issues: number;
  scores: MidiRepairScores;
  issues: MidiRepairIssue[];
};

export type MidiRepairDiffNote = {
  idx: number;
  onset: number;
  duration: number;
  pitch: number;
  velocity: number;
  orig_onset: number;
  orig_duration: number;
  orig_pitch: number;
  status: "kept" | "modified" | "removed" | "manual_deleted";
  reasons: string[];
};

export type MidiRepairPreviewResponse = {
  options: MidiRepairOptions;
  beats: number[];
  chords: [number, string][];
  has_backup: boolean;
  before: MidiRepairSnapshot;
  after: MidiRepairSnapshot;
  diff: {
    kept_count: number;
    modified_count: number;
    removed_count: number;
    actions: {
      ghosts_removed: number;
      overlaps_fixed: number;
      range_clamped: number;
      hand_spans_fixed: number;
      notes_quantized: number;
      velocities_balanced: number;
      pedals_cleaned: number;
      manual_edits: number;
    };
    notes: MidiRepairDiffNote[];
  };
};

export type MidiRepairManualUpdate = {
  idx: number;
  onset?: number;
  duration?: number;
  pitch?: number;
  velocity?: number;
};

export type MidiRepairApplyResponse = {
  project: ProjectDetail;
  created_take_id?: string;
  updated_take_id?: string;
  has_backup?: boolean;
  summary: {
    note_count: number;
    pedal_count: number;
    duration: number;
    actions: MidiRepairPreviewResponse["diff"]["actions"];
    issue_counts: Record<MidiRepairIssueType, number>;
    scores: MidiRepairScores;
  };
};

// お気に入りから学習した好みのベクトル (cover_studio/styles.py)。学習したモデルでだけ使える
export type Style = {
  id: string;
  name: string;
  created: string;
  state: "queued" | "running" | "done" | "error" | "cancelled";
  error: string | null;
  model: { id: string; label: string };
  takes: { project: string; take: string }[];
  init: number[];
  used?: number;
  loss_before?: number;
  loss_after?: number;
  distance?: number;
  trained?: string;
  job?: string;
  progress?: number | null;
  message?: string;
  log?: string[];
};

// 全曲のお気に入り (学習に使う候補)
export type FavoriteTake = {
  project: string;
  title: string;
  take: string;
  number: number;
  name: string;
  model: string;
  channel: number;
  style: string | null;
  audio: boolean;
};

export type StylesData = { styles: Style[]; favorites: FavoriteTake[] };

export type SongscriptionWord = {
  word: string;
  start: number;
  end: number;
  probability: number | null;
  uncertain?: boolean;
  timing_source?: string;
};

export type ReviewStatus = "needs_review" | "verified" | "edited" | "skipped";
export type VocalEventType = "lyric" | "vocable" | "humming" | "unknown_vocal";
export type VocalRole = "lead" | "backing" | "unknown";

export type SongscriptionCandidate = {
  candidate_id: string;
  text: string;
  event_type?: VocalEventType;
  language?: string | null;
  source_audio_variant?: string;
  config?: Record<string, unknown>;
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
  confidence?: number | null;
  coverage?: number | null;
  status?: "pending" | "accepted" | "rejected";
};

export type SongscriptionStreamInfo = {
  id: string;
  label: string;
  has_isolated_audio?: boolean;
  audio_url?: string | null;
};

export type MultiSingerSeparationStatus = {
  available: boolean;
  provider: string | null;
  status: string;
  note?: string;
};

export type SongscriptionSegment = {
  id: number;
  start: number;
  end: number;
  text: string;
  text_raw?: string;
  text_edited?: string | null;
  event_type?: VocalEventType;
  vocal_role?: VocalRole;
  is_adlib?: boolean | null;
  stream_id?: string;
  overlap_group_id?: string | null;
  language?: string | null;
  lang: string;
  section?: string | null;
  chord?: string | null;
  confidence?: number | null;
  low_confidence?: boolean;
  review_status?: ReviewStatus | null;
  review_reasons?: string[];
  review_reason?: string | null;
  timing_source?: string;
  timing_granularity?: "word" | "line" | "region";
  source_audio_variant?: string;
  candidates?: SongscriptionCandidate[];
  words?: SongscriptionWord[];
};

export type SongscriptionDetectedLanguage = {
  code: string;
  name: string;
  probability: number;
};

export type SongscriptionTranscript = {
  schema_version?: number;
  mode?: "extract" | "align";
  model?: string;
  config_hash?: string;
  language: string;
  language_name: string;
  language_confidence: number;
  detected_languages: SongscriptionDetectedLanguage[];
  is_multilingual: boolean;
  vocal_isolated: boolean;
  vocal_cached?: boolean;
  instrumental_cached?: boolean;
  alignment_method?: "word" | "line";
  duration: number;
  streams?: SongscriptionStreamInfo[];
  multi_singer_separation?: MultiSingerSeparationStatus;
  segments: SongscriptionSegment[];
  raw_asr_segments?: Record<string, unknown>[];
  diagnostics?: Record<string, unknown>;
  edit_history?: Record<string, unknown>[];
  full_text: string;
  structured_text?: string;
};

export type SongscriptionItem = {
  id: string;
  title: string;
  created: string;
  completed?: string | null;
  updated?: string | null;
  state: "queued" | "running" | "done" | "error" | "cancelled";
  mode?: "extract" | "align";
  error: string | null;
  source_kind: "project" | "youtube" | "upload";
  project_id: string | null;
  youtube_url?: string | null;
  youtube_id?: string | null;
  requested_language: string;
  model: string;
  isolate_vocals: boolean;
  language: string | null;
  language_name: string | null;
  language_confidence: number | null;
  detected_languages: SongscriptionDetectedLanguage[];
  vocal_isolated: boolean;
  vocal_cached?: boolean;
  instrumental_cached?: boolean;
  available_streams?: string[];
  alignment_method?: "word" | "line" | null;
  duration: number | null;
  segment_count: number;
  review_needs_count?: number;
  review_done_count?: number;
  job?: string;
  progress?: number | null;
  message?: string;
  log?: string[];
  transcript?: SongscriptionTranscript;
};

export type SongscriptionsResponse = {
  models: { id: string; label: string }[];
  vocal_isolation_available: boolean;
  items: SongscriptionItem[];
};

export type RetranscribeLineResult = {
  id: number;
  old_text: string;
  new_text: string;
  old_confidence: number;
  new_confidence: number;
  improved: boolean;
  changed: boolean;
  auto_resolved: boolean;
  variant: string;
};

export type RetranscribeFlaggedResponse = {
  item: SongscriptionItem;
  retranscribed_count: number;
  improved_count: number;
  auto_resolved_count: number;
  results: RetranscribeLineResult[];
};

export type ProjectLyricsResponse = {
  project_id: string;
  title: string;
  has_audio: boolean;
  state: "idle" | "queued" | "running" | "done" | "error" | "cancelled";
  mode: "extract" | "align";
  error: string | null;
  requested_language?: string;
  model?: string;
  isolate_vocals?: boolean;
  language: string | null;
  language_name: string | null;
  language_confidence: number | null;
  detected_languages: SongscriptionDetectedLanguage[];
  vocal_isolated: boolean;
  vocal_cached: boolean;
  alignment_method: "word" | "line" | null;
  segment_count: number;
  duration: number | null;
  updated: string | null;
  job?: string;
  progress?: number | null;
  message?: string;
  log?: string[];
  models: { id: string; label: string }[];
  vocal_isolation_available: boolean;
  transcript: SongscriptionTranscript | null;
};

export type LiveStem = { stem: string; duration: number; pos: number; done: boolean };
// 今の段階の進み具合 (ステム分離の "分離し終えた塊 / 全体の塊")
export type LiveProgress = { done: number; total: number };
export type LiveTranscriptionData = {
  seq: number;
  // [id, stem, start 秒, end 秒, pitch, final (0/1)]
  events: [string, string, number, number, number, number][];
  stems: LiveStem[];
  stage: string | null;
  progress: LiveProgress | null;
  active: boolean;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

let apiBase = "";

export function configureApiBase(base: string) {
  apiBase = base.replace(/\/$/, "");
}

const apiUrl = (path: string) => `${apiBase}${path}`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // 本文が JSON でなければ状態の文字列のまま
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const p = (pid: string) => `/api/projects/${encodeURIComponent(pid)}`;

export const api = {
  config: () => request<Config>("/api/config"),
  unload: () => request<Config>("/api/engine/unload", { method: "POST" }),
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (pid: string) => request<ProjectDetail>(p(pid)),
  createProject: (form: FormData) => request<ProjectDetail>("/api/projects", { method: "POST", body: form }),
  renameProject: (pid: string, title: string) => request<ProjectDetail>(p(pid), json("PATCH", { title })),
  deleteProject: (pid: string) => request<void>(p(pid), { method: "DELETE" }),
  transcribe: (pid: string) => request<ProjectDetail>(`${p(pid)}/transcribe`, { method: "POST" }),
  transcribeLive: (pid: string, since: number) =>
    request<LiveTranscriptionData>(`${p(pid)}/transcribe/live?since=${since}`),
  uploadSource: (pid: string, file: File) => {
    const form = new FormData();
    form.append("midi", file);
    return request<ProjectDetail>(`${p(pid)}/source`, { method: "PUT", body: form });
  },
  sourceView: (pid: string) => request<SourceView>(`${p(pid)}/source/view`),
  generate: (
    pid: string,
    body: { params: Params; model: string | null; count: number; seed: number | null; continue_from: ContinueFrom | null },
  ) => request<ProjectDetail>(`${p(pid)}/takes`, json("POST", body)),
  updateTake: (pid: string, tid: string, body: Partial<Pick<Take, "name" | "favorite" | "memo">>) =>
    request<Take>(`${p(pid)}/takes/${tid}`, json("PATCH", body)),
  deleteTake: (pid: string, tid: string) => request<void>(`${p(pid)}/takes/${tid}`, { method: "DELETE" }),
  takeView: (pid: string, tid: string) => request<CoverView>(`${p(pid)}/takes/${tid}/view`),
  previewMidiRepair: (
    pid: string,
    body: {
      target: "take" | "source";
      take_id?: string | null;
      options: Partial<MidiRepairOptions>;
      manual_deletes?: number[];
      manual_keeps?: number[];
      manual_updates?: MidiRepairManualUpdate[];
    },
  ) => request<MidiRepairPreviewResponse>(`${p(pid)}/midi-repair/preview`, json("POST", body)),
  applyMidiRepair: (
    pid: string,
    body: {
      target: "take" | "source";
      take_id?: string | null;
      save_mode: "new_take" | "overwrite_take" | "overwrite_source";
      name?: string;
      options: Partial<MidiRepairOptions>;
      manual_deletes?: number[];
      manual_keeps?: number[];
      manual_updates?: MidiRepairManualUpdate[];
    },
  ) => request<MidiRepairApplyResponse>(`${p(pid)}/midi-repair/apply`, json("POST", body)),
  revertMidiRepair: (pid: string, body: { target: "take" | "source"; take_id?: string | null }) =>
    request<{ project: ProjectDetail; reverted: boolean }>(`${p(pid)}/midi-repair/revert`, json("POST", body)),
  score: (pid: string, tid: string, model: string | null) =>
    request<ProjectDetail>(`${p(pid)}/takes/${tid}/score`, json("POST", { model })),
  scoreXml: async (pid: string, tid: string) => {
    const res = await fetch(urls.takeScore(pid, tid), { cache: "no-store" });
    if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
    return res.text();
  },
  styles: () => request<StylesData>("/api/styles"),
  createStyle: (body: { name: string; model: string | null }) => request<Style>("/api/styles", json("POST", body)),
  renameStyle: (sid: string, name: string) => request<Style>(`/api/styles/${sid}`, json("PATCH", { name })),
  deleteStyle: (sid: string) => request<void>(`/api/styles/${sid}`, { method: "DELETE" }),
  projectLyrics: (pid: string) => request<ProjectLyricsResponse>(`${p(pid)}/lyrics`),
  extractProjectLyrics: (
    pid: string,
    body: { language: string; model: string; isolate_vocals: boolean; force_separate?: boolean },
  ) => request<ProjectLyricsResponse>(`${p(pid)}/lyrics/extract`, json("POST", body)),
  alignProjectLyrics: (
    pid: string,
    body: { lyrics_text: string; language: string; model: string; isolate_vocals: boolean },
  ) => request<ProjectLyricsResponse>(`${p(pid)}/lyrics/align`, json("POST", body)),
  saveProjectLyrics: (pid: string, segments: SongscriptionSegment[]) =>
    request<ProjectLyricsResponse>(`${p(pid)}/lyrics`, json("PUT", { segments })),
  deleteProjectLyrics: (pid: string) => request<void>(`${p(pid)}/lyrics`, { method: "DELETE" }),
  songscriptions: () => request<SongscriptionsResponse>("/api/songscriptions"),
  songscription: (sid: string) => request<SongscriptionItem>(`/api/songscriptions/${encodeURIComponent(sid)}`),
  createSongscription: (form: FormData) =>
    request<SongscriptionItem>("/api/songscriptions", { method: "POST", body: form }),
  retrySongscription: (sid: string) =>
    request<SongscriptionItem>(`/api/songscriptions/${encodeURIComponent(sid)}/retry`, { method: "POST" }),
  alignSongscription: (
    sid: string,
    body: { lyrics_text: string; language: string; model: string; isolate_vocals: boolean },
  ) => request<SongscriptionItem>(`/api/songscriptions/${encodeURIComponent(sid)}/align`, json("POST", body)),
  updateSongscription: (sid: string, body: { title?: string; segments?: SongscriptionSegment[]; youtube_url?: string }) =>
    request<SongscriptionItem>(`/api/songscriptions/${encodeURIComponent(sid)}`, json("PATCH", body)),
  retranscribeSongscriptionFlagged: (
    sid: string,
    body?: { segment_ids?: number[]; model?: string; language?: string },
  ) =>
    request<RetranscribeFlaggedResponse>(
      `/api/songscriptions/${encodeURIComponent(sid)}/retranscribe-flagged`,
      json("POST", body ?? {}),
    ),
  deleteSongscription: (sid: string) =>
    request<void>(`/api/songscriptions/${encodeURIComponent(sid)}`, { method: "DELETE" }),
  cancelJob: (jobId: string) => request<unknown>(`/api/jobs/${jobId}`, { method: "DELETE" }),
};

export const urls = {
  audio: (pid: string) => apiUrl(`${p(pid)}/audio`),
  sourceMidi: (pid: string) => apiUrl(`${p(pid)}/source.mid`),
  takeMidi: (pid: string, tid: string) => apiUrl(`${p(pid)}/takes/${tid}/cover.mid`),
  takeScore: (pid: string, tid: string) => apiUrl(`${p(pid)}/takes/${tid}/score.musicxml`),
  songscriptionAudio: (sid: string) => apiUrl(`/api/songscriptions/${encodeURIComponent(sid)}/audio`),
  songscriptionVocalAudio: (sid: string) => apiUrl(`/api/songscriptions/${encodeURIComponent(sid)}/vocal-audio`),
  songscriptionInstrumentalAudio: (sid: string) =>
    apiUrl(`/api/songscriptions/${encodeURIComponent(sid)}/instrumental-audio?v=bsr`),
  songscriptionStreamAudio: (sid: string, streamId: string) =>
    apiUrl(`/api/songscriptions/${encodeURIComponent(sid)}/streams/${encodeURIComponent(streamId)}/audio`),
};
