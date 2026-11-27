import type { ReviewStatus, SongscriptionSegment, VocalEventType, VocalRole } from "../api";

export type ExportPolicy = "main_only" | "with_adlib" | "merged_labeled" | "per_stream";

export const EVENT_TYPE_OPTIONS: { value: VocalEventType; label: string }[] = [
  { value: "lyric", label: "Lời hát" },
  { value: "vocable", label: "Vocalise / Âm đệm" },
  { value: "humming", label: "Humming / Ngân" },
  { value: "unknown_vocal", label: "Nghi có vocal" },
];

export const VOCAL_ROLE_OPTIONS: { value: VocalRole; label: string }[] = [
  { value: "unknown", label: "Chưa rõ vai trò" },
  { value: "lead", label: "Giọng chính" },
  { value: "backing", label: "Hát bè / Đệm" },
];

export const STREAM_OPTIONS: { value: string; label: string }[] = [
  { value: "stream_a", label: "Luồng A" },
  { value: "stream_b", label: "Luồng B" },
  { value: "stream_c", label: "Luồng C" },
];

export function streamLabel(streamId?: string | null): string {
  const sid = (streamId || "stream_a").trim().toLowerCase();
  if (sid === "stream_a") return "Luồng A";
  if (sid === "stream_b") return "Luồng B";
  if (sid === "stream_c") return "Luồng C";
  const m = sid.match(/^stream_([a-z0-9]+)$/);
  return m ? `Luồng ${m[1].toUpperCase()}` : sid;
}

export function getEffectiveEventType(seg: SongscriptionSegment): VocalEventType {
  if (
    seg.event_type === "lyric" ||
    seg.event_type === "vocable" ||
    seg.event_type === "humming" ||
    seg.event_type === "unknown_vocal"
  ) {
    return seg.event_type;
  }
  return seg.text.trim() ? "lyric" : "unknown_vocal";
}

export function isAdlibOrVocalise(seg: SongscriptionSegment): boolean {
  const ev = getEffectiveEventType(seg);
  return ev === "vocable" || ev === "humming" || ev === "unknown_vocal" || seg.is_adlib === true;
}

export function isMainLyric(seg: SongscriptionSegment): boolean {
  return getEffectiveEventType(seg) === "lyric" && seg.is_adlib !== true;
}

export function isOverlapSegment(seg: SongscriptionSegment, allSegments: SongscriptionSegment[]): boolean {
  if (seg.overlap_group_id) return true;
  if ((seg.stream_id || "stream_a") !== "stream_a") return true;
  return allSegments.some(
    other =>
      other.id !== seg.id && Math.min(seg.end, other.end) - Math.max(seg.start, other.start) >= 0.2,
  );
}

export const LANGUAGE_OPTIONS: [string, string][] = [
  ["auto", "✦ Tự động nhận diện (Đa ngôn ngữ)"],
  ["ja", "Tiếng Nhật (Japanese · 日本語)"],
  ["en", "Tiếng Anh (English)"],
  ["vi", "Tiếng Việt (Vietnamese)"],
  ["ko", "Tiếng Hàn (Korean · 한국어)"],
  ["zh", "Tiếng Trung (Chinese · 中文)"],
  ["yue", "Tiếng Quảng Đông (Cantonese)"],
  ["fr", "Tiếng Pháp (French)"],
  ["es", "Tiếng Tây Ban Nha (Spanish)"],
  ["de", "Tiếng Đức (German)"],
  ["it", "Tiếng Ý (Italian)"],
  ["pt", "Tiếng Bồ Đào Nha (Portuguese)"],
  ["ru", "Tiếng Nga (Russian)"],
  ["th", "Tiếng Thái (Thai)"],
  ["id", "Tiếng Indonesia (Indonesian)"],
];

export const SECTION_OPTIONS: string[] = [
  "",
  "Intro",
  "Verse 1",
  "Verse 2",
  "Pre-Chorus",
  "Chorus",
  "Bridge",
  "Outro",
  "Instrumental",
];

export const PIPELINE_STAGES: { key: string; label: string; short: string }[] = [
  { key: "preparing_audio", label: "Đang chuẩn bị dữ liệu âm thanh…", short: "1. Chuẩn bị âm thanh" },
  { key: "separating_vocals", label: "Đang tách riêng giọng hát AI (BS-RoFormer)…", short: "2. Tách giọng hát" },
  { key: "recognizing_lyrics", label: "Đang nhận diện lời hát đa ngôn ngữ…", short: "3. Nhận diện lời" },
  { key: "aligning_timestamps", label: "Đang căn chỉnh dấu thời gian từng từ/câu…", short: "4. Căn thời gian" },
  { key: "saving_transcript", label: "Đang lưu bản lời bài hát…", short: "5. Lưu kết quả" },
  { key: "completed", label: "Hoàn tất!", short: "6. Hoàn tất" },
];

export const STAGE_LABELS: Record<string, string> = {
  processing_gpu: "Modal GPU đang xử lý tách giọng, nhận lời và rà soát; đang chờ kết quả…",
  preparing_audio: "Bước 1/5 · Đang chuẩn bị dữ liệu âm thanh…",
  separating_vocals: "Bước 2/5 · Đang tách riêng giọng hát AI (BS-RoFormer)…",
  isolating_vocals: "Bước 2/5 · Đang tách riêng giọng hát AI (BS-RoFormer)…",
  loading_whisper: "Bước 3/5 · Đang nạp mô hình nhận diện đa ngôn ngữ…",
  detecting_language: "Bước 3/5 · Đang tự động nhận diện ngôn ngữ bài hát…",
  recognizing_lyrics: "Bước 3/5 · Đang nhận diện lời hát từ kênh giọng hát…",
  transcribing_lyrics: "Bước 3/5 · Đang trích xuất lời hát & căn dấu thời gian…",
  aligning_timestamps: "Bước 4/5 · Đang căn chỉnh dấu thời gian & cấu trúc bài hát…",
  reviewing_vocals: "Bước 4/5 · Đang rà soát ad-lib / scat…",
  refining_script: "Bước 4/5 · Đang lọc tạp âm & hoàn thiện script…",
  saving_transcript: "Bước 5/5 · Đang lưu bản lời bài hát…",
  completed: "Hoàn tất!",
  stopping: "Đang dừng tác vụ…",
};

export function resolveStageIndex(message?: string | null, progress?: number | null): number {
  if (!message) {
    if (progress != null && progress >= 0.95) return 4;
    if (progress != null && progress >= 0.85) return 3;
    if (progress != null && progress >= 0.42) return 2;
    if (progress != null && progress >= 0.12) return 1;
    return 0;
  }
  if (message === "preparing_audio") return 0;
  if (message === "processing_gpu") return 1;
  if (message === "separating_vocals" || message === "isolating_vocals") return 1;
  if (
    message === "recognizing_lyrics" ||
    message === "loading_whisper" ||
    message === "detecting_language" ||
    message === "transcribing_lyrics"
  )
    return 2;
  if (message === "aligning_timestamps" || message === "refining_script" || message === "reviewing_vocals") return 3;
  if (message === "saving_transcript") return 4;
  if (message === "completed") return 5;
  return 2;
}

export function getEffectiveReviewStatus(seg: SongscriptionSegment): ReviewStatus {
  if (seg.review_status === "verified" || seg.review_status === "edited" || seg.review_status === "skipped") {
    return seg.review_status;
  }
  const ev = getEffectiveEventType(seg);
  const words = seg.words ?? [];
  const uncertainWordCount = words.filter(w => w.uncertain).length;
  const manyUncertainWords = uncertainWordCount >= 2 && words.length > 0 && uncertainWordCount / words.length >= 0.28;
  if (
    seg.review_status === "needs_review" ||
    ev === "unknown_vocal" ||
    seg.low_confidence ||
    manyUncertainWords
  ) {
    return "needs_review";
  }
  return "verified";
}

export function isLineFlaggedForReview(seg: SongscriptionSegment): boolean {
  const st = getEffectiveReviewStatus(seg);
  return st === "needs_review" || st === "skipped";
}

export function getReviewReason(seg: SongscriptionSegment): string {
  if (seg.review_reason) return seg.review_reason;
  const ev = getEffectiveEventType(seg);
  if (ev === "unknown_vocal") {
    return "Nghi có giọng hát nhưng ASR chưa nhận rõ chữ — cần nghe đối chiếu";
  }
  const st = getEffectiveReviewStatus(seg);
  if (st === "skipped") return "Đã đánh dấu xem lại sau";
  const uncertainWords = (seg.words ?? []).filter(w => w.uncertain).length;
  if (seg.confidence != null && seg.confidence < 0.55) {
    return `Điểm ASR thấp (${Math.round(seg.confidence * 100)}%)`;
  }
  if (uncertainWords > 0) {
    return `${uncertainWords} từ có độ chắc chắn thấp`;
  }
  if (seg.confidence == null && seg.low_confidence) {
    return "Chưa khớp điểm neo âm học";
  }
  return "Cần kiểm tra";
}

export type ReviewStats = {
  total: number;
  needsReview: number;
  verified: number;
  edited: number;
  skipped: number;
  reviewedCount: number;
  progressPercent: number;
};

export function computeReviewStats(segments: SongscriptionSegment[]): ReviewStats {
  let needsReview = 0;
  let verified = 0;
  let edited = 0;
  let skipped = 0;
  for (const seg of segments) {
    const st = getEffectiveReviewStatus(seg);
    if (st === "needs_review") needsReview++;
    else if (st === "skipped") skipped++;
    else if (st === "edited") edited++;
    else verified++;
  }
  const total = segments.length;
  const flaggedTotal = needsReview + skipped;
  const reviewedCount = total - flaggedTotal;
  const progressPercent = total > 0 ? Math.round((reviewedCount / total) * 100) : 0;
  return { total, needsReview: flaggedTotal, verified, edited, skipped, reviewedCount, progressPercent };
}

export function computeAvgConfidence(segments: SongscriptionSegment[]): number | null {
  const vals = segments
    .map(s => s.confidence)
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!vals.length) return null;
  const sum = vals.reduce((acc, v) => acc + v, 0);
  return sum / vals.length;
}

export function formatClock(sec: number): string {
  const safe = Math.max(0, Math.floor(sec));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

export function formatPreciseClock(sec: number): string {
  const safe = Math.max(0, sec);
  const mins = Math.floor(safe / 60);
  const rem = safe - mins * 60;
  return `${mins}:${rem.toFixed(2).padStart(5, "0")}`;
}

export function formatLrcClock(sec: number): string {
  const safe = Math.max(0, sec);
  const mins = Math.floor(safe / 60);
  const rem = safe - mins * 60;
  return `${String(mins).padStart(2, "0")}:${rem.toFixed(2).padStart(5, "0")}`;
}

export function formatSrtClock(sec: number): string {
  const totalMs = Math.max(0, Math.round(sec * 1000));
  const hrs = Math.floor(totalMs / 3_600_000);
  const mins = Math.floor((totalMs % 3_600_000) / 60_000);
  const secs = Math.floor((totalMs % 60_000) / 1000);
  const ms = totalMs % 1000;
  return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

export function downloadTextFile(filename: string, content: string, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function stripImportedLyricsFormat(raw: string): string {
  const normalized = raw.replace(/\r\n?/g, "\n");
  const srtTime = /^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$/m;
  if (srtTime.test(normalized)) {
    return normalized
      .split(/\n\s*\n/)
      .map(block =>
        block
          .split("\n")
          .map(line => line.trim())
          .filter(line => line && !/^\d+$/.test(line) && !/^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->/.test(line))
          .join(" "),
      )
      .filter(Boolean)
      .join("\n");
  }

  return normalized
    .split("\n")
    .flatMap(line => {
      const trimmed = line.trim();
      if (/^\[(?:ti|ar|al|by|la|length|offset):.*\]$/i.test(trimmed)) return [];

      // LRC permits several timestamps before one lyric to represent repeated
      // occurrences. Keep each occurrence as a reference line; the backend
      // will recalculate its timing from the audio rather than trusting tags.
      const stamped = trimmed.match(/^(?:\[(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?\])+\s*/);
      if (stamped) {
        const count = (stamped[0].match(/\[/g) ?? []).length;
        const text = trimmed
          .slice(stamped[0].length)
          .replace(/<\d{1,2}:\d{2}(?:\.\d{1,3})?>/g, "")
          .trim();
        return text ? Array.from({ length: count }, () => text) : [];
      }

      return [
        trimmed
          .replace(/\[(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]/g, "")
          .replace(/<\d{1,2}:\d{2}(?:\.\d{1,3})?>/g, "")
          .trim(),
      ];
    })
    .filter((line, idx, arr) => line !== "" || (idx > 0 && arr[idx - 1] !== ""))
    .join("\n")
    .trim();
}

export function filterSegmentsForExport(segments: SongscriptionSegment[], policy: ExportPolicy): SongscriptionSegment[] {
  if (policy === "main_only") {
    return segments.filter(s => isMainLyric(s) && s.text.trim().length > 0);
  }
  return segments.filter(s => s.text.trim().length > 0 || getEffectiveEventType(s) === "unknown_vocal");
}

function formatSegmentExportText(seg: SongscriptionSegment, policy: ExportPolicy, multiStream: boolean): string {
  const ev = getEffectiveEventType(seg);
  let body = seg.text.trim();
  if (!body && ev === "unknown_vocal") {
    body = "[Nghi có vocal]";
  } else if (ev === "humming" && !body.startsWith("(")) {
    body = `(${body})`;
  } else if (seg.is_adlib && ev === "vocable" && policy === "merged_labeled" && !body.startsWith("(")) {
    body = `(${body})`;
  }
  if (policy === "merged_labeled" && multiStream) {
    return `[${streamLabel(seg.stream_id)}] ${body}`;
  }
  return body;
}

export function exportTranscriptFiles(
  title: string,
  language: string,
  languageName: string,
  segments: SongscriptionSegment[],
) {
  const safeBaseName = (title || "lyrics").replace(/[\\/:*?"<>|]+/g, "_");
  const hasWordTimings = segments.some(s => s.words && s.words.length > 0);
  const streamIds = Array.from(new Set(segments.map(s => s.stream_id || "stream_a")));
  const multiStream = streamIds.length > 1;

  const runByPolicy = (policy: ExportPolicy, emit: (list: SongscriptionSegment[], suffix: string) => void) => {
    if (policy === "per_stream" && multiStream) {
      for (const sid of streamIds) {
        const subset = filterSegmentsForExport(
          segments.filter(s => (s.stream_id || "stream_a") === sid),
          "with_adlib",
        );
        if (subset.length) emit(subset, `_${sid}`);
      }
      return;
    }
    emit(filterSegmentsForExport(segments, policy), policy === "main_only" ? "_main" : "");
  };

  return {
    hasWordTimings,
    multiStream,
    streamIds,
    exportTxt: (includeSections = true, policy: ExportPolicy = "with_adlib") => {
      runByPolicy(policy, (list, suffix) => {
        const lines: string[] = [];
        for (const seg of list) {
          if (includeSections && seg.section) {
            if (lines.length) lines.push("");
            lines.push(`[${seg.section}]`);
          }
          lines.push(formatSegmentExportText(seg, policy, multiStream));
        }
        const header = `${title}\nNgôn ngữ: ${languageName}\n${"—".repeat(40)}\n\n`;
        downloadTextFile(`${safeBaseName}${suffix}.txt`, header + lines.join("\n") + "\n");
      });
    },
    exportLrc: (policy: ExportPolicy = "with_adlib") => {
      runByPolicy(policy, (list, suffix) => {
        const meta = [`[ti:${title}]`, `[la:${language}]`, `[by:Harmony Auralis AI Lyrics]`, ""];
        const lines = list.map(s => `[${formatLrcClock(s.start)}]${formatSegmentExportText(s, policy, multiStream)}`);
        downloadTextFile(`${safeBaseName}${suffix}.lrc`, [...meta, ...lines].join("\n") + "\n");
      });
    },
    exportEnhancedLrc: (policy: ExportPolicy = "with_adlib") => {
      runByPolicy(policy, (list, suffix) => {
        const meta = [
          `[ti:${title}]`,
          `[la:${language}]`,
          `[by:Harmony Auralis AI Lyrics (Enhanced Word-Sync)]`,
          "",
        ];
        const lines = list.map(s => {
          const lineStart = `[${formatLrcClock(s.start)}]`;
          const prefix = policy === "merged_labeled" && multiStream ? `[${streamLabel(s.stream_id)}] ` : "";
          if (s.words && s.words.length > 0 && getEffectiveEventType(s) !== "humming") {
            const wordTags = s.words.map(w => `<${formatLrcClock(w.start)}>${w.word}`).join("");
            return `${lineStart}${prefix}${wordTags}`;
          }
          return `${lineStart}${formatSegmentExportText(s, policy, multiStream)}`;
        });
        downloadTextFile(`${safeBaseName}${suffix}.elrc.lrc`, [...meta, ...lines].join("\n") + "\n");
      });
    },
    exportSrt: (policy: ExportPolicy = "with_adlib") => {
      runByPolicy(policy, (list, suffix) => {
        const blocks = list.map(
          (s, idx) =>
            `${idx + 1}\n${formatSrtClock(s.start)} --> ${formatSrtClock(s.end)}\n${formatSegmentExportText(s, policy, multiStream)}`,
        );
        downloadTextFile(`${safeBaseName}${suffix}.srt`, blocks.join("\n\n") + "\n");
      });
    },
    exportJson: (extra?: Record<string, unknown>) => {
      downloadTextFile(
        `${safeBaseName}.lyrics.json`,
        JSON.stringify(
          {
            title,
            language,
            language_name: languageName,
            ...extra,
            segments,
          },
          null,
          2,
        ),
        "application/json;charset=utf-8",
      );
    },
  };
}

