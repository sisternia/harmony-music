import type { ReviewStatus, SongscriptionSegment } from "../api";

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
  preparing_audio: "Bước 1/5 · Đang chuẩn bị dữ liệu âm thanh…",
  separating_vocals: "Bước 2/5 · Đang tách riêng giọng hát AI (BS-RoFormer)…",
  isolating_vocals: "Bước 2/5 · Đang tách riêng giọng hát AI (BS-RoFormer)…",
  loading_whisper: "Bước 3/5 · Đang nạp mô hình nhận diện đa ngôn ngữ…",
  detecting_language: "Bước 3/5 · Đang tự động nhận diện ngôn ngữ bài hát…",
  recognizing_lyrics: "Bước 3/5 · Đang nhận diện lời hát từ kênh giọng hát…",
  transcribing_lyrics: "Bước 3/5 · Đang trích xuất lời hát & căn dấu thời gian…",
  aligning_timestamps: "Bước 4/5 · Đang căn chỉnh dấu thời gian & cấu trúc bài hát…",
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
  if (message === "separating_vocals" || message === "isolating_vocals") return 1;
  if (
    message === "recognizing_lyrics" ||
    message === "loading_whisper" ||
    message === "detecting_language" ||
    message === "transcribing_lyrics"
  )
    return 2;
  if (message === "aligning_timestamps" || message === "refining_script") return 3;
  if (message === "saving_transcript") return 4;
  if (message === "completed") return 5;
  return 2;
}

export function getEffectiveReviewStatus(seg: SongscriptionSegment): ReviewStatus {
  if (seg.review_status === "verified" || seg.review_status === "edited" || seg.review_status === "skipped") {
    return seg.review_status;
  }
  const words = seg.words ?? [];
  const uncertainWordCount = words.filter(w => w.uncertain).length;
  const manyUncertainWords = uncertainWordCount >= 2 && words.length > 0 && uncertainWordCount / words.length >= 0.28;
  if (seg.review_status === "needs_review" || seg.low_confidence || manyUncertainWords) {
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
  const st = getEffectiveReviewStatus(seg);
  if (st === "skipped") return "Đã đánh dấu xem lại sau";
  const uncertainWords = (seg.words ?? []).filter(w => w.uncertain).length;
  if (seg.confidence != null && seg.confidence < 0.55) {
    return `Độ tin cậy nhận diện thấp (${Math.round(seg.confidence * 100)}%)`;
  }
  if (uncertainWords > 0) {
    return `${uncertainWords} từ có độ tin cậy thấp`;
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

export function exportTranscriptFiles(
  title: string,
  language: string,
  languageName: string,
  segments: SongscriptionSegment[],
) {
  const safeBaseName = (title || "lyrics").replace(/[\\/:*?"<>|]+/g, "_");
  const hasWordTimings = segments.some(s => s.words && s.words.length > 0);

  return {
    hasWordTimings,
    exportTxt: (includeSections = true) => {
      const lines: string[] = [];
      for (const seg of segments) {
        if (includeSections && seg.section) {
          if (lines.length) lines.push("");
          lines.push(`[${seg.section}]`);
        }
        lines.push(seg.text);
      }
      const header = `${title}\nNgôn ngữ: ${languageName}\n${"—".repeat(40)}\n\n`;
      downloadTextFile(`${safeBaseName}.txt`, header + lines.join("\n") + "\n");
    },
    exportLrc: () => {
      const meta = [`[ti:${title}]`, `[la:${language}]`, `[by:Harmony Auralis AI Lyrics]`, ""];
      const lines = segments.map(s => `[${formatLrcClock(s.start)}]${s.text}`);
      downloadTextFile(`${safeBaseName}.lrc`, [...meta, ...lines].join("\n") + "\n");
    },
    exportEnhancedLrc: () => {
      const meta = [
        `[ti:${title}]`,
        `[la:${language}]`,
        `[by:Harmony Auralis AI Lyrics (Enhanced Word-Sync)]`,
        "",
      ];
      const lines = segments.map(s => {
        const lineStart = `[${formatLrcClock(s.start)}]`;
        if (s.words && s.words.length > 0) {
          const wordTags = s.words.map(w => `<${formatLrcClock(w.start)}>${w.word}`).join("");
          return `${lineStart}${wordTags}`;
        }
        return `${lineStart}${s.text}`;
      });
      downloadTextFile(`${safeBaseName}.elrc.lrc`, [...meta, ...lines].join("\n") + "\n");
    },
    exportSrt: () => {
      const blocks = segments.map(
        (s, idx) => `${idx + 1}\n${formatSrtClock(s.start)} --> ${formatSrtClock(s.end)}\n${s.text}`,
      );
      downloadTextFile(`${safeBaseName}.srt`, blocks.join("\n\n") + "\n");
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

