export const DEFAULT_SONGSCRIPTION_MODEL_ID = "qwen3-asr-1.7b";

export const SONGSCRIPTION_MODELS = [
  {
    id: DEFAULT_SONGSCRIPTION_MODEL_ID,
    label: "Qwen3-ASR 1.7B + Whisper Align (Khuyên dùng · Nhận dạng giọng hát)",
  },
  {
    id: "whisper-large-v3",
    label: "Whisper Large v3 (OpenAI · Chạy thuần GPU không cần API)",
  },
];

/** Resolve removed or saved legacy recognizers to the current singing model. */
export function normalizeSongscriptionModelId(modelId?: string | null): string {
  const id = modelId?.trim().toLowerCase();
  return id === "whisper-large-v3" ? id : DEFAULT_SONGSCRIPTION_MODEL_ID;
}

/** Keep supported options available while an older backend response is cached. */
export function getSongscriptionModels(models?: readonly { id: string; label: string }[]) {
  return SONGSCRIPTION_MODELS.map(model => {
    const current = models?.find(candidate => candidate.id === model.id);
    return current && !/gemini/i.test(current.label) ? current : model;
  });
}
