import type { SongscriptionWord } from '../api';

export function getPlaybackPosition(time: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  const position = Number.isFinite(time) ? time : 0;
  return Math.min(duration, Math.max(0, position));
}

export function getPlaybackPercent(time: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return getPlaybackPosition(time, duration) / duration * 100;
}

export function getWordPlaybackState(
  word: Pick<SongscriptionWord, 'start' | 'end'>,
  time: number,
): 'upcoming' | 'active' | 'sung' {
  if (!Number.isFinite(time) || !Number.isFinite(word.start) || time < word.start) return 'upcoming';
  const end = Number.isFinite(word.end) ? Math.max(word.start, word.end) : word.start;
  return time < end ? 'active' : 'sung';
}
