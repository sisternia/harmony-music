import type { SongNote } from './transcription';

// One piano key can be struck again while its previous sound is sustained.
// Keep the press duration separate, and release the previous voice on retrigger.
export function normalizeMidiNotes(notes: SongNote[]): SongNote[] {
  const byPitch = new Map<number, SongNote[]>();
  for (const note of notes) {
    const group = byPitch.get(note.pitch) ?? [];
    group.push(note);
    byPitch.set(note.pitch, group);
  }

  const result: SongNote[] = [];
  for (const group of byPitch.values()) {
    group.sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
    let previous: SongNote | undefined;
    for (const note of group) {
      const keyEnd = Math.max(note.start + 0.02, note.keyEnd ?? note.end);
      const current = { ...note, keyEnd, end: Math.max(keyEnd, note.end) };
      if (previous && current.start - previous.start <= 0.005) {
        previous.keyEnd = Math.max(previous.keyEnd ?? previous.end, current.keyEnd);
        previous.end = Math.max(previous.end, current.end);
        previous.velocity = Math.max(previous.velocity, current.velocity);
        continue;
      }
      if (previous) {
        previous.keyEnd = Math.min(previous.keyEnd ?? previous.end, current.start);
        previous.end = Math.min(previous.end, current.start);
        result.push(previous);
      }
      previous = current;
    }
    if (previous) result.push(previous);
  }
  return result.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}
