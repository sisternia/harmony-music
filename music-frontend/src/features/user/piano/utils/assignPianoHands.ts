import type { SongNote } from './transcription';

export function assignPianoHands(notes: SongNote[], tracks?: number[]): SongNote[] {
  if (!notes.length) return notes;

  // 1. Check if multi-track MIDI already separates Left Hand and Right Hand into distinct tracks
  if (tracks && tracks.length === notes.length) {
    const stats = new Map<number, { count: number; pitchSum: number }>();
    for (let i = 0; i < notes.length; i++) {
      const tr = tracks[i];
      const entry = stats.get(tr) ?? { count: 0, pitchSum: 0 };
      entry.count++;
      entry.pitchSum += notes[i].pitch;
      stats.set(tr, entry);
    }
    const majorTracks = [...stats.entries()]
      .filter(([, s]) => s.count >= Math.max(4, notes.length * 0.06))
      .map(([tr, s]) => ({ tr, count: s.count, mean: s.pitchSum / s.count }))
      .sort((a, b) => a.mean - b.mean);

    if (
      majorTracks.length === 2 &&
      majorTracks[0].count + majorTracks[1].count >= notes.length * 0.88 &&
      majorTracks[1].mean - majorTracks[0].mean >= 6
    ) {
      const leftTrack = majorTracks[0].tr;
      const rightTrack = majorTracks[1].tr;
      const midBoundary = (majorTracks[0].mean + majorTracks[1].mean) / 2;
      return notes.map((note, idx) => {
        const tr = tracks[idx];
        const hand: 'L' | 'R' = tr === leftTrack ? 'L' : tr === rightTrack ? 'R' : note.pitch < midBoundary ? 'L' : 'R';
        return note.hand === hand ? note : { ...note, hand };
      });
    }
  }

  // 2. Pianist cluster-aware dynamic hand separation
  const indexed = notes.map((note, index) => ({ note, index }));
  indexed.sort((a, b) => a.note.start - b.note.start || a.note.pitch - b.note.pitch);

  const hands = new Array<'L' | 'R'>(notes.length);
  let rhCenter = 67;
  let lhCenter = 48;

  let pos = 0;
  while (pos < indexed.length) {
    const cluster = [indexed[pos++]];
    while (pos < indexed.length && indexed[pos].note.start - cluster[0].note.start <= 0.032) {
      cluster.push(indexed[pos++]);
    }
    cluster.sort((a, b) => a.note.pitch - b.note.pitch);

    const minP = cluster[0].note.pitch;
    const maxP = cluster[cluster.length - 1].note.pitch;
    const clusterHands = new Array<'L' | 'R'>(cluster.length);

    if (maxP - minP > 13 && cluster.length >= 2) {
      let bestSplit = 1;
      let bestScore = -Infinity;
      for (let s = 1; s < cluster.length; s++) {
        const lhSpan = cluster[s - 1].note.pitch - minP;
        const rhSpan = maxP - cluster[s].note.pitch;
        if (lhSpan > 15 || rhSpan > 15) continue;
        const gap = cluster[s].note.pitch - cluster[s - 1].note.pitch;
        const mid = (cluster[s].note.pitch + cluster[s - 1].note.pitch) / 2;
        const score = gap - Math.abs(mid - 59) * 0.35;
        if (score > bestScore) {
          bestScore = score;
          bestSplit = s;
        }
      }
      if (bestScore === -Infinity) {
        for (let k = 0; k < cluster.length; k++) {
          clusterHands[k] = cluster[k].note.pitch < 60 ? 'L' : 'R';
        }
      } else {
        for (let k = 0; k < cluster.length; k++) {
          clusterHands[k] = k < bestSplit ? 'L' : 'R';
        }
      }
    } else {
      const avg = cluster.reduce((sum, item) => sum + item.note.pitch, 0) / cluster.length;
      let hand: 'L' | 'R';
      if (minP >= 63) hand = 'R';
      else if (maxP <= 54) hand = 'L';
      else {
        const distR = Math.abs(avg - rhCenter);
        const distL = Math.abs(avg - lhCenter);
        hand = avg >= 60 ? (distL + 2.2 < distR ? 'L' : 'R') : (distR + 2.2 < distL ? 'R' : 'L');
      }
      clusterHands.fill(hand);
    }

    let rhSum = 0, rhCnt = 0, lhSum = 0, lhCnt = 0;
    for (let k = 0; k < cluster.length; k++) {
      const hand = clusterHands[k];
      hands[cluster[k].index] = hand;
      if (hand === 'R') {
        rhSum += cluster[k].note.pitch;
        rhCnt++;
      } else {
        lhSum += cluster[k].note.pitch;
        lhCnt++;
      }
    }
    if (rhCnt) rhCenter = rhCenter * 0.72 + (rhSum / rhCnt) * 0.28;
    if (lhCnt) lhCenter = lhCenter * 0.72 + (lhSum / lhCnt) * 0.28;
  }

  return notes.map((note, idx) => (note.hand === hands[idx] ? note : { ...note, hand: hands[idx] }));
}

