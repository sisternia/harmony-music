const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// Exercise the production helpers without adding a test runner or depending
// on Node's version-specific TypeScript loader.
const source = readFileSync(path.join(__dirname, '../src/features/studio/utils/lyricsPlayback.ts'), 'utf8');
const output = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const api = {};
vm.runInNewContext(output.outputText, { exports: api });
const { getPlaybackPosition, getPlaybackPercent, getWordPlaybackState, getActiveSegmentIds, prepareWordsForKaraoke } = api;

test('seek fill follows elapsed audio instead of a fixed halfway position', () => {
  assert.equal(getPlaybackPercent(0, 262.25), 0);
  assert.ok(Math.abs(getPlaybackPercent(13, 262.25) - 4.957102) < 0.000001);
  assert.equal(getPlaybackPercent(131.125, 262.25), 50);
  assert.equal(getPlaybackPercent(262.25, 262.25), 100);
});

test('seeking is clamped to the beginning and end, including very short audio', () => {
  assert.equal(getPlaybackPosition(-5, 10), 0);
  assert.equal(getPlaybackPosition(15, 10), 10);
  assert.equal(getPlaybackPercent(5, 0.5), 100);
  assert.equal(getPlaybackPosition(0.25, 0.5), 0.25);
});

test('unloaded or invalid duration never paints a phantom progress fill', () => {
  for (const duration of [0, -1, NaN, Infinity]) {
    assert.equal(getPlaybackPercent(13, duration), 0);
    assert.equal(getPlaybackPosition(13, duration), 0);
  }
  assert.equal(getPlaybackPercent(NaN, 10), 0);
});

test('word boundaries activate the current word and retain completed words', () => {
  const first = { start: 0.55, end: 1.43 };
  const next = { start: 1.43, end: 1.93 };
  assert.equal(getWordPlaybackState(first, 0.54), 'upcoming');
  assert.equal(getWordPlaybackState(first, 0.55), 'active');
  assert.equal(getWordPlaybackState(first, 1.43), 'sung');
  assert.equal(getWordPlaybackState(next, 1.43), 'active');
  assert.equal(getWordPlaybackState(first, 13), 'sung');
});

test('seeking backwards clears words that have not been sung at the new position', () => {
  const word = { start: 8.47, end: 9.07 };
  assert.equal(getWordPlaybackState(word, 13), 'sung');
  assert.equal(getWordPlaybackState(word, 0), 'upcoming');
  assert.equal(getWordPlaybackState(word, 8.8), 'active');
});

test('zero-length and malformed words cannot leave a stuck active highlight', () => {
  assert.equal(getWordPlaybackState({ start: 2, end: 2 }, 2), 'sung');
  assert.equal(getWordPlaybackState({ start: 2, end: 1 }, 2), 'sung');
  assert.equal(getWordPlaybackState({ start: 2, end: NaN }, 2), 'sung');
  assert.equal(getWordPlaybackState({ start: NaN, end: 3 }, 2), 'upcoming');
});

test('overlapping multi-stream segments highlight simultaneously without stealing focus', () => {
  const segs = [
    { id: 1, start: 10.0, end: 13.0, stream_id: 'stream_a' },
    { id: 2, start: 11.0, end: 12.5, stream_id: 'stream_b' },
    { id: 3, start: 13.0, end: 15.0, stream_id: 'stream_a' },
  ];
  const at115 = getActiveSegmentIds(segs, 11.5);
  assert.deepEqual([...at115], [1, 2]);
  const at130 = getActiveSegmentIds(segs, 13.0);
  assert.deepEqual([...at130], [3]);

  // Same-stream overlapping rows MUST NEVER highlight two rows on stream_a simultaneously
  const sameStreamOv = [
    { id: 10, start: 22.0, end: 25.0, stream_id: 'stream_a' },
    { id: 11, start: 24.0, end: 26.5, stream_id: 'stream_a' },
  ];
  assert.deepEqual([...getActiveSegmentIds(sameStreamOv, 23.5)], [10]);
  assert.deepEqual([...getActiveSegmentIds(sameStreamOv, 24.2)], [11]);
});

test('prepareWordsForKaraoke repairs collapsed forced-aligner word runs for smooth tracking', () => {
  // Real collapsed segment: "go" stalled 3.84s, followed by 5 words stacked at 78.72-78.77
  const collapsed = [
    { word: "Don't", start: 74.72, end: 74.88 },
    { word: ' wanna', start: 74.88, end: 74.93 },
    { word: ' go', start: 74.88, end: 78.72 },
    { word: " 'cause", start: 78.72, end: 78.77 },
    { word: ' I', start: 78.72, end: 78.77 },
    { word: ' really', start: 78.72, end: 78.77 },
    { word: ' wanna', start: 78.72, end: 78.77 },
    { word: ' stay', start: 78.80, end: 78.85 },
  ];
  const fixed = prepareWordsForKaraoke(collapsed, 74.72, 78.85, 80.80);
  assert.ok(fixed[2].end - fixed[2].start < 1.2, 'stalled word must not hog 3.84s');
  for (let i = 0; i < fixed.length; i++) {
    assert.ok(fixed[i].end - fixed[i].start >= 0.1, `word ${i} duration too short`);
    if (i > 0) assert.ok(fixed[i].start >= fixed[i - 1].end, `word ${i} overlaps previous`);
  }
  // Small <= 0.16s gap between consecutive words bridges smoothly without flickering off
  assert.equal(getWordPlaybackState({ start: 2.31, end: 2.79 }, 2.83, 2.87), 'active');
  assert.equal(getWordPlaybackState({ start: 2.31, end: 2.79 }, 2.87, 2.87), 'sung');
});
