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
const { getPlaybackPosition, getPlaybackPercent, getWordPlaybackState } = api;

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
