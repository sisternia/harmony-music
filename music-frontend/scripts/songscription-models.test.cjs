const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

function loadHelper(filename) {
  const source = readFileSync(path.join(__dirname, '../src/features/studio/utils', filename), 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  const api = {};
  vm.runInNewContext(output.outputText, { exports: api });
  return api;
}

const models = loadHelper('songscriptionModels.ts');

test('new and saved legacy selections use Qwen while Whisper remains selectable', () => {
  assert.equal(models.DEFAULT_SONGSCRIPTION_MODEL_ID, 'qwen3-asr-1.7b');
  for (const id of [undefined, null, '', 'gemini-songscription', 'turbo', 'qwen3-asr-1.7b']) {
    assert.equal(models.normalizeSongscriptionModelId(id), 'qwen3-asr-1.7b');
  }
  assert.equal(models.normalizeSongscriptionModelId(' WHISPER-LARGE-V3 '), 'whisper-large-v3');
});

test('cached legacy choices cannot put the removed recognizer back into a select', () => {
  const cached = [{ id: 'gemini-songscription', label: 'Gemini Audio' }, { id: 'whisper-large-v3', label: 'Whisper' }];
  const result = models.getSongscriptionModels(cached);
  assert.equal(result.length, 2);
  assert.equal(result[0].id, 'qwen3-asr-1.7b');
  assert.equal(result[1].id, 'whisper-large-v3');
  assert.ok(result.every(model => !/gemini/i.test(model.id + model.label)));
});

test('vocal review maps to step four with the correct progress text', () => {
  const lyrics = loadHelper('lyricsUtils.ts');
  assert.equal(lyrics.resolveStageIndex('reviewing_vocals', .78), 3);
  assert.match(lyrics.STAGE_LABELS.reviewing_vocals, /Bước 4\/5.*ad-lib \/ scat/);
});
