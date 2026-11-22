const fs = require('node:fs');
const path = require('node:path');
const { renderTheme, themePath, tokens } = require('./generate-theme.cjs');

const root = path.resolve(__dirname, '..');
const failures = [];
const check = (condition, message) => { if (!condition) failures.push(message); };
const normalize = value => value.replace(/\r\n/g, '\n');

check(normalize(fs.readFileSync(themePath, 'utf8')) === renderTheme(),
  'theme.css is out of sync. Run npm run theme:generate.');

// Every DOM entry must import the theme as a JS dependency. Metro does not
// include our local CSS @imports in the delivered stylesheets.
for (const entry of [
  'src/features/studio/screen/ExpoStudio.tsx',
  'src/features/auth/screen/AuthDialog.tsx',
  'src/features/user/piano/screen/PianoKeyboard.tsx',
]) {
  check(/import\s+["'][^"']*styles\/theme\.css["']/.test(fs.readFileSync(path.join(root, entry), 'utf8')),
    `${entry}: missing direct theme import.`);
}

// Retained upstream HTML samples/main.css are not imported by any app route.
const isLegacy = file => file === 'src/features/user/piano/styles/main.css'
  || file.startsWith('src/features/user/piano/pages/');
function inspect(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) { inspect(full); continue; }
    const file = path.relative(root, full).replaceAll('\\', '/');
    if (!/\.(?:css|tsx?|html)$/.test(file) || isLegacy(file) || full === themePath) continue;
    const content = fs.readFileSync(full, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    check(!/#[\da-f]{3,8}\b|\brgba?\(\s*\d|\bhsla?\(\s*\d/i.test(content),
      `${file}: literal color found. Add a semantic token instead.`);
    check(!/@import\s+["'][^"']*theme\.css["']/.test(content),
      `${file}: use a direct JS theme import instead of CSS @import.`);
  }
}
inspect(path.join(root, 'src'));

// Validate references and cycles in the generated CSS custom properties.
const css = renderTheme();
const properties = new Map([...css.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map(m => [m[1], m[2]]));
function visit(name, chain = []) {
  if (chain.includes(name)) { failures.push(`Circular theme alias: ${[...chain, name].join(' -> ')}`); return; }
  const value = properties.get(name);
  if (!value) { failures.push(`Missing theme alias: ${name}`); return; }
  for (const ref of value.matchAll(/var\((--[\w-]+)/g)) visit(ref[1], [...chain, name]);
}
for (const name of properties.keys()) visit(name);

function luminance(hex) {
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map(v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
}
let minimumText = Infinity;
let pairsChecked = 0;
function contrast(foreground, background, minimum = 4.5) {
  const value = name => tokens.colors[name] || tokens.palette[name] || name;
  const [a, b] = [value(foreground), value(background)];
  check(/^#[\da-f]{6}$/i.test(a) && /^#[\da-f]{6}$/i.test(b), `Invalid contrast pair: ${foreground}/${background}`);
  const [low, high] = [luminance(a), luminance(b)].sort((x, y) => x - y);
  const ratio = (high + 0.05) / (low + 0.05);
  check(ratio >= minimum, `${foreground}/${background}: ${ratio.toFixed(2)}:1, expected ${minimum}:1.`);
  if (minimum === 4.5) minimumText = Math.min(minimumText, ratio);
  pairsChecked++;
}
for (const surface of ['background', 'backgroundRaised', 'surface', 'surfaceMuted', 'lavenderSoft', 'roseSoft', 'coralSoft']) {
  for (const text of ['text', 'textMuted', 'textSubtle', 'primary', 'secondary']) contrast(text, surface);
}
for (const pair of [
  ['onPrimary', 'primary'], ['onPrimary', 'primaryHover'],
  ['success', 'successSoft'], ['warning', 'warningSoft'], ['danger', 'dangerSoft'],
  ['instrumentText', 'instrumentSurface'], ['instrumentTextMuted', 'instrumentSurface'],
  ['instrumentTextMuted', 'keyBlack'], ['scoreInk', 'scorePaper'], ['textMuted', 'keyWhite'],
  ['printInk', 'printPaper'],
]) contrast(...pair);
for (const surface of ['surface', 'background', 'backgroundRaised', 'surfaceMuted']) contrast('controlBorder', surface, 3);

// Pressed black keys have two overlaid note gradients in velocity mode.
// Check their brightest combined fill, not just the unpressed base color.
for (const note of [tokens.palette.coral, tokens.colors.leftHand, tokens.colors.rightHand]) {
  const opacity = 1 - (1 - 0.40) * (1 - 0.24);
  const pressed = '#' + [1, 3, 5].map(i => Math.round(
    parseInt(note.slice(i, i + 2), 16) * opacity
    + parseInt(tokens.colors.keyBlack.slice(i, i + 2), 16) * (1 - opacity),
  ).toString(16).padStart(2, '0')).join('');
  contrast('onPrimary', pressed);
}

const config = require('../app.json').expo;
check(config.android.adaptiveIcon.backgroundColor === tokens.colors.surfaceMuted,
  'Android adaptive icon background differs from the surfaceMuted token.');
check(config.plugins.find(plugin => Array.isArray(plugin) && plugin[0] === 'expo-splash-screen')?.[1].backgroundColor === tokens.colors.background,
  'Splash screen background differs from the background token.');

if (failures.length) {
  console.error(failures.map(message => `- ${message}`).join('\n'));
  process.exitCode = 1;
} else {
  console.log(`Theme OK: CSS/native/canvas share tokens; no literal colors in active UI; ${pairsChecked} contrast pairs pass (text minimum ${minimumText.toFixed(2)}:1).`);
}
