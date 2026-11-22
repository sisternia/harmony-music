const fs = require('node:fs');
const path = require('node:path');
const tokens = require('../src/theme/tokens.json');

const themePath = path.resolve(__dirname, '../src/styles/theme.css');
const kebab = value => value.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`);

function renderTheme() {
  const declarations = [
    '/* Generated from src/theme/tokens.json. Run npm run theme:generate to update. */',
    ':root {',
    '  /* Reference swatches: decoration and music visualization. */',
    ...Object.entries(tokens.palette).map(([name, value]) => `  --palette-${kebab(name)}: ${value};`),
    '',
    '  /* Shared semantic values for CSS, native views and canvas. */',
    ...Object.entries(tokens.colors).map(([name, value]) => `  --color-${kebab(name)}: ${value};`),
    '',
    '  /* UI roles and component aliases. Keep large surfaces neutral. */',
    ...Object.entries(tokens.css).map(([name, value]) => `  --${name}: ${value};`),
    '',
    '  color-scheme: light;',
    '}',
    '',
  ];
  return declarations.join('\n');
}

if (require.main === module) {
  fs.writeFileSync(themePath, renderTheme());
  console.log('Generated src/styles/theme.css from src/theme/tokens.json.');
}

module.exports = { renderTheme, themePath, tokens };
