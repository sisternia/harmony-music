import tokens from './tokens.json';

// Native surfaces and canvas need literal values. CSS is generated from the same source.
export const colors = {
  ...tokens.palette,
  ...tokens.colors,
} as const;

// A small range of thumbnail tints keeps songs distinguishable within the palette.
export const coverBackgrounds = [
  "var(--gradient-cover-lavender)",
  "var(--gradient-cover-rose)",
  "var(--gradient-cover-coral)",
] as const;
