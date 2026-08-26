// Generated from atlantic-ledger-brand.json; sha256:dcde5eb20b123b897ca32cd482213a38e889b42b417a320538bcf49784ffaa48. Do not edit.

export const BRAND_LABEL = "Atlantic Ledger logo" as const;
export const BRAND_VIEW_BOX = "0 0 64 64" as const;
export const BRAND_CLEAR_SPACE = 10 as const;

export const BRAND_COLORS = {
  "atlanticInk": "#071827",
  "ledgerPaper": "#F4F1E8",
  "atlanticBlue": "#245B78"
} as const;

export const BRAND_MARKS = {
  "standard": {
    "minSize": 24,
    "paths": {
      "diagonal": "M24 6 H39 L20 58 H6 Z",
      "gateBody": "M36 6 H46 V29 H40 V41 H46 V48 H58 V58 H36 Z",
      "crossbar": "M15 30 H40 V40 H15 Z"
    }
  },
  "micro": {
    "minSize": 16,
    "maxSize": 23,
    "paths": {
      "diagonal": "M24 4 H40 L20 60 H4 Z",
      "gateBody": "M36 4 H48 V28 H40 V44 H48 V48 H60 V60 H36 Z",
      "crossbar": "M14 28 H40 V40 H14 Z"
    }
  }
} as const;

export const BRAND_TONES = {
  "ink": {
    "minSize": 16,
    "diagonal": "#071827",
    "gateBody": "#071827",
    "crossbar": "#071827"
  },
  "reverse": {
    "minSize": 16,
    "diagonal": "#F4F1E8",
    "gateBody": "#F4F1E8",
    "crossbar": "#F4F1E8"
  },
  "blue": {
    "minSize": 24,
    "diagonal": "#245B78",
    "gateBody": "#245B78",
    "crossbar": "#245B78"
  },
  "duotone": {
    "minSize": 32,
    "diagonal": "#245B78",
    "gateBody": "#071827",
    "crossbar": "#245B78"
  }
} as const;

export type BrandVariant = keyof typeof BRAND_MARKS;
export type BrandTone = keyof typeof BRAND_TONES;
export type BrandPart = "diagonal" | "gateBody" | "crossbar";
