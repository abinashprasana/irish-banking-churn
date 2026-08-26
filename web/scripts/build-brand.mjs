import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const webDirectory = resolve(scriptDirectory, "..");
const repositoryRoot = resolve(webDirectory, "..");
const manifestPath = resolve(
  repositoryRoot,
  "assets/brand/atlantic-ledger-brand.json",
);
const checkOnly = process.argv.includes("--check");

const manifestSource = await readFile(manifestPath, "utf8");
const normalizedManifestSource = manifestSource.replace(/\r\n?/g, "\n");
const manifest = JSON.parse(normalizedManifestSource);
const manifestHash = createHash("sha256")
  .update(normalizedManifestSource)
  .digest("hex");
const generatedMarker = `Generated from atlantic-ledger-brand.json; sha256:${manifestHash}`;

validateManifest(manifest);

const outputs = await createOutputs(manifest);
const drift = [];

for (const [relativePath, expected] of outputs) {
  const outputPath = resolve(repositoryRoot, relativePath);

  if (checkOnly) {
    let actual;
    try {
      actual = await readFile(outputPath);
    } catch (error) {
      if (error && error.code === "ENOENT") {
        drift.push(`${relativePath} is missing`);
        continue;
      }
      throw error;
    }

    if (
      !normalizeOutputForComparison(relativePath, actual).equals(
        normalizeOutputForComparison(relativePath, expected),
      )
    ) {
      drift.push(`${relativePath} differs from the canonical manifest`);
    }
    continue;
  }

  await writeFile(outputPath, expected);
  process.stdout.write(`wrote ${relativePath}\n`);
}

if (checkOnly && drift.length > 0) {
  process.stderr.write(
    `Atlantic Ledger brand assets are out of date:\n${drift
      .map((message) => `- ${message}`)
      .join("\n")}\nRun \`pnpm brand:build\` from web/.\n`,
  );
  process.exitCode = 1;
} else if (checkOnly) {
  process.stdout.write(
    `Atlantic Ledger brand assets match ${manifestHash.slice(0, 12)}.\n`,
  );
}

function normalizeOutputForComparison(relativePath, output) {
  if (!/\.(?:svg|ts)$/i.test(relativePath)) return output;
  return Buffer.from(output.toString("utf8").replace(/\r\n?/g, "\n"));
}

async function createOutputs(brand) {
  const standardReverse = createMarkSvg({
    brand,
    variant: "standard",
    tone: "reverse",
  });
  const microReverse = createMarkSvg({
    brand,
    variant: "micro",
    tone: "reverse",
  });
  const standardMono = createMarkSvg({
    brand,
    variant: "standard",
    tone: "ink",
  });
  const tile = createTileSvg(brand);
  const favicon128 = await sharp(Buffer.from(tile)).resize(128, 128).png().toBuffer();
  const appleIcon180 = await sharp(Buffer.from(tile))
    .resize(180, 180)
    .png()
    .toBuffer();
  const comparison1x = createComparisonSheet(brand, 1);
  const comparison2x = createComparisonSheet(brand, 2);

  return new Map([
    [brand.exports.standardSvg, Buffer.from(standardReverse)],
    [brand.exports.microSvg, Buffer.from(microReverse)],
    [brand.exports.monoSvg, Buffer.from(standardMono)],
    [brand.exports.favicon128, favicon128],
    [brand.exports.comparison1x, Buffer.from(comparison1x)],
    [brand.exports.comparison2x, Buffer.from(comparison2x)],
    [brand.exports.nextIcon, Buffer.from(tile)],
    [brand.exports.appleIcon180, appleIcon180],
    [brand.exports.typescript, Buffer.from(createTypescriptModule(brand))],
  ]);
}

function createComparisonSheet(brand, pixelRatio) {
  const sizes = [16, 20, 24, 32, 38, 44, 64, 96, 128];
  const width = 1080;
  const height = 1400;
  const marginX = 51;
  const cellWidth = 326;
  const cellHeight = 190;
  const gridOffsetY = 44;
  const sections = [
    {
      name: "LEDGER PAPER / INK + DUOTONE",
      y: 90,
      background: brand.colors.ledgerPaper,
      foreground: brand.colors.atlanticInk,
      border: "#BDC7C9",
      reverse: false,
    },
    {
      name: "ATLANTIC INK / REVERSE",
      y: 748,
      background: brand.colors.atlanticInk,
      foreground: brand.colors.ledgerPaper,
      border: brand.colors.atlanticBlue,
      reverse: true,
    },
  ];

  const sectionMarkup = sections
    .map((section) => {
      const cells = sizes
        .map((size, index) => {
          const column = index % 3;
          const row = Math.floor(index / 3);
          const x = marginX + column * cellWidth;
          const y = section.y + gridOffsetY + row * cellHeight;
          const variant = size < 24 ? "micro" : "standard";
          const tone = section.reverse ? "reverse" : size >= 32 ? "duotone" : "ink";
          const paths = brand.marks[variant].paths;
          const colors = brand.tones[tone];
          const scale = size / 64;
          const markX = x + (cellWidth - size) / 2;
          const markY = y + 40 + (128 - size) / 2;
          const label = `${size}px / ${variant} / ${tone}`.toUpperCase();

          return `<g>
  <line x1="${x}" y1="${y + cellHeight}" x2="${x + cellWidth - 16}" y2="${y + cellHeight}" stroke="${section.border}" stroke-width="1"/>
  <text x="${x + 8}" y="${y + 22}" fill="${section.foreground}" font-family="monospace" font-size="11" letter-spacing="1.1">${label}</text>
  <g transform="translate(${markX} ${markY}) scale(${scale})">
    <path d="${paths.diagonal}" fill="${colors.diagonal}"/>
    <path d="${paths.gateBody}" fill="${colors.gateBody}"/>
    <path d="${paths.crossbar}" fill="${colors.crossbar}"/>
  </g>
</g>`;
        })
        .join("\n");

      return `<g>
  <rect x="34" y="${section.y - 20}" width="1012" height="630" fill="${section.background}" stroke="${section.border}" stroke-width="1"/>
  <text x="${marginX + 8}" y="${section.y + 8}" fill="${section.foreground}" font-family="monospace" font-size="12" font-weight="700" letter-spacing="1.6">${section.name}</text>
  ${cells}
</g>`;
    })
    .join("\n");

  return `<!-- ${generatedMarker} -->
<svg xmlns="http://www.w3.org/2000/svg" width="${width * pixelRatio}" height="${height * pixelRatio}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">
  <rect width="${width}" height="${height}" fill="#E8E5DC"/>
  <text x="42" y="38" fill="${brand.colors.atlanticInk}" font-family="serif" font-size="22" font-weight="700">Atlantic Ledger / Ledger Gate size proof</text>
  <text x="42" y="59" fill="${brand.colors.atlanticBlue}" font-family="monospace" font-size="10" letter-spacing="1.4">CANONICAL 64×64 GEOMETRY · MICRO MASTER AT 16–23PX</text>
  ${sectionMarkup}
</svg>`;
}

function createMarkSvg({ brand, variant, tone }) {
  const paths = brand.marks[variant].paths;
  const colors = brand.tones[tone];
  const variablePrefix = "--atlantic-ledger";

  return `<!-- ${generatedMarker} -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="${brand.viewBox}" role="img" aria-label="${escapeXml(brand.brand.label)}" focusable="false">
  <title>${escapeXml(brand.brand.label)}</title>
  <path class="ledger-gate-part ledger-gate-diagonal" data-brand-part="diagonal" d="${paths.diagonal}" fill="var(${variablePrefix}-diagonal, ${colors.diagonal})"/>
  <path class="ledger-gate-part ledger-gate-body" data-brand-part="gate-body" d="${paths.gateBody}" fill="var(${variablePrefix}-gate-body, ${colors.gateBody})"/>
  <path class="ledger-gate-part ledger-gate-crossbar" data-brand-part="crossbar" d="${paths.crossbar}" fill="var(${variablePrefix}-crossbar, ${colors.crossbar})"/>
</svg>
`;
}

function createTileSvg(brand) {
  const paths = brand.marks.micro.paths;
  const ink = brand.colors.atlanticInk;
  const paper = brand.colors.ledgerPaper;

  return `<!-- ${generatedMarker} -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="${brand.viewBox}" role="img" aria-label="${escapeXml(brand.brand.label)}" focusable="false">
  <title>${escapeXml(brand.brand.label)}</title>
  <rect width="64" height="64" fill="${ink}"/>
  <path data-brand-part="diagonal" d="${paths.diagonal}" fill="${paper}"/>
  <path data-brand-part="gate-body" d="${paths.gateBody}" fill="${paper}"/>
  <path data-brand-part="crossbar" d="${paths.crossbar}" fill="${paper}"/>
</svg>
`;
}

function createTypescriptModule(brand) {
  const serializedColors = JSON.stringify(brand.colors, null, 2);
  const serializedMarks = JSON.stringify(brand.marks, null, 2);
  const serializedTones = JSON.stringify(brand.tones, null, 2);

  return `// ${generatedMarker}. Do not edit.

export const BRAND_LABEL = ${JSON.stringify(brand.brand.label)} as const;
export const BRAND_VIEW_BOX = ${JSON.stringify(brand.viewBox)} as const;
export const BRAND_CLEAR_SPACE = ${brand.clearSpace} as const;

export const BRAND_COLORS = ${serializedColors} as const;

export const BRAND_MARKS = ${serializedMarks} as const;

export const BRAND_TONES = ${serializedTones} as const;

export type BrandVariant = keyof typeof BRAND_MARKS;
export type BrandTone = keyof typeof BRAND_TONES;
export type BrandPart = "diagonal" | "gateBody" | "crossbar";
`;
}

function validateManifest(brand) {
  const expectedViewBox = "0 0 64 64";
  const expectedVariants = ["standard", "micro"];
  const expectedTones = ["ink", "reverse", "blue", "duotone"];
  const expectedParts = ["diagonal", "gateBody", "crossbar"];

  if (brand.viewBox !== expectedViewBox) {
    throw new Error(`Brand viewBox must be ${expectedViewBox}.`);
  }
  if (brand.clearSpace !== 10) {
    throw new Error("Brand clearSpace must be the 10-unit stem width.");
  }

  for (const variant of expectedVariants) {
    if (!brand.marks?.[variant]) {
      throw new Error(`Missing ${variant} brand master.`);
    }
    for (const part of expectedParts) {
      const path = brand.marks[variant].paths?.[part];
      if (typeof path !== "string" || path.length === 0) {
        throw new Error(`Missing ${variant}.${part} path.`);
      }
    }
  }

  for (const tone of expectedTones) {
    if (!brand.tones?.[tone]) {
      throw new Error(`Missing ${tone} brand tone.`);
    }
    for (const part of expectedParts) {
      if (!/^#[0-9A-F]{6}$/.test(brand.tones[tone][part])) {
        throw new Error(`${tone}.${part} must be an uppercase six-digit hex.`);
      }
    }
  }

  for (const [name, relativePath] of Object.entries(brand.exports ?? {})) {
    if (typeof relativePath !== "string" || relativePath.startsWith("..")) {
      throw new Error(`Invalid export path for ${name}.`);
    }
  }
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
