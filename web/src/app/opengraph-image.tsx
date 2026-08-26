import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { ImageResponse } from "next/og";

import {
  BRAND_MARKS,
  BRAND_TONES,
  BRAND_VIEW_BOX,
} from "@/components/brand-geometry.generated";

export const alt = "Atlantic Ledger governed banking AI case study";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const dynamic = "force-static";

const sourceSerif = readFile(
  join(process.cwd(), "src", "app", "source-serif-4-semibold.ttf"),
);
const plexMono = readFile(
  join(process.cwd(), "src", "app", "ibm-plex-mono-medium.ttf"),
);

export default async function OpenGraphImage() {
  const [sourceSerifData, plexMonoData] = await Promise.all([
    sourceSerif,
    plexMono,
  ]);
  const geometry = BRAND_MARKS.standard;
  const duotone = BRAND_TONES.duotone;
  const reverse = BRAND_TONES.reverse;

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        position: "relative",
        overflow: "hidden",
        background: "#F4F1E8",
        color: "#071827",
        padding: "72px 76px",
      }}
    >
      <div
        style={{
          width: "65%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <svg width="58" height="58" viewBox={BRAND_VIEW_BOX}>
            <path d={geometry.paths.diagonal} fill={duotone.diagonal} />
            <path d={geometry.paths.gateBody} fill={duotone.gateBody} />
            <path d={geometry.paths.crossbar} fill={duotone.crossbar} />
          </svg>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span
              style={{
                fontFamily: "Source Serif 4",
                fontSize: 30,
                fontWeight: 600,
                letterSpacing: -0.2,
                lineHeight: 1,
              }}
            >
              Atlantic Ledger
            </span>
            <span
              style={{
                marginTop: 8,
                color: "#245B78",
                fontFamily: "IBM Plex Mono",
                fontSize: 12,
                fontWeight: 500,
                letterSpacing: 2.2,
              }}
            >
              BANKING AI CASE STUDY
            </span>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            fontFamily: "Source Serif 4",
            fontSize: 64,
            fontWeight: 600,
            lineHeight: 0.98,
            letterSpacing: -1.4,
          }}
        >
          <span style={{ display: "flex" }}>Know who may leave</span>
          <span
            style={{
              display: "flex",
              color: "#245B78",
            }}
          >
            Decide with care
          </span>
          <div
            style={{
              marginTop: 30,
              color: "#243744",
              display: "flex",
              alignItems: "center",
              gap: 12,
              fontFamily: "IBM Plex Mono",
              fontSize: 12,
              fontWeight: 500,
              letterSpacing: 1.35,
              lineHeight: 1.2,
            }}
          >
            <span style={{ display: "flex" }}>IRISH BANKING CHURN</span>
            <span style={{ color: "#245B78" }}>/</span>
            <span style={{ display: "flex" }}>GOVERNED RETENTION INTELLIGENCE</span>
          </div>
        </div>
      </div>

      <div
        style={{
          width: 292,
          height: 292,
          marginLeft: "auto",
          flexShrink: 0,
          alignSelf: "center",
          border: "1px solid #245B78",
          display: "flex",
          position: "relative",
          alignItems: "center",
          justifyContent: "center",
          background: "#071827",
        }}
      >
        <svg width="204" height="204" viewBox={BRAND_VIEW_BOX}>
          <path d={geometry.paths.diagonal} fill={reverse.diagonal} />
          <path d={geometry.paths.gateBody} fill={reverse.gateBody} />
          <path d={geometry.paths.crossbar} fill={reverse.crossbar} />
        </svg>
        <span
          style={{
            position: "absolute",
            right: 18,
            bottom: 15,
            color: "#BDC7C9",
            fontFamily: "IBM Plex Mono",
            fontSize: 10,
            fontWeight: 500,
            letterSpacing: 1.6,
          }}
        >
          LEDGER GATE / 01
        </span>
      </div>
    </div>,
    {
      ...size,
      fonts: [
        {
          name: "Source Serif 4",
          data: sourceSerifData,
          style: "normal",
          weight: 600,
        },
        {
          name: "IBM Plex Mono",
          data: plexMonoData,
          style: "normal",
          weight: 500,
        },
      ],
    },
  );
}
