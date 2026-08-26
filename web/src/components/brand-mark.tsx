import type { SVGProps } from "react";

import {
  BRAND_LABEL,
  BRAND_MARKS,
  BRAND_TONES,
  BRAND_VIEW_BOX,
  type BrandTone,
  type BrandVariant,
} from "@/components/brand-geometry.generated";

export type BrandMarkVariant = BrandVariant;
export type BrandMarkTone = BrandTone;
export type BrandMarkMotion = "static" | "intro";

type BrandMarkAccessibility =
  | { decorative: true; label?: never }
  | { decorative?: false; label?: string };

export type BrandMarkProps = Omit<
  SVGProps<SVGSVGElement>,
  "aria-hidden" | "aria-label" | "children" | "color"
> &
  BrandMarkAccessibility & {
    variant?: BrandMarkVariant;
    tone?: BrandMarkTone;
    motion?: BrandMarkMotion;
  };

function classes(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function BrandMark({
  className,
  decorative = false,
  label = BRAND_LABEL,
  motion = "static",
  tone = "ink",
  variant = "standard",
  ...svgProps
}: BrandMarkProps) {
  const geometry = BRAND_MARKS[variant];
  const resolvedTone = variant === "micro" && tone === "duotone" ? "ink" : tone;
  const colours = BRAND_TONES[resolvedTone];

  return (
    <svg
      {...svgProps}
      className={classes("brand-symbol", className)}
      viewBox={BRAND_VIEW_BOX}
      role={decorative ? undefined : "img"}
      aria-hidden={decorative ? true : undefined}
      aria-label={decorative ? undefined : label}
      data-motion={motion}
      data-tone={resolvedTone}
      data-variant={variant}
      focusable="false"
    >
      <path
        className="brand-symbol__part brand-symbol__diagonal"
        d={geometry.paths.diagonal}
        fill={colours.diagonal}
      />
      <path
        className="brand-symbol__part brand-symbol__gate"
        d={geometry.paths.gateBody}
        fill={colours.gateBody}
      />
      <path
        className="brand-symbol__part brand-symbol__crossbar"
        d={geometry.paths.crossbar}
        fill={colours.crossbar}
      />
    </svg>
  );
}
