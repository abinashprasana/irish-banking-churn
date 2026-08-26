import {
  BrandMark,
  type BrandMarkMotion,
  type BrandMarkTone,
  type BrandMarkVariant,
} from "@/components/brand-mark";

export type BrandLockupProps = {
  className?: string;
  descriptor?: string;
  markVariant?: BrandMarkVariant;
  motion?: BrandMarkMotion;
  tone?: BrandMarkTone;
  wordmark?: string;
};

function classes(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function BrandLockup({
  className,
  descriptor,
  markVariant = "standard",
  motion = "static",
  tone = "ink",
  wordmark = "Atlantic Ledger",
}: BrandLockupProps) {
  return (
    <span className={classes("brand-lockup", className)}>
      <BrandMark
        className="brand-lockup__mark"
        variant={markVariant}
        tone={tone}
        motion={motion}
        decorative
      />
      <span className="brand-lockup__type">
        <strong className="brand-lockup__wordmark">{wordmark}</strong>
        {descriptor ? (
          <small className="brand-lockup__descriptor">{descriptor}</small>
        ) : null}
      </span>
    </span>
  );
}
