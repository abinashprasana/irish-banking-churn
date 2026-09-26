"use client";

import { useEffect, useRef, useState } from "react";
import { animate, useReducedMotion } from "motion/react";

import { useRevealState } from "@/components/in-view";

const formats = {
  integer: (value: number) => Math.round(value).toLocaleString("en-IE"),
  fixed3: (value: number) => value.toFixed(3),
};

type CountUpProps = {
  value: number;
  format: keyof typeof formats;
};

/**
 * Counts from zero once, when the number first scrolls into view. The final value
 * sets the width through an invisible copy, so the layout never shifts, and screen
 * readers only ever hear the final value.
 */
export function CountUp({ value, format }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const state = useRevealState(ref);
  const reduceMotion = useReducedMotion();
  const final = formats[format](value);
  const [animated, setAnimated] = useState(formats[format](0));

  useEffect(() => {
    if (state !== "shown" || reduceMotion) return;
    const controls = animate(0, value, {
      duration: 0.9,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (latest) => setAnimated(formats[format](latest)),
    });
    return () => controls.stop();
  }, [state, reduceMotion, value, format]);

  const shown = reduceMotion || state === undefined ? final : state === "pending" ? formats[format](0) : animated;

  return (
    <span ref={ref} className="count-up">
      <span className="count-up__sizer" aria-hidden="true">
        {final}
      </span>
      <span className="count-up__value" aria-hidden="true">
        {shown}
      </span>
      <span className="visually-hidden">{final}</span>
    </span>
  );
}
