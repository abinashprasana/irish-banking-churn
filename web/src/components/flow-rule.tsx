"use client";

import { type ReactNode, useRef } from "react";
import { motion, useReducedMotion, useScroll } from "motion/react";

/**
 * A vertical rule that draws down the system steps as the reader scrolls
 * through them. With reduced motion the rule is shown complete.
 */
export function FlowRule({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 75%", "end 55%"] });

  return (
    <div className="flow-rule" ref={ref}>
      <motion.span
        className="flow-rule__line"
        aria-hidden="true"
        style={reduceMotion ? undefined : { scaleY: scrollYProgress }}
      />
      {children}
    </div>
  );
}
