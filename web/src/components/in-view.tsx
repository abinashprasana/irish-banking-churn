"use client";

import {
  type ElementType,
  type ReactNode,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from "react";

/**
 * "pending" means the element started below the fold and is waiting to be seen.
 * "shown" means it has since scrolled into view. Undefined means nothing should
 * animate: before hydration, without JavaScript, or when it was already visible
 * on load, so content never blinks out and back in.
 */
export type RevealState = "pending" | "shown" | undefined;

export function useRevealState(ref: RefObject<Element | null>): RevealState {
  const [state, setState] = useState<RevealState>(undefined);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    let first = true;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (!first) setState("shown");
          observer.disconnect();
        } else if (first) {
          setState("pending");
        }
        first = false;
      },
      { rootMargin: "0px 0px -12% 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);

  return state;
}

type InViewProps = {
  as?: ElementType;
  className?: string;
  children: ReactNode;
} & Record<string, unknown>;

export function InView({ as: Tag = "div", children, ...rest }: InViewProps) {
  const ref = useRef<HTMLElement>(null);
  const state = useRevealState(ref);
  return (
    <Tag ref={ref} data-inview={state} {...rest}>
      {children}
    </Tag>
  );
}
