"use client";

import { useEffect, useState } from "react";

import { BrandLockup } from "@/components/brand-lockup";
import { site } from "@/lib/site";

const links = [
  { href: "#system", label: "System" },
  { href: "#evidence", label: "Evidence" },
  { href: "#decision", label: "Decision replay" },
  { href: "#governance", label: "Governance" },
];

export function Navigation() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  return (
    <header className="site-header">
      <div className="header-inner">
        <a className="header-brand-link" href="#top" aria-label="Atlantic Ledger home">
          <BrandLockup
            className="header-brand-lockup"
            descriptor="Banking AI case study"
            motion="intro"
            tone="duotone"
          />
        </a>

        <nav className="desktop-nav" aria-label="Case study chapters">
          {links.map((link) => (
            <a key={link.href} href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>

        <a
          className="header-lab-link"
          href={site.labUrl}
          target="_blank"
          rel="noreferrer"
        >
          Open lab <span aria-hidden="true">↗</span>
        </a>

        <button
          className="menu-button"
          type="button"
          aria-expanded={open}
          aria-controls="mobile-navigation"
          aria-label={open ? "Close navigation" : "Open navigation"}
          onClick={() => setOpen((current) => !current)}
        >
          <span />
          <span />
        </button>
      </div>

      <nav
        id="mobile-navigation"
        className="mobile-nav"
        aria-label="Mobile case study chapters"
        data-open={open}
      >
        {links.map((link) => (
          <a key={link.href} href={link.href} onClick={() => setOpen(false)}>
            {link.label}
          </a>
        ))}
        <a
          href={site.labUrl}
          target="_blank"
          rel="noreferrer"
          onClick={() => setOpen(false)}
        >
          Open interactive lab <span aria-hidden="true">↗</span>
        </a>
      </nav>
    </header>
  );
}
