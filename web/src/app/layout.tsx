import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";

import { Navigation } from "@/components/navigation";
import { site } from "@/lib/site";

import "./globals.css";

const sourceSerif = localFont({
  src: [
    {
      path: "./source-serif-4-latin.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "./source-serif-4-semibold.ttf",
      weight: "600",
      style: "normal",
    },
  ],
  variable: "--font-source-serif",
  display: "swap",
});

const plexSans = localFont({
  src: "./ibm-plex-sans-latin.woff2",
  weight: "400",
  style: "normal",
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = localFont({
  src: "./ibm-plex-mono-500-latin.woff2",
  weight: "500",
  style: "normal",
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: site.title,
  description: site.description,
  applicationName: site.name,
  authors: [{ name: site.author, url: site.repositoryUrl }],
  creator: site.author,
  category: "technology",
  keywords: [
    "Irish banking",
    "customer churn",
    "explainable AI",
    "XGBoost",
    "SHAP",
    "responsible AI",
    "AI governance",
    "machine learning case study",
  ],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    locale: "en_IE",
    url: "/",
    title: site.title,
    siteName: site.name,
    description: site.description,
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "Atlantic Ledger governed banking AI case study",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: site.title,
    description: site.description,
    images: ["/opengraph-image"],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#F4F1E8",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": ["CreativeWork", "SoftwareSourceCode"],
    name: site.name,
    headline: site.title,
    description: site.description,
    author: { "@type": "Person", name: site.author },
    codeRepository: site.repositoryUrl,
    programmingLanguage: ["TypeScript", "Python"],
    applicationCategory: "Machine learning case study",
    inLanguage: "en-IE",
    isAccessibleForFree: true,
  };

  return (
    <html
      lang="en-IE"
      className={`${sourceSerif.variable} ${plexSans.variable} ${plexMono.variable}`}
    >
      <body>
        <a className="skip-link" href="#main-content">
          Skip to case study
        </a>
        <Navigation />
        {children}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
      </body>
    </html>
  );
}
