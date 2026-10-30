import type { Metadata, Viewport } from "next";
import { Instrument_Sans, JetBrains_Mono, Reem_Kufi } from "next/font/google";
import type { ReactNode } from "react";

import "./globals.css";

const display = Reem_Kufi({ subsets: ["latin"], variable: "--font-display" });
const body = Instrument_Sans({ subsets: ["latin"], variable: "--font-body" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata =
  process.env.NEXT_PUBLIC_EDITION === "public"
  ? { title: "Estimation with B-Tig", description: "Play Estimation against three B-Tig bots." }
  : { title: "Estimation", description: "Play Estimation against the bots, or watch them play a Bola." };

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0a1020",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
