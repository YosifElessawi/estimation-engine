import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";

// The full table: the FastAPI server (../server) answers every /api request; scripts/dev.mjs
// starts it on a free local port and passes that address here.
// The public table (NEXT_PUBLIC_EDITION=public): `npm run build:public` writes plain static files
// to out/, which `python -m server.public serve` serves next to its own /api.
const api = process.env.ESTIMATION_API ?? "http://127.0.0.1:8000";
const publicEdition = process.env.NEXT_PUBLIC_EDITION === "public";

export default function config(phase: string): NextConfig {
  if (publicEdition && phase === PHASE_PRODUCTION_BUILD) return { output: "export" };
  return {
    async rewrites() {
      return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
    },
    experimental: {
      // Jumping to the end of a watched game plays the whole Bola first; PIMC bots take seconds.
      proxyTimeout: 300_000,
    },
  };
}
