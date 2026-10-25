import type { NextConfig } from "next";

// The FastAPI server (../server) answers every /api request. scripts/dev.mjs starts it on a
// free local port and passes that address here.
const api = process.env.ESTIMATION_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
  experimental: {
    // Jumping to the end of a watched game plays the whole Bola first; PIMC bots take seconds.
    proxyTimeout: 300_000,
  },
};

export default nextConfig;
