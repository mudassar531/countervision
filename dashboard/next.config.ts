import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `make demo` builds a fully static export under `dashboard/out/` so the
  // live walkthrough has no Node server in the loop — it can be served by
  // any HTTP server, including `python3 -m http.server`. The dashboard reads
  // `public/data/analytics.json` at build time (via the page server
  // component) and bakes every panel into the HTML.
  output: "export",

  // Skip Next/Image's runtime optimizer so static export works without a
  // server. We don't use <Image /> anywhere — frames / heatmaps / thumbnails
  // are plain <img> by design (Phase 6 README explains why).
  images: { unoptimized: true },
};

export default nextConfig;
