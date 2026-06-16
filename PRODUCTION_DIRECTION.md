# PRODUCTION_DIRECTION.md — read this, do not act on it yet

## Status
The 8-phase build is **complete**. What exists today is a **demo**: it processes
**recorded video files** (`./videos/`) offline on an Apple-Silicon MacBook (MPS), writes a static
`analytics.json`, and renders a static Next.js dashboard. This is correct and finished for the
client meeting.

## The known next horizon (DO NOT BUILD NOW)
After we win the deal, the client gives us **live CCTV access**. The production system processes
**live camera streams (RTSP/ONVIF) continuously**, not recorded files one-shot. **Do not write any
of this yet.** No RTSP code, no database, no API server, no edge-deployment, no streaming service.
The demo stays exactly as it is. This section exists only so you don't make decisions now that make
the live pivot harder later.

**Guiding principle for production: "same brain, new body."** The analytics core is reused; only the
input and the serving shell change. The ideal end state is that going live means **swapping the input
source (recorded file → live stream), not rewriting the logic.**

## So, from here on, when making ANY change to the codebase:
Keep these seams clean so the future pivot is a swap, not a rewrite. (These are constraints to
*preserve*, not features to build.)

1. **Keep the analytics modules input-agnostic.** `detect_track`, `zones`, `identity`,
   `cross_camera`, `aggregate` should operate on frames/tracks regardless of whether the source is a
   file or a stream. Do not bake "recorded file" assumptions deep into them. The file-vs-stream
   decision lives only at the ingestion boundary.
2. **Keep `device` swappable.** Demo runs `mps`. Production edge hardware runs `cuda`/TensorRT. Don't
   hardcode MPS inside modules — keep it config-driven as it already is.
3. **Treat `analytics.json` as a stable contract.** In production the same shape will be served from a
   database/API; the dashboard must not need rewriting. Version the schema; don't break it casually.
4. **Keep models behind the provider abstraction.** The licence swap (see below) must be a config/
   model change, not a code rewrite.

## Known production facts (context only — not to build now)
- **Deployment shape:** processing runs on an **edge box at the store** (e.g. NVIDIA Jetson Orin or a
  small on-prem GPU machine) that pulls the local camera streams and runs the CV there. Only
  **metadata** (counts, dwell, embeddings, alerts) goes to the cloud. The **dashboard + API are
  cloud-hosted**. We do **not** stream raw video to the cloud (bandwidth + privacy).
- **Licence gate (hard requirement before any PAID deployment):** the demo uses YOLO26 (**AGPL-3.0**)
  and InsightFace **non-commercial** face weights. These are demo-only. Production must swap detection
  to **RF-DETR (Apache-2.0)** or an Ultralytics Enterprise licence, and license/retrain compliant face
  weights. Flag this the moment production work begins.
- **New shell pieces production will need (later):** live RTSP/ONVIF ingestion with reconnect; a
  continuous 24/7 service with rolling windowed aggregation; a real DB (Postgres/TimescaleDB) +
  object storage; a live API the dashboard polls; auth, retention/DPIA, monitoring.

## Operating rule
Re-read this file (plus `PROGRESS.md` and `COPILOT_BUILD_PROMPT.md`) at the start of every session.
**Do not begin any production/live-CCTV work until explicitly instructed.** Until then, this is
awareness only — it shapes how you keep the seams clean, nothing more.
