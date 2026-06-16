# CounterVision — Copilot instructions

CounterVision is an **offline multi-camera CCTV → retail-analytics demo** for a client meeting
(product by Agents Limited). It processes recorded `.mp4`s in `./videos/` with a Python CV pipeline
(YOLO26 + supervision + InsightFace on Apple Silicon MPS) and renders a navy-themed Next.js dashboard.
The point is to **impress a client and show commercial potential** — visual polish and a believable
retail story matter as much as the code.

## Before doing anything
1. Read **`COPILOT_BUILD_PROMPT.md`** in full — it is the complete, self-contained build spec (the
   "what", the "why", and the phase order). Everything you need is in there.
2. Read/maintain **`PROGRESS.md`** (current phase, decisions, next steps). Create it if missing.
   Re-read both files at the start of every session.
3. Read **`PRODUCTION_DIRECTION.md`** before making any code change. The 8-phase demo build is
   complete; the next horizon is **live CCTV in production**, but **do not begin any
   production/live-CCTV work until explicitly instructed**. The doc lists four architectural
   seams to preserve so the future pivot is a swap, not a rewrite (input-agnostic analytics
   modules, config-driven `device`, stable `analytics.json` schema, models behind a provider
   abstraction).

## How to work
- Build **strictly phase-by-phase** per `COPILOT_BUILD_PROMPT.md`: THINK → CODE → VALIDATE (run on real
  footage) → PUSH → update `PROGRESS.md` → stop and report. Do not jump ahead.
- **Never fabricate metrics or outputs.** If a value can't be computed, mark it locked.
- The **live demo must run with zero live inference** — pre-render all artifacts; the dashboard reads static files only.

## Library APIs — use Context7
Before writing code against `ultralytics` (YOLO26), `supervision`, `insightface`, `onnxruntime`,
`next`, `tailwindcss`, `shadcn`, or `recharts`, **query Context7 for the current API**. Do not rely on
training-data memory for versions or APIs — these libraries changed in 2025–2026.

## Apple Silicon (M2 Pro) rules
- Always pass `device="mps"` to Ultralytics — forgetting it silently falls back to CPU (~10× slower).
- Set `export PYTORCH_ENABLE_MPS_FALLBACK=1`. Use FP32 on MPS. Python 3.11 (onnxruntime-silicon ceiling).
- The face stage (InsightFace) is CPU-bound on M2 — sample frames, batch crops, embed only quality-gated faces.

## GitHub
- Use the **`mudassar531`** account. Create the repo and push via the **`gh`/`git` CLI** in the terminal
  (the built-in GitHub MCP is read-only). Confirm `gh auth status` shows `mudassar531` before the first push.

## Tools available
- **Context7** (live docs), **playwright** (screenshot + self-critique the dashboard UI),
  **memory** + `PROGRESS.md` (cross-session continuity), **filesystem**, built-in GitHub (read-only).