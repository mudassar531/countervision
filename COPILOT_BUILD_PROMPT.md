# COPILOT_BUILD_PROMPT.md — CounterVision Retail Analytics Demo

> **Agent:** Copilot (Opus 4.7), agent mode.
> **Mission:** Build an offline multi-camera video-analytics pipeline + Next.js dashboard that turns
> the recorded CCTV in `./videos/` into a polished, navy-themed **retail analytics demo** for a client
> meeting. Product: **CounterVision** (Agents Limited). **The goal is to impress a client and show
> clear commercial potential — visual polish and a believable retail story matter as much as the code.**
> This file is the **complete, self-contained source of truth.** No external docs are required.

---

## 0. FIRST, DO THESE THINGS (persistence + grounding)

1. **Read `.github/copilot-instructions.md`** (auto-loads each session) and **`./PROGRESS.md`**. Create
   `PROGRESS.md` if missing; update it at the end of every phase (date, phase, what's done, what's next,
   decisions, deviations). On any new session, re-read this file + `PROGRESS.md` before acting.
2. **Use Context7 for live docs.** Before writing code against any fast-moving library — `ultralytics`
   (YOLO26), `supervision`, `insightface`, `onnxruntime`, `next`, `tailwindcss`, `shadcn`, `recharts` —
   query Context7 for the current API. Do **not** rely on training-data memory for versions/APIs (these
   all changed in 2025–2026).
3. **GitHub:** use the **`mudassar531`** account. Repo creation and pushes go through the **`gh`/`git`
   CLI in the terminal** (the built-in GitHub MCP is read-only). Confirm `gh auth status` shows
   `mudassar531` before the first push; run `gh repo create` if no remote exists.

---

## 1. Operating contract (per-phase gate — do not skip)

Work **one phase at a time**:
1. **THINK** — restate the phase goal, list files you'll create/change, note risks → write to `PROGRESS.md`.
2. **CODE** — implement only that phase.
3. **VALIDATE** — actually run it on the real footage in `./videos/`. Paste the command + a short output summary. Confirm acceptance criteria.
4. **PUSH** — commit (clear message), push to `main` on `mudassar531`. Update `PROGRESS.md`. **Then stop and report** before the next phase.

If criteria fail, fix before moving on. **Never fabricate metrics or outputs.**

---

## 2. Context

- **Footage:** recorded `.mp4`, **multiple cameras**, overhead wide/fisheye angle of an open-plan office. Files already exist (structure below). **Offline batch only — no live RTSP.**
- **The staging:** present this office footage to a client **as if it were a retail store** (a D. Watson–style health/beauty shop). Each camera = a store "area". Speak retail throughout: footfall, dwell, zones, heatmap, queue, unique/repeat visitors, store-wide journey.
- **Differentiator:** retail analytics **+ face-based repeat-visitor detection and a watchlist** (SCRFD/ArcFace). Frame any behaviour/identity flag as a **review prompt**, never an accusation.
- **Hardware:** **Apple Silicon MacBook M2 Pro** → **PyTorch MPS** + **CoreML**, **never CUDA**.
- **Brand:** deep navy **`#0A1347`** (Agents Limited), professional/corporate. Dashboard must look genuinely client-ready.
- **DEMO RELIABILITY (critical):** the live meeting must run with **zero live inference**. Pre-render **all** artifacts (analytics.json, annotated mp4s, heatmaps, thumbnails) ahead of time; the dashboard only reads static files and plays pre-rendered video. Nothing in the demo path depends on a model running in the room. Everything must load instantly and never crash mid-meeting.

---

## 3. Repo structure

**Current:**
```
COUNTERVISION/
├── .github/copilot-instructions.md
├── videos/
│   ├── camera-1/  20260607205350587.mp4
│   ├── camera-3/  20260608003129784.mp4 , 20260608005449323.mp4
│   └── camera-5/  20260608044448561_..._Barkerend - Kingz....mp4
└── COPILOT_BUILD_PROMPT.md
```

**Target (create the rest):**
```
├── PROGRESS.md                 # you maintain this
├── pipeline/                   # Python (uv-managed)
│   ├── config.yaml             # cameras → area, zones, lines, thresholds
│   ├── detect_track.py         # YOLO26(MPS) + BoT-SORT
│   ├── zones.py                # supervision LineZone / PolygonZone / HeatMap
│   ├── identity.py             # InsightFace SCRFD+ArcFace: embed, cluster, match, watchlist
│   ├── journey.py              # cross-camera person journey via ArcFace
│   ├── timeparse.py            # parse recording start time from filename (§6b)
│   ├── render.py               # annotated mp4 per camera
│   ├── aggregate.py            # analytics.json + sqlite + insights
│   ├── main.py                 # orchestrates everything
│   └── pyproject.toml / uv.lock
├── data/output/
│   ├── analytics.json          # pipeline↔dashboard contract
│   ├── analytics.db
│   ├── annotated/<camera>.mp4
│   ├── heatmaps/<camera>.png
│   ├── frames/<camera>.jpg     # clean first-frame per camera for heatmap overlay
│   └── alerts/<id>.jpg
├── dashboard/                  # Next.js 16 app (build copies artifacts into public/)
└── watchlist/                  # reference face jpgs for watchlist matching
```

**Multi-camera rule:** auto-discover cameras by listing `videos/*/`, process **every** `.mp4`, tag all outputs with `camera_id`. No hard-coded filenames.

---

## 4. Locked tech stack + key facts (self-contained)

**Detection — Ultralytics YOLO26** (`yolo26s.pt`; bump to `yolo26m.pt` if seated/distant people are missed). Released Jan 2026: NMS-free end-to-end, DFL removed, exports cleanly to CoreML/ONNX, runs on MPS. Run with `device="mps"`. *License: AGPL-3.0 — fine for this demo; see §10 for the commercial swap.*

**Tracking — BoT-SORT** (Ultralytics default; `tracker="botsort.yaml"`). Enable ReID and enlarge the track buffer so seated / briefly-occluded people keep stable IDs (dwell time depends on stable IDs). Add supervision `DetectionsSmoother` to reduce jitter. ByteTrack is the lighter fallback.

**Spatial — Roboflow `supervision`** (MIT). `LineZone` → footfall in/out; `PolygonZone` → zone presence/dwell; `HeatMapAnnotator` → foot-traffic heatmap; `JSONSink`/`CSVSink` for output. Use `InferenceSlicer` (SAHI-style tiling) if small/distant people are missed. Note the API renamed `ByteTrack`→`ByteTrackTracker` in recent versions — confirm via Context7.

**Identity — InsightFace `buffalo_l`** (SCRFD detector + ArcFace recognizer, 512-d embeddings). `face.normed_embedding` = the vector; `face.det_score` = the face-quality score (the `q=` already shown in the footage). **Quality-gate by `det_score`** before embedding; compare with **cosine similarity**, cutoff **start 0.30–0.45, tune on this footage** (record chosen value in PROGRESS.md). On Apple Silicon the CoreML provider gives little speedup for SCRFD (dynamic shapes) so the face stage is effectively **CPU-bound** — that's fine for offline batch; just sample every N frames, batch crops, and embed only quality-gated faces. *License: code MIT, but pretrained weights are non-commercial/research — demo-fine, swap for production (§10).*

**Cross-camera journey:** match ArcFace embeddings across cameras to link the same person → per-person journeys + store-wide unique count. (Face match is more reliable here than body ReID for "same person came back".)

**Dashboard — Next.js 16 (App Router) + React 19 + TypeScript + Tailwind v4 + shadcn/ui + Recharts 3.** Static read of `analytics.json`. Native `<video>` + `<canvas>` overlay. Navy `#0A1347` via shadcn CSS variables. All MIT-licensed.

**Glue:** offline pipeline → static artifacts → static Next.js. **No backend** for the demo.

---

## 5. Apple Silicon rules + environment setup

**Rules (important):**
- Always pass `device="mps"` to Ultralytics — if you forget, it silently falls back to CPU and runs ~10× slower.
- `export PYTORCH_ENABLE_MPS_FALLBACK=1` so unimplemented ops fall back to CPU instead of crashing.
- Use **FP32** on MPS (FP16 gives little benefit). Python **3.11** (onnxruntime-silicon ceiling).
- First run downloads YOLO26 + buffalo_l weights (needs internet; buffalo_l first load can take ~30s).
- **Fisheye/overhead caveat:** overhead-fisheye angles degrade COCO-trained detectors. If person recall is poor on a sample, in order: enable `InferenceSlicer`, bump to `yolo26m.pt`, crop to the usable central ROI for zones, and only consider longitude-latitude dewarping if the footage is heavily distorted.

```bash
brew install ffmpeg uv node
cd pipeline && uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch torchvision ultralytics supervision opencv-python insightface onnxruntime-silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "import torch; print('mps', torch.backends.mps.is_available())"   # must print True
cd ../ && npx create-next-app@latest dashboard --ts --eslint --app --tailwind --use-npm
cd dashboard && npx shadcn@latest init
npx shadcn@latest add sidebar card chart table button badge separator tabs avatar
# GitHub (mudassar531): gh auth status  ->  if needed: gh auth login ; gh repo create
```
Pin versions in `uv.lock` + `package-lock.json`.

---

## 6. `config.yaml` (camera → retail area)

Placeholder staging — **confirm/redraw against real footage in Phase 2**:
```yaml
fps_fallback: 15
store_name: "CounterVision Demo Store"
cameras:
  camera-1: { area: "Cosmetics & Skincare",   zones: [], entry_line: null }
  camera-3: { area: "Fragrance & Promo Aisle", zones: [], entry_line: null }
  camera-5: { area: "Entrance & Billing",      zones: [], entry_line: null }
identity:
  enabled: true
  quality_min: 0.55      # det_score gate; tune
  cosine_match: 0.38     # repeat/watchlist cutoff; tune 0.30–0.45
  sample_every_n_frames: 5
behaviour: { loiter_seconds: 90 }
```
Provide a small OpenCV click-to-draw helper to capture zone polygons + entry-line coords from a sample frame and paste them back into `config.yaml`.

### 6b. Real timestamps from filenames (do this — makes the demo credible)
Filenames encode the recording start time: `20260607205350587` → `2026-06-07 20:53:50.587` (`YYYYMMDDHHMMSSmmm`, optionally followed by `_...`). Parse this in `timeparse.py`; use `recording_start + frame_index / fps` as the real wall-clock for every event, so footfall-by-hour, journeys, and "peak hour" show real times. **Caveat:** the burned-in overlay clock disagrees with the filename — treat the **filename** as authoritative and be consistent. If a filename can't be parsed, fall back to file mtime and note it.

---

## 7. `analytics.json` schema (the contract — version it; doc in `docs/schema.md`)

```jsonc
{
  "version": 1, "generated_at": "ISO8601",
  "store": { "name": "CounterVision Demo Store", "cameras": ["camera-1","camera-3","camera-5"], "window": {"start":"ISO","end":"ISO"} },
  "kpis": { "footfall_total":0,"unique_visitors":0,"repeat_visitors":0,"watchlist_hits":0,
            "avg_dwell_seconds":0,"peak_hour":"HH:00","active_alerts":0 },
  "footfall_by_hour": [ { "hour":"20:00","count":0 } ],
  "areas": [ { "camera_id":"camera-1","area":"Cosmetics & Skincare","footfall":0,"avg_dwell_seconds":0,
               "occupancy_timeseries":[],"heatmap":"heatmaps/camera-1.png","frame":"frames/camera-1.jpg" } ],
  "visitors": [ { "person_id":"P001","first_seen":"ISO","visits":1,"cameras_seen":["camera-1","camera-5"],
                  "journey":[{"camera_id":"camera-1","t":"ISO","area":"Cosmetics & Skincare"}],
                  "is_repeat":false,"watchlist_match":null } ],
  "alerts": [ { "id":"","type":"repeat_visitor|watchlist|loiter|queue_spike","camera_id":"","area":"",
                "timestamp":"ISO","confidence":0.0,"thumbnail":"alerts/x.jpg","severity":"info|warn|high",
                "copy":"non-accusatory review prompt" } ],
  "insights": [ { "title":"","detail":"plain-English retail recommendation tied to a number" } ]
}
```
Every value comes from real computation; if something can't be computed (e.g. POS conversion), omit or mark it locked — never fake it.

---

## 8. Phases

### Phase 0 — Scaffold + camera discovery + timeparse + CI
Layout, uv env, `config.yaml`, camera auto-discovery from `videos/*/`, `timeparse.py`, logging, `PROGRESS.md`, CI smoke test (lint + run on a 5s clip, assert `analytics.json` keys).
**Done when:** `python pipeline/main.py --dry-run` lists every camera, file, **parsed recording-start time**, fps, frame size, and area without error.

### Phase 1 — Detect + track (per camera)
YOLO26s on MPS + BoT-SORT (ReID, enlarged buffer). Write tracks + `annotated/<camera>.mp4` (boxes + persistent IDs) + a clean `frames/<camera>.jpg`.
**Done when:** seated people keep stable IDs through a brief occlusion; annotated video plays; ID-switch count reported.

### Phase 2 — Zones, footfall, dwell, heatmap, occupancy
Draw zones/lines per camera. `LineZone`→footfall; `PolygonZone`+per-`tracker_id` frames÷fps→dwell; per-frame count→occupancy; `HeatMapAnnotator`→`heatmaps/<camera>.png`. **Confirm the retail area mapping against what the footage actually shows.**
**Done when:** footfall matches an eyeball count on a clip; dwell + heatmap + occupancy produced per camera.

### Phase 3 — Identity: unique + repeat + watchlist
InsightFace buffalo_l. Quality-gate, embed, cluster→unique visitors; session gallery→repeat (visit count); `./watchlist/`→watchlist hits. Emit repeat/watchlist events into `alerts[]` with thumbnails. Tune `quality_min` + `cosine_match` on real footage; record values in PROGRESS.md.
**Done when:** a reappearing person yields unique<raw-tracks + a repeat flag; a planted watchlist face triggers a hit; `identity.enabled:false` removes all face processing.

### Phase 4 — Cross-camera journey
Match embeddings across camera-1/3/5 → per-person `journey` (area→area→counter) + store-wide unique count.
**Done when:** ≥1 person is correctly linked across two cameras and their journey is in analytics.json.

### Phase 5 — Aggregate → analytics.json + sqlite + insights
Roll everything into the §7 schema + `analytics.db`. **Generate `insights[]`** — turn numbers into plain-English retail recommendations (e.g. "Highest dwell in Cosmetics but most exits without reaching Billing — consider a staffed consult there"; "Peak footfall 8–9 PM — staff accordingly"; "12 repeat visitors today — loyalty-program opportunity"). Document schema.
**Done when:** analytics.json validates, all values real, 3–5 useful insights generated from the data.

### Phase 6 — Next.js dashboard (navy, client-ready) — SPEND POLISH BUDGET HERE
Reading `analytics.json`. Screens/panels:
- **Branded Overview/landing:** CounterVision wordmark, one-line value prop, store name, date window, headline KPIs — the first thing the client sees.
- **KPI cards:** footfall, unique, repeat, watchlist hits, avg dwell, peak hour, active alerts.
- **Per-area heatmap (hero):** heatmap PNG overlaid on that camera's `frame.jpg`, with zone labels.
- **Footfall-by-hour** (Recharts area, real times), **dwell-by-area** (bar), **occupancy** (line).
- **Customer journey (highlight):** visualize a visitor moving across areas/cameras over time (stepped path / simple Sankey / arrowed area map) — the "wow" for a multi-camera store.
- **Annotated video player:** `<video>` + canvas overlay (sync overlays via `currentTime × fps`).
- **Alerts feed:** thumbnails, severity tiers, non-accusatory copy.
- **Insights panel:** the plain-English recommendations from `insights[]`.
Navy `#0A1347` theme via shadcn tokens; polished spacing/typography; responsive; loading states; visible keyboard focus. Use the Playwright MCP to screenshot and self-critique the UI until it looks genuinely client-ready. Build step copies `data/output/*` into `dashboard/public/`.
**Done when:** `npm run dev` shows a populated, polished navy dashboard driven entirely by the JSON, with the annotated video playing and the journey + insights panels populated.

### Phase 7 — One-command demo + talk-track
`make demo` runs the full pipeline over `videos/` then launches the dashboard (build with `output:'export'` so it's static and bulletproof). Write `docs/DEMO_SCRIPT.md` (~3-min walkthrough, business-value framing) + a README on pointing it at a real client's footage and redrawing zones.
**Done when:** `make demo` produces the dashboard end-to-end from raw footage in one command, and the demo runs with zero live inference.

---

## 9. Guardrails
- Real metrics only — no placeholders, no invented conversion/POS numbers (mark locked).
- Behaviour/identity alerts are **review prompts**, tiered, deduped, rate-capped (avoid alert fatigue).
- Faces: store **embeddings + thumbnails only**, never a raw-face database; keep `identity` toggleable.
- Stable, versioned `analytics.json` contract — the dashboard depends on it.
- `device="mps"`; `PYTORCH_ENABLE_MPS_FALLBACK=1`; FP32 on MPS; CoreML export only if you need ANE speed.
- **Demo path = zero live inference:** pre-render all artifacts; static dashboard; loads instantly; never crashes mid-meeting.
- Query **Context7** before coding against any library. Pin all versions. Update `PROGRESS.md` + commit per phase. Push as `mudassar531`.

## 10. Licensing & commercial swap (know this for the client Q&A)
The **demo** is free to run as-is. For a **shippable commercial product**, two things must change: YOLO26 is **AGPL-3.0** (covers your whole app, incl. SaaS) → swap detection to **RF-DETR Nano–Large (Apache-2.0)** or **YOLOX (Apache-2.0)**, ONNX/CoreML-exported; and InsightFace **pretrained weights are non-commercial** → license or retrain compliant face weights. `supervision`, Next.js, shadcn, Recharts, onnxruntime are all permissive (MIT/Apache) and safe to ship.

## 11. Stretch (only after Phase 7, only if asked)
Live RTSP ingestion, body-ReID journeys without faces, POS webhook for real conversion, edge-deploy profile, and the commercial licensing swap from §10.