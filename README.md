# CounterVision

Offline multi-camera CCTV → retail-analytics demo for a client meeting.
Product of **Agents Limited**. Processes recorded `.mp4` / `.mov` files in
`./videos/` with a Python CV pipeline (YOLO26 + supervision + InsightFace on
Apple Silicon MPS) and renders a navy-themed Next.js dashboard.

> **Build status:** see [`PROGRESS.md`](./PROGRESS.md) for the current phase.
> The full build spec is [`COPILOT_BUILD_PROMPT.md`](./COPILOT_BUILD_PROMPT.md);
> the agent operating rules are in [`.github/copilot-instructions.md`](./.github/copilot-instructions.md);
> the 3-minute walkthrough is [`docs/DEMO_SCRIPT.md`](./docs/DEMO_SCRIPT.md);
> the dashboard contract is [`docs/schema.md`](./docs/schema.md).

## TL;DR — run the demo

```bash
make install          # one-time: uv venv + Python deps + npm install
make demo             # full pipeline (~25 min) then serves static dashboard
                      # → http://localhost:3000
make demo-quick       # ~10 s — re-aggregate against existing outputs + serve
make help             # list every target
```

`make demo` runs the full pipeline over `./videos/` then static-exports
the dashboard and serves it via `python3 -m http.server`. **Zero live
inference** in the demo path — the dashboard reads only the
pre-rendered `analytics.json` and the static heatmaps / annotated
mp4s. Nothing in the room depends on a model running.

## Point CounterVision at a real client's footage

1. **Drop their mp4s into `videos/<camera-id>/`.** Filenames must start
   with `YYYYMMDDHHMMSSmmm` (the standard NVR export format) so the
   pipeline parses the real wall-clock for every event. Example:
   `videos/till-cam/20260612180530423.mp4`. Camera-folder names
   become the `camera_id` everywhere in the schema.
2. **Edit `pipeline/config.yaml`** so each camera entry has a
   human-meaningful `area` label (e.g. `"Till queue"`,
   `"Skincare aisle"`, `"Entrance"`). The dashboard renders these
   labels verbatim.
3. **Install + run the full pipeline.**
   ```bash
   make install
   make pipeline   # ~25 min for 3 cameras × 3-minute windows on M2 Pro
   ```
4. **Tune the entry line per scene.** The auto-default puts a
   horizontal line at 75 % of frame height. For honest footfall you
   want it where customers actually enter the camera's field of view:
   ```bash
   python pipeline/main.py --draw-zones till-cam
   ```
   Left-click polygon vertices, press `l` to switch to entry-line
   mode, click start and end of the line, press `s` to save back to
   `config.yaml`. Re-run `make zones` + `make aggregate` after.
5. **(Optional) Seed the watchlist** for the demo:
   ```bash
   python pipeline/main.py --seed-watchlist till-cam P002 known_staff
   ```
   Then re-run `make identity` + `make aggregate`. The dashboard's
   Alerts panel will show a non-accusatory verification prompt with
   the cosine score attached.
6. **Tunable thresholds** (in `config.yaml` under `identity:`):
   * `quality_min` (default `0.55`) — minimum `det_score` for a face to
     be embedded. Lower for low-light footage; higher to suppress
     jitter.
   * `cosine_match` (default `0.32`) — in-camera clustering cutoff.
     Lower merges more aggressively (over-clusters less); higher
     splits more (better precision per cluster).
   * `cross_camera_match` (default `0.50`) — deliberately *higher*
     than the clustering cutoff. False cross-camera merges hurt the
     headline number more than missed matches.
   * `sample_every_n_frames` (default `5`) — face-detection cadence.
     Lower = more faces seen, more CPU.

The pipeline never invents data. Anything it can't compute from the
footage (POS conversion, weather, quantified staffing) is emitted as a
**locked** KPI with an explicit "needs integration" reason; the
dashboard renders these as "Unlock with integration" pills rather than
fabricating a number.

## Phase 0 — dry-run on already-installed cameras

```bash
# 1. Create the env (Python 3.11; onnxruntime-silicon ceiling)
brew install uv ffmpeg
cd pipeline
uv venv --python 3.11
uv pip install -e ".[dev]"
# 2. Sanity-check: discover cameras, parse recording-start from filenames,
#    probe fps + frame size. No model inference.
uv run python main.py --dry-run

# 3. Phase 1 — run YOLO26 + BoT-SORT on MPS. Install heavy CV deps first.
#    The first run downloads yolo26s.pt (~20 MB). Default window is 180 s
#    per camera; override with --start-seconds / --duration-seconds, or
#    --full for the whole clip.
uv pip install -e ".[cv]"
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python main.py --run-detect-track                            # default 180 s window
uv run python main.py --run-detect-track --duration-seconds 30      # quick smoke
uv run python main.py --run-detect-track --start-seconds 60 --full  # everything after 60 s

# 4. Phase 2 — populate zones + entry lines (auto defaults, or interactive).
#    Then run analytics over the existing tracks JSONL (no model re-run).
uv run python main.py --draw-zones-default            # central 60% polygon + horizontal line @ 75%
# Optional: redraw zones interactively on a single camera (cv2 GUI)
# uv run python main.py --draw-zones camera-1
uv run python main.py --run-zones                     # writes zones/<cam>.json + heatmaps/<cam>.png

# 5. Phase 3 — face identity (InsightFace buffalo_l, CPU on M2).
#    First run downloads ~326 MB of buffalo_l weights to ~/.insightface.
uv pip install -e ".[cv,identity]"
uv run python main.py --run-identity
# Demo helper: copy a discovered person's full frame into ./watchlist/
# and re-run to see the watchlist alert flow fire end-to-end:
# uv run python main.py --seed-watchlist camera-5 P006 staff_lead_demo
# uv run python main.py --run-identity

# 6. Phase 4 — cross-camera identity (de-dup people across cameras).
#    Reads identity/<cam>.json from Phase 3 (no model run, no video
#    decode); writes data/output/cross_camera.json with the headline
#    store-wide unique-visitor count. Honest framing for non-overlapping
#    recordings: "same face seen across the captured period" (repeat
#    presence), not single continuous trips.
uv run python main.py --run-cross-camera

# 7. Phase 5 — aggregate everything into the dashboard contract.
#    Reads Phase 1-4 outputs and writes analytics.json + analytics.db.
#    Generates 3-5 plain-English insights from reliable numbers only;
#    locks anything uncomputable (POS / weather / staffing) so the
#    dashboard renders 'data not available' instead of a made-up number.
uv run python main.py --run-aggregate

# 9. Phase 6 — Next.js 16 dashboard (navy, client-ready).
#    Reads only data/output/analytics.json (no live inference).
cd dashboard && npm install        # one-time
npm run dev                        # auto-runs predev: copies data/output -> public/data
# open http://localhost:3000
npm run build                      # static prerender
cd ..

# 10. Tests + lint
cd pipeline
uv run ruff check ..
uv run pytest ../tests -q
```

The dry-run is what CI runs on every push (`.github/workflows/ci.yml`).

## Phase 6 — dashboard

The dashboard is a self-contained Next.js 16 app (App Router, React 19,
Tailwind v4, shadcn/ui, Recharts 3) under `dashboard/`. It reads
**only** `data/output/analytics.json` (plus the static heatmaps,
frames, mp4s, thumbnails the JSON points at). A
`predev`/`prebuild` hook (`dashboard/scripts/copy-data.cjs`) copies
the Phase 1–5 artefacts into `dashboard/public/data/` on demand.

```bash
cd dashboard
npm install
npm run dev   # http://localhost:3000
npm run build # static prerender, ready for `next export` in Phase 7
```

The page lays out 10 panels in spec order:

1. **Branded Overview** (navy hero with capture window).
2. **KPI cards** — reliable numbers up front; locked KPIs (POS /
   weather / staffing) render as "Unlock with integration" pills,
   never as errors or fake numbers.
3. **Per-area heatmap hero** — camera tabs; heatmap ↔ clean frame.
4. **Footfall by hour** (bar) — hedged with low-confidence pill.
5. **Dwell by area** (bar, avg + max) — high confidence.
6. **Per-area detail** — occupancy timeline + visitor thumbnails
   (the P006 "7 merged ids" badge is visible on camera-5).
7. **Cross-camera presence** — hedged; the `render_hint` from the
   schema is shown verbatim ("repeat presence, not a single
   continuous trip").
8. **Annotated walkthrough** — `<video>` plays the pre-rendered
   mp4 directly; no canvas overlay sync needed (overlays were burned
   in by Phase 1).
9. **Alerts feed** — non-accusatory review prompts with similarity.
10. **Plain-English insights** — 5 reliable retail recommendations.

Every metric carries an explicit `confidence` pill
(`high` / `medium` / `low` / `locked`) so the client sees at a glance
what's hard fact vs what to verify.

## Phase 5 outputs

`uv run python main.py --run-aggregate` reads every Phase 1–4 output
and writes two files (both gitignored):

- `data/output/analytics.json` — the **single file the dashboard
  reads.** Schema documented in [`docs/schema.md`](./docs/schema.md);
  versioned (`version: 1`). Carries reliable KPIs (high/medium
  confidence) alongside hedged KPIs (low confidence with explicit
  `note`/`method`) and **locked KPIs** (`{"value": null,
  "locked": true, "reason": "..."}`) for things like POS conversion
  and weather that this footage cannot compute.
- `data/output/analytics.db` — faithful sqlite mirror of the same
  data (tables: `areas`, `visitors`, `alerts`, `footfall_by_hour`,
  `occupancy_timeseries`, `cross_camera_links`, `insights`, `kpis`).

**`insights[]` generation rules:** every insight ties to a reliable
per-area number (dwell, occupancy, unique-faces). No insight is built
on a cross-camera link or near-zero footfall — see
`test_no_insight_built_on_cross_camera` for the asserted invariant.

## Phase 4 outputs

`uv run python main.py --run-cross-camera` reads each Phase-3
`identity/<camera>.json`, matches centroids across cameras at a
deliberately-higher threshold, and writes:

- `data/output/cross_camera.json` — schema:
  - `headline.store_wide_unique_visitors` (the dashboard's headline
    KPI), `naive_total_per_camera_sum`,
    `saved_by_cross_camera_dedup`, `cross_camera_links_count`,
    `no_reliable_cross_camera_matches`, `headline_message`.
  - `thresholds` block records the two distinct cosine cutoffs
    (`in_camera_cluster` for Phase 3 = 0.32,
    `cross_camera_match` for Phase 4 = 0.50) and the
    `min_face_appearances_for_cross_camera` gate.
  - `cross_camera_links[]` — only pairs above
    `cross_camera_match`, each carrying `similarity`, `time_gap`,
    and a `presence_note` framing the link as "**repeat presence
    across the captured period**, not a single continuous trip"
    (the camera windows do not overlap by hours).
  - `store_wide_persons[]` — connected components of the
    cross-camera match graph, numbered `S001…` by earliest
    `first_seen`.

**Honesty paths enforced in code:**

- If no pair clears the high threshold,
  `no_reliable_cross_camera_matches: true` fires,
  `store_wide_unique_visitors` falls back to the per-camera sum,
  and the CLI prints a ⚠ warning. We never invent a link.
- Persons with `face_appearances < 3` are excluded from the matching
  pool (centroid too noisy) but **kept in the headline count** so
  they aren't disappeared from totals.

## Phase 3 outputs

`uv run python main.py --run-identity` runs InsightFace `buffalo_l`
(SCRFD + ArcFace, CPU on M2) over the same window as Phase 1, sampling
every `identity.sample_every_n_frames` frames. Writes, per camera:

- `data/output/identity/<camera>.json` —
  `unique_visitors_count` (AUTHORITATIVE),
  `unique_visitors_locked: false` (UNLOCKING Phase 2's sentinel),
  `persons[]` with `linked_tracker_ids` + authoritative
  `track_dwell_seconds_authoritative` (union of frames where any
  linked tracker ID is alive — replaces Phase 2's provisional
  per-track dwell), `face_dwell_seconds_authoritative`, `visit_count`,
  `is_repeat`, `watchlist_match`, `watchlist_similarity`.
- `data/output/persons/<camera>/<Pxxx>.jpg` (thumbnail) +
  `<Pxxx>_full.jpg` (full frame at the best-quality moment, used by
  `--seed-watchlist` for redetectable seeds).
- `data/output/alerts/<id>.jpg` — full-frame screenshot per alert.
- `data/output/phase3_summary.json` — orchestrator-level roll-up
  with the tuned thresholds.

**Alert design.** Every event is a non-accusatory review prompt:

- Watchlist: *"Possible match with watchlist entry 'X' (face
  similarity 0.XX). Please verify before acting."*
- Repeat:   *"Possibly a returning visitor — face seen across N
  separate visits in this window. Please verify."*

Severity is set from cosine similarity for watchlist hits
(`info < 0.45`, `warn 0.45–0.60`, `high ≥ 0.60`) so the dashboard
can prioritise without dropping any signal.

**Watchlist seeding for the demo.**
`uv run python main.py --seed-watchlist CAM PID [LABEL]` copies the
person's full-frame jpg into `./watchlist/`. Re-run `--run-identity`
to see the alert fire — useful for showing the flow without supplying
real reference photos. Watchlist images are gitignored (faces = PII).

## Phase 2 outputs

`uv run python main.py --run-zones` (after `--draw-zones-default` or a
manual `--draw-zones CAMERA`) writes, per camera:

- `data/output/zones/<camera>.json` — versioned schema:
  `area`, `footfall.{in_count, out_count, events}`, `footfall_by_hour`,
  `zones[].{polygon, occupancy_peak, dwell_seconds_by_track_provisional,
  avg_dwell_seconds_provisional, provisional_note}`,
  `occupancy_timeseries[]`, `person_tracks.{count, note}`,
  `videos_considered`, `videos_skipped`,
  **`unique_visitors_locked: true`** and `unique_visitors_note`.
- `data/output/heatmaps/<camera>.png` — gaussian-blurred density of box
  bottom-center points across the window, composited over the
  Phase-1 first-frame jpg with alpha blending. Empty heat → black
  PNG (we don't lie with JET's deep-blue default).
- `data/output/phase2_summary.json` — orchestrator-level roll-up.

**Guardrails enforced in code + JSON:**

- No field is called `unique_visitors`. Tracker IDs are not visitor
  identities and **Phase 2 will not emit a unique-visitor count.** That
  comes from Phase 3 face linking.
- Per-zone dwell uses the `*_provisional` suffix + a
  `provisional_note` because one person can fragment across multiple
  tracker IDs during occlusion. Phase 3 face linking replaces these.
- Multiple videos per camera are explicitly tracked
  (`videos_considered` / `videos_skipped`) so it is obvious when a
  continuation file hasn't been processed yet.

## Phase 1 outputs

`uv run python main.py --run-detect-track` writes, per camera:

- `data/output/annotated/<camera>.mp4` — 1080p, 25 fps, boxes + IDs +
  short traces colored by `tracker_id` so visual ID stability is
  obvious. Each frame also carries a navy HUD with the camera label,
  area, real wall-clock time and live person count.
- `data/output/frames/<camera>.jpg` — the clean first decoded frame of
  the processing window (used as the heatmap backdrop in Phase 2).
- `data/output/tracks/<camera>.jsonl` — one record per detection
  (`camera_id`, `frame_idx`, `wall_clock`, `tracker_id`, `xyxy`,
  `conf`). Phase 2+ consume this so they never re-run the model.
- `data/output/phase1_summary.json` — orchestrator summary: per-camera
  unique IDs, ID-switch proxy count, fps and elapsed time.

**ID-switch metric is a proxy**, not MOTA. We have no ground truth, so
"ID switch" here means: a brand-new `tracker_id` whose bounding box
overlaps (IoU ≥ `detect.id_switch_iou`) a recently-disappeared
`tracker_id` within `detect.id_switch_lookback_frames`. Useful as a
churn metric; not directly comparable to published numbers.

## Repo layout

```
COUNTERVISION/
├── PROGRESS.md                 # phase-by-phase log
├── COPILOT_BUILD_PROMPT.md     # build spec
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/ci.yml
├── pipeline/                   # Python (uv-managed)
│   ├── pyproject.toml
│   ├── config.yaml             # cameras, zones, processing window, detect/identity thresholds
│   └── countervision/
│       ├── __init__.py
│       ├── botsort_demo.yaml   # BoT-SORT overrides (track_buffer, new_track_thresh)
│       ├── detect_track.py     # Phase 1: YOLO26 + BoT-SORT + IdSwitchCounter
│       ├── discover.py         # camera + video discovery, probing, processing_window
│       ├── logging_setup.py
│       ├── main.py             # entrypoint (--dry-run, --run-detect-track)
│       └── timeparse.py        # YYYYMMDDHHMMSSmmm → datetime
├── tests/                      # pytest
├── data/output/                # pipeline artifacts (gitignored)
│   ├── annotated/<cam>.mp4
│   ├── frames/<cam>.jpg
│   ├── tracks/<cam>.jsonl
│   └── phase1_summary.json
├── videos/                     # input footage (one folder per camera)
└── watchlist/                  # reference face JPGs (gitignored)
```

## Hardware / platform notes

This project targets **Apple Silicon (M2 Pro)** with PyTorch MPS + CoreML.
Never CUDA. From Phase 1 onward: `device="mps"` on Ultralytics calls and
`export PYTORCH_ENABLE_MPS_FALLBACK=1` so unimplemented ops fall back to CPU.

## Licensing

Demo only. For commercial shipping the YOLO26 (AGPL-3.0) and InsightFace
pretrained weights (non-commercial) must be swapped — see §10 of
`COPILOT_BUILD_PROMPT.md`.
