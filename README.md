# CounterVision

Offline multi-camera CCTV → retail-analytics demo for a client meeting.
Product of **Agents Limited**. Processes recorded `.mp4` / `.mov` files in
`./videos/` with a Python CV pipeline (YOLO26 + supervision + InsightFace on
Apple Silicon MPS) and renders a navy-themed Next.js dashboard.

> **Build status:** see [`PROGRESS.md`](./PROGRESS.md) for the current phase.
> The full build spec is [`COPILOT_BUILD_PROMPT.md`](./COPILOT_BUILD_PROMPT.md);
> the agent operating rules are in [`.github/copilot-instructions.md`](./.github/copilot-instructions.md).

## Quick start (Phase 0 only — dry-run)

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

# 7. Tests + lint
uv run ruff check ..
uv run pytest ../tests -q
```

The dry-run is what CI runs on every push (`.github/workflows/ci.yml`).

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
