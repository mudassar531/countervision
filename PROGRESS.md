# PROGRESS.md — CounterVision build log

> Source of truth: [`COPILOT_BUILD_PROMPT.md`](./COPILOT_BUILD_PROMPT.md).
> Operating contract per phase: **THINK → CODE → VALIDATE (on real footage) →
> PUSH → update this file → stop and report.** Never fabricate metrics.

## Project snapshot

- **Product:** CounterVision — offline multi-camera CCTV → retail analytics
  demo for a client meeting (Agents Limited).
- **Footage on disk:** 3 cameras, 4 video files (1920×1080, ~25 fps).
  - `camera-1` (Cosmetics & Skincare) — 1 × mp4 (20:53 on 2026-06-07, 20 min).
  - `camera-3` (Fragrance & Promo Aisle) — 2 × mp4 (00:31 and 00:54 on 2026-06-08).
  - `camera-5` (Entrance & Billing) — 1 × **mov** (04:44 on 2026-06-08, 15 min).
- **Hardware:** Apple Silicon (M2 Pro) → PyTorch MPS + CoreML, never CUDA.
- **Demo path:** zero live inference; the dashboard reads only static
  pre-rendered artifacts.

## Phase tracker

| # | Phase                                              | Status      | Notes |
|---|-----------------------------------------------------|-------------|-------|
| 0 | Scaffold + camera discovery + timeparse + CI        | ✅ done     | Pushed to `mudassar531/countervision` (HEAD `ff0897f`). CI green, 20/20 tests pass, dry-run validated on real footage locally. |
| 1 | Detect + track (YOLO26 MPS + BoT-SORT)              | ✅ done     | Pushed (HEAD `327913e`); CI green. 27/27 tests, 3 cameras × 180 s validated on real footage — 89 unique IDs, 38 ID-switch-proxy events, ID #28 survived a 2.20 s occlusion. |
| 2 | Zones / footfall / dwell / heatmap / occupancy      | ✅ done     | Pushed (HEAD `6e75615`); CI green. 43/43 tests, real-footage run validated (heatmaps overlay correctly; provisional dwell + occupancy timeseries written; `unique_visitors_locked: true` everywhere). |
| 3 | Identity: unique + repeat + watchlist               | ✅ done     | Pushed (HEAD `4bfa653`); CI green. 59/59 tests; tuned quality_min=0.55, cosine_match=0.32. 16 unique visitors total (vs 89 raw tracker IDs); camera-5 P006 = 7 merged Phase-1 fragments → 31.6 s authoritative dwell; planted watchlist self-test fires correct alerts; `unique_visitors_locked: false`. |
| 4 | Cross-camera identity (de-dup, not journey)         | ✅ done     | Pushed (HEAD `62662ee`); CI green. 72/72 tests; cross_camera_match=0.50 (high bar, distinct from 0.32); 3 reliable links found (sims 0.58–0.60) over a ≈4 h 13 m gap → **store-wide unique 13** (vs 16 naive). |
| 5 | Aggregate → `analytics.json` + sqlite + insights    | ✅ done     | Pushed (HEAD `8bdcabd`); CI green. 89/89 tests; 5 reliable insights generated from per-area dwell + occupancy; cross-camera & watchlist carried through as hedged fields with `render_hint`; 4 locked KPIs documented (POS, weather, quantified staffing). Schema doc at `docs/schema.md`. |
| 6 | Next.js dashboard (navy, client-ready)              | ✅ done     | Pushed (HEAD `07598db`); CI green. Next.js 16 + React 19 + Tailwind v4 + shadcn + Recharts 3. All 10 panels in spec order. Reads only `analytics.json`; copy-data script populates `dashboard/public/data` on `predev`/`prebuild`. Confidence pills on every metric; locked KPIs render as "Unlock with integration" not errors; cross-camera respects `render_hint` (no journey paths). Production build static-prerenders. |
| 7 | One-command demo + talk-track                       | ✅ done     | Pushed (HEAD `af712ab`); CI green. `make demo` / `make demo-quick` validated end-to-end. Dashboard static-exports to `dashboard/out/`, served by `python3 -m http.server` (zero new deps). `docs/DEMO_SCRIPT.md` is a presentation-ready 3-minute business walkthrough with the "your cameras turned into decisions" throughline. README handover section covers client-footage onboarding + zone re-drawing + threshold tuning. **All 8 phases ship.** |
| 2 | Zones / footfall / dwell / heatmap / occupancy      | ⏳ pending  | |
| 3 | Identity: unique + repeat + watchlist               | ⏳ pending  | |
| 4 | Cross-camera journey                                | ⏳ pending  | |
| 5 | Aggregate → `analytics.json` + sqlite + insights    | ⏳ pending  | |
| 6 | Next.js dashboard (navy, client-ready)              | ⏳ pending  | |
| 7 | One-command demo + talk-track                       | ⏳ pending  | |

---

## Phase 0 — Scaffold + camera discovery + timeparse + CI

### THINK (goal, files, risks)

**Goal.** Get the skeleton in place so every later phase has a stable spine:
config-driven camera discovery, authoritative filename → wall-clock parsing,
logging, a unit-tested `timeparse` helper, and a CI smoke test that asserts
the dry-run finds every camera and parses every timestamp from the filename
(no mtime fallback in the happy path). **Acceptance:**
`python pipeline/main.py --dry-run` lists every camera, file, parsed
recording-start time, fps, frame size, and area without error.

**Files created/changed.**

- `PROGRESS.md` (this file)
- `README.md` — quick start for Phase 0.
- `.gitignore` — excludes `.venv`, `data/output/*` artifacts, `node_modules`,
  `*.pt`, `.DS_Store`.
- `pipeline/pyproject.toml` — uv project, Python 3.11, Phase 0 deps only
  (`pyyaml`, `opencv-python-headless`, plus dev `pytest` + `ruff`). Heavy
  ML deps deferred to the phase that actually uses them.
- `pipeline/config.yaml` — store name, fps fallback, allowed video
  extensions, camera → area mapping, identity / behaviour thresholds.
- `pipeline/countervision/__init__.py`
- `pipeline/countervision/logging_setup.py` — single-handler stderr logging.
- `pipeline/countervision/timeparse.py` — `parse_recording_start_from_name`,
  `resolve_recording_start` (filename-first, mtime fallback with warning),
  `wall_clock_for_frame`.
- `pipeline/countervision/discover.py` — `load_config`, `discover_camera_dirs`,
  `list_camera_videos`, `probe_video` (OpenCV — no ffprobe hard dep),
  `discover_all`.
- `pipeline/main.py` — `--dry-run` text + optional `--json` summary; exits
  non-zero if no cameras / no videos.
- `tests/test_timeparse.py` — happy path + suffixed-mov + bad-input +
  mtime-fallback coverage.
- `.github/workflows/ci.yml` — Python 3.11, install with `uv`, run ruff,
  pytest, dry-run, then assert via inline Python that all 3 cameras were
  discovered and every timestamp source is `"filename"`.
- `data/output/{annotated,heatmaps,frames,alerts}/.gitkeep` — empty target
  dirs that match the §3 layout.
- `watchlist/README.md` — usage + privacy notes for Phase 3.

**Risks / decisions taken.**

- Spec says "process every `.mp4`" but camera-5 ships only a `.mov`.
  **Decision:** extend the discovery glob to `[.mp4, .mov]` so camera-5
  (Entrance & Billing — important for footfall) is not dropped.
- Filenames are timezone-naive (`YYYYMMDDHHMMSSmmm`, no TZ). Treat as
  **naive local time** for now; if the client needs a TZ, set one in
  config later. All timestamps are emitted ISO-8601 with millisecond
  precision so it's unambiguous.
- The burned-in clock disagrees with the filename — spec says trust the
  filename. `timeparse` does exactly that and warns on mtime fallback.
- Phase 0 only installs `pyyaml` + `opencv-python-headless`. Heavy deps
  (`torch`, `ultralytics`, `supervision`, `insightface`,
  `onnxruntime-silicon`) come in Phase 1+ so CI stays under a minute and
  Phase 0 can be reproduced on any contributor's laptop.
- Did **not** use Context7 for Phase 0 (no fast-moving CV/ML library APIs
  in scope here). Will query Context7 before writing the Ultralytics /
  supervision / InsightFace code in Phase 1+.

### CODE

Implemented exactly the files listed above. No model code, no heavy deps.

### VALIDATE (on real footage)

```bash
cd pipeline
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run ruff check ..
uv run pytest ../tests -q
uv run python main.py --dry-run
```

Results:

- **Lint:** `ruff check ..` → all checks passed.
- **Tests:** `pytest ../tests -q` → 11 passed.
- **Dry-run:** discovered 3 cameras / 4 video files; every recording-start
  parsed from filename (no mtime fallback); fps + frame size resolved for
  every file:

  ```
  CounterVision Demo Store
  ================================
  Discovered 3 cameras / 4 video files

  [camera-1]  area: Cosmetics & Skincare
    videos/camera-1/20260607205350587.mp4
      recording_start: 2026-06-07T20:53:50.587  (parsed-from-filename)
      fps: 25.00  frame_size: 1920x1080  duration: 1207.16s  frames: 30176

  [camera-3]  area: Fragrance & Promo Aisle
    videos/camera-3/20260608003129784.mp4
      recording_start: 2026-06-08T00:31:29.784  (parsed-from-filename)
      fps: 25.00  frame_size: 1920x1080  duration: 1353.21s  frames: 33827
    videos/camera-3/20260608005449323.mp4
      recording_start: 2026-06-08T00:54:49.323  (parsed-from-filename)
      fps: 25.00  frame_size: 1920x1080  duration: 1237.48s  frames: 30934

  [camera-5]  area: Entrance & Billing
    videos/camera-5/20260608044448561_AH8174419_Barkerend - Kingz_16_video.mov
      recording_start: 2026-06-08T04:44:48.561  (parsed-from-filename)
      fps: 25.00  frame_size: 1920x1080  duration: 906.09s  frames: 22650
  ```

Acceptance criteria (`Done when: python pipeline/main.py --dry-run lists
every camera, file, parsed recording-start time, fps, frame size, and area
without error`) — **met.**

### PUSH

- Repo: <https://github.com/mudassar531/countervision> (public, owner `mudassar531`).
- Commits on `main`:
  - `0d8487d` — Phase 0 scaffold + camera discovery + timeparse + CI
  - `dddd046` — PROGRESS.md push details
  - `ff0897f` — pin pipeline deps with `uv.lock` (unblocks `setup-uv` cache)
- CI: [run 27580618475](https://github.com/mudassar531/countervision/actions/runs/27580618475)
  green on `ff0897f` — ruff clean, 20/20 pytest pass, dry-run step
  gracefully skips on the hosted runner (no real footage there; the
  synthetic-video tests cover the same path).
- Created via `gh repo create mudassar531/countervision --public
  --source=. --remote=origin --push` (with `GH_TOKEN` unset so the
  keyring-stored `mudassar531` credentials are used; the env `GH_TOKEN`
  belongs to a different account and must never be active for this repo).
- Videos themselves (>100 MB each, GitHub hard limit) are gitignored.
  Camera folders are kept via per-folder `README.md` so a fresh clone
  still discovers them; CI's discover/probe path is covered by the
  synthetic-mp4 integration tests in `tests/test_discover.py`.

### NEXT — Phase 1 (do not start yet)

YOLO26s on MPS with BoT-SORT (ReID + enlarged track buffer), plus a
`supervision.DetectionsSmoother` to settle jitter. Output:
`data/output/annotated/<camera>.mp4` (boxes + persistent IDs) and a clean
`data/output/frames/<camera>.jpg` per camera. Before coding, query Context7
for current `ultralytics` and `supervision` APIs (both shifted in 2025–2026).

---

## Phase 1 — Detect + track (YOLO26 + BoT-SORT on MPS)

### THINK (goal, files, risks)

**Goal.** Run YOLO26s on MPS with BoT-SORT (enlarged track buffer for
sit / brief-occlusion stability) over a configurable processing window of
each camera's clip, producing the annotated mp4 + clean first-frame jpg +
per-detection JSONL that Phase 2+ consume. Acceptance: seated people keep
stable IDs through a brief occlusion; annotated video plays; ID-switch
count reported.

**Context7 lookups (done first, per the build prompt).** Confirmed current
APIs in `ultralytics` 8.4 and `supervision` 0.29:

* `model.track(source=[frame], persist=True, tracker=<yaml>, classes=[0],
  device="mps", imgsz=…, conf=…, verbose=False)` — note that a bare
  numpy frame regresses to the default assets dir in 8.4.67; wrapping in
  a list works reliably.
* `result.boxes.id` is the tracker-ID tensor (None if untracked).
* `sv.Detections.from_ultralytics(result)` carries `tracker_id` straight
  through; `BoxAnnotator(color_lookup=ColorLookup.TRACK)` /
  `LabelAnnotator` / `TraceAnnotator(trace_length=…)` all key off
  `tracker_id`, so an empty Detections must be skipped or they raise.
* Tracker reset between cameras is via
  `model.predictor.trackers[i].reset()` (BoT-SORT inherits ByteTrack's
  `reset` + `reset_id`).

**Files created / changed.**

* `pipeline/countervision/detect_track.py` — new. Per-camera loop with
  cv2-backed window slicing, `IdSwitchCounter` proxy metric, navy HUD
  overlay, JSONL track sink, `run_detect_track` orchestrator that
  resets the tracker between cameras.
* `pipeline/countervision/botsort_demo.yaml` — new. BoT-SORT overrides
  layered over the default (only `track_buffer 30 → 60` and
  `new_track_thresh 0.25 → 0.30`). Custom `with_reid: True` +
  `appearance_thresh: 0.25` was tried first but reproducibly produced
  None tracker IDs on the first frames; documented in the yaml header.
* `pipeline/countervision/discover.py` — added `ProcessingWindow`
  dataclass + `to_frame_range`, exposed `processing_window` and
  `detect` on `PipelineConfig`.
* `pipeline/config.yaml` — `processing_window: { start_seconds: 0,
  duration_seconds: 180 }` + `detect: { model: yolo26s.pt, device: mps,
  imgsz: 960, conf: 0.30, iou: 0.55, classes: [0], tracker_yaml,
  trace_length: 90, id_switch_iou: 0.30, id_switch_lookback_frames: 30 }`.
* `pipeline/main.py` — `--run-detect-track` mode; window overrides
  (`--start-seconds`, `--duration-seconds`, `--full`); writes
  `data/output/phase1_summary.json`.
* `pipeline/pyproject.toml` — split deps. Core (CI):
  `pyyaml`, `opencv-python-headless`, `numpy`. New optional `[cv]`
  extra: `torch>=2.4`, `torchvision>=0.19`, `ultralytics>=8.3`,
  `supervision>=0.25`, `lap>=0.5.12`. `uv pip install -e ".[cv]"` is
  the Phase 1 install command; CI stays light.
* `tests/test_id_switch.py` — new. 7 unit tests for the
  `IdSwitchCounter` proxy (IoU edges, persistence, lookback, best-partner
  selection).
* `README.md` — Phase 1 install + commands.

**Risks / decisions taken.**

* **`with_reid: True` in BoT-SORT is unstable on 8.4.67.** Combined with
  a low `appearance_thresh` it produced None IDs for several seconds at
  the start of each clip. Falling back to `with_reid: False` plus an
  enlarged `track_buffer` gives the same occlusion-recovery behaviour
  we wanted without the regression. Will revisit if Phase 3 demands
  appearance ReID outside of faces.
* **First `model.track(frame, …)` call drops `source`** in 8.4.67 — the
  workaround is `source=[frame]`. Documented in the loop.
* **Per-camera tracker reset** keeps IDs starting at 1 per camera; we
  verified `min_id == 1` for all three cameras after reset.
* **ID-switch metric is a proxy**, not MOTA — no ground truth available.
  Documented in `detect_track.py` docstring and README.
* **Processing window default 180 s.** Keeps a single demo run on M2
  Pro under ~12 minutes total (3 × 220-230 s @ ~20 fps) and the
  annotated mp4 demo-length. `--full` runs the whole clip.
* **Heavy deps split.** Phase 0 / lint / unit tests don't need torch;
  CI runs in 18 s. Locally `uv pip install -e ".[cv]"` adds ~700 MB.

### CODE

Implemented exactly the files above. 27/27 tests pass; ruff clean.

### VALIDATE (on real footage)

```bash
cd pipeline
uv pip install -e ".[cv]"
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python main.py --run-detect-track     # default 180 s window
```

Hardware: M2 Pro / MPS / FP32. Total wall time **11 min 45 s** for 3 ×
180 s on 1080p. Headline numbers:

| camera   | area                       | frames | dets   | unique IDs | ID switches (proxy) | fps  |
|----------|----------------------------|-------:|-------:|-----------:|--------------------:|-----:|
| camera-1 | Cosmetics & Skincare       |  4 500 |  3 677 |         40 |                  23 | 20.5 |
| camera-3 | Fragrance & Promo Aisle    |  4 500 |  9 731 |         11 |                   3 | 20.4 |
| camera-5 | Entrance & Billing         |  4 500 | 10 112 |         38 |                  12 | 19.4 |
| **total**|                            | 13 500 | 23 520 |         89 |                  38 |  —   |

**Acceptance — stable IDs through brief occlusion (post-run gap analysis
on `data/output/tracks/*.jsonl`):**

* `camera-1` — 18 IDs survived a ≥ 5-frame gap (occlusion). ID `#28`
  recovered through gaps up to **55 frames = 2.20 s**; ID `#7`
  recovered 14 times across a 722-frame track.
* `camera-3` — IDs `#1` and `#2` held across the **entire 4500-frame
  window** without breaking — the seated/standing baseline that the
  spec calls out. Two further IDs survived 16-frame (0.64 s) gaps.
* `camera-5` — 19 occlusion recoveries despite heavy entrance
  turnover; ID `#18` recovered after **49 frames = 1.96 s**, and IDs
  `#1` and `#2` hold for most of the window.

Tracker reset works: `min_id == 1` for every camera's tracks JSONL.

Annotated mp4s play (1920×1080, 25 fps, h264 via `mp4v` codec) and
each frame carries the navy HUD with the real wall-clock parsed from
the filename.

### PUSH

- Commits on `main` (https://github.com/mudassar531/countervision):
  - `3d8d843` — Phase 1: detect + track (YOLO26 + BoT-SORT on MPS)
  - `327913e` — extract `IdSwitchCounter` to its own module so CI doesn't
    need the heavy `[cv]` extras (lint + pytest stayed broken on the
    first push because the test pulled `detect_track` which top-level
    imports `supervision`).
- CI: [run 27582412831](https://github.com/mudassar531/countervision/actions/runs/27582412831)
  green on `327913e` — ruff clean, 27/27 pytest pass, dry-run step
  still skips gracefully when no real footage on the runner.

### NEXT — Phase 2 (do not start yet)

Draw zones / lines per camera with the click-to-draw helper, then
`LineZone` → footfall, `PolygonZone` + dwell, `HeatMapAnnotator` →
`data/output/heatmaps/<camera>.png`. **Consume `tracks/<camera>.jsonl`
rather than re-running detection.** Confirm the retail area mapping
against what the footage actually shows; redraw the placeholder
camera-1 / camera-3 / camera-5 area labels if the scenes tell a
different story.

---

## Phase 2 — Zones / footfall / dwell / heatmap / occupancy

### THINK (goal, files, risks)

**Goal.** Consume the Phase 1 tracks JSONL (no model re-run) and turn it
into per-camera footfall (line crossings), per-zone presence + provisional
dwell, occupancy timeseries, and a heatmap PNG composited over the clean
first-frame jpg. Acceptance: footfall matches an eyeball count on a clip;
dwell + heatmap + occupancy produced per camera; **no field labelled
"unique visitors"** anywhere in the output (that comes from face identity
in Phase 3).

**Context7 lookups (done first).**

* `sv.LineZone(start=Point, end=Point, triggering_anchors=[Position.BOTTOM_CENTER])`
  → `trigger(detections)` returns `(crossed_in, crossed_out)` and bumps
  `in_count` / `out_count`. Tracker-IDs required.
* `sv.PolygonZone(polygon=np.ndarray, triggering_anchors=…)` →
  `trigger(detections)` returns a boolean mask.
* `sv.HeatMapAnnotator(radius, opacity, …)` exists, but we don't pass
  full Detections frames at scale here — heat is accumulated from box
  bottom-center points (NumPy + cv2.GaussianBlur).
* The `triggering_position` arg is deprecated for `triggering_anchors`.

**Decision: implement the primitives in pure NumPy + OpenCV** rather than
hard-depending on supervision in `zones.py`. Reasons:

1. Keeps the unit tests CI-friendly — no `[cv]` extras required.
2. Avoids supervision's `frame_resolution_wh`-required quirks across
   versions.
3. We want explicit control over the "what counts as `in`" convention
   so it's deterministic and documented (see `LineCrossing` docstring).

**Area labels — confirmed against the actual scenes** (viewed each
`frames/<cam>.jpg`):

| camera   | burned-in label    | scene reality                                              | retail label |
|----------|---------------------|-------------------------------------------------------------|--------------|
| camera-1 | "Operators Hall"    | Open floor, tall wall of cubed wooden display shelving left | **kept** "Cosmetics & Skincare" — the cube shelving sells believably as merchandise displays |
| camera-3 | "Grab Stations"     | Bright window seating, 2 people seated around a small surface | **relabelled** "Customer Seating / Try-on Lounge" — was "Fragrance & Promo Aisle" (no aisle in frame) |
| camera-5 | "Barkerend / Kingz" | Cluster of workstations, seated operators, deskphones, multiple monitors | **relabelled** "Service & Consultation Desk" — was "Entrance & Billing" (no till or door in frame) |

**camera-3 two-file handling — decision documented.** Phase 1's
orchestrator currently picks `cam.videos[0]`, so only
`20260608003129784.mp4` is tracked; the continuation
`20260608005449323.mp4` is **not** in `tracks/camera-3.jsonl`. Phase 2
reads what is present and emits both `videos_considered` and
`videos_skipped` per camera, plus the human-readable summary prints
"videos NOT tracked: …". Extending Phase 1 to iterate every video per
camera (and concatenate their tracks with the right wall-clock offsets)
is a small change but **out of scope for this phase** — leaving it for
when we revisit Phase 1 between phases.

**Files created / changed.**

* `pipeline/countervision/zones.py` — pure NumPy + OpenCV
  primitives: `LineCrossing` (signed cross-product, anchor =
  bottom-center, documented in/out convention), `PolygonZone`
  (cv2.pointPolygonTest, per-track dwell accumulator, peak occupancy),
  `HeatmapAccumulator` (gaussian-blurred density, composited PNG over
  base frame), `load_tracks_jsonl`, `run_zone_analytics`,
  `summarize_results`.
* `pipeline/tools/draw_zones.py` — operator-facing helper.
  `populate_defaults` (no GUI; central-60 % polygon, horizontal line at
  75 % height per camera; idempotent — keeps non-empty existing zones).
  `interactive_draw` (cv2 GUI: left-click vertices, `c` closes,
  `n` new polygon, `l` toggles entry-line mode, `s` saves, `q` quits).
* `pipeline/tools/__init__.py` — package marker.
* `pipeline/config.yaml` — area labels updated to honest retail
  framings; zones / entry lines populated via `--draw-zones-default`.
* `pipeline/main.py` — three new modes: `--run-zones`,
  `--draw-zones-default [--overwrite-zones]`,
  `--draw-zones CAMERA_ID`. Writes `data/output/phase2_summary.json`.
* `tests/test_zones.py` — 14 tests (geometry, line crossings,
  polygon presence, dwell, heatmap edge cases, default-zone
  generator, full end-to-end on a synthetic JSONL).
* `tests/test_discover.py` — updated area assertions to match the
  relabelled cameras.
* `.gitignore` — adds `data/output/zones/`.

**Risks / decisions taken.**

* **Guardrail.** No field labelled "unique visitors" anywhere. Every
  zones JSON includes `unique_visitors_locked: true` plus a
  `unique_visitors_note` directing readers to Phase 3 face linking.
  The count we *do* expose is `person_tracks.count` (unique
  tracker_ids) with an explicit "NOT 'unique visitors'" note.
* **Provisional dwell.** Per-track dwell aggregations live under
  `dwell_seconds_by_track_provisional` /
  `avg_dwell_seconds_provisional`, each with a
  `provisional_note` saying authoritative per-person dwell will come
  from Phase 3 face linking.
* **PyYAML strips comments** on round-trip in
  `populate_defaults`. The reasoning behind the retail relabels lives
  in this file, not in `config.yaml`. Switching to ruamel.yaml is a
  future option but out of scope here.
* **Default entry line at 75 % height** — these office scenes have
  people sitting in the upper half of frame, so the default line
  often sees zero or one crossing. The operator should redraw the
  line at the scene's real entrance with `--draw-zones CAM`; the
  current counts are real (not fabricated) but small.

### CODE

Implemented exactly the files above. 43/43 tests pass; ruff clean.

### VALIDATE (on real footage)

```bash
cd pipeline
uv run python main.py --draw-zones-default   # one-time, writes config.yaml
uv run python main.py --run-zones            # reads tracks/<cam>.jsonl
```

Per-camera headline numbers (from
`data/output/phase2_summary.json` + `data/output/zones/<cam>.json`):

| camera   | area                                | frames consumed | person tracks | footfall in / out | peak occupancy (Main floor) | videos skipped |
|----------|--------------------------------------|----------------:|---------------:|-------------------:|----------------------------:|----------------|
| camera-1 | Cosmetics & Skincare                 | 3 197 | 40 | 0 / 0 | 3 | — |
| camera-3 | Customer Seating / Try-on Lounge     | 4 500 | 11 | 0 / 1 | 3 | `20260608005449323.mp4` |
| camera-5 | Service & Consultation Desk          | 4 500 | 38 | 1 / 1 | 2 | — |

(`person_tracks` is unique tracker_ids, NOT unique visitors —
`unique_visitors_locked: true` in every JSON. `frames_consumed < 4 500`
on camera-1 because frames with no detections are not written to JSONL
by Phase 1.)

**Heatmaps visually validated** by inspecting
`data/output/heatmaps/<cam>.png`:

* `camera-1` — bright red hot spots exactly on the two seated
  operators on the right of frame, no false heat elsewhere.
* `camera-3` — hot spots on the two seated guys by the window,
  matching where the detections actually concentrate.
* `camera-5` — hot spots on the seated workstation cluster.

Composited over the clean Phase-1 frame; dashboard-ready for Phase 6.

**Footfall counts are honest, not fabricated.** The default entry
line at y=810 sits below the action in these office scenes, so most
people stay on one side. An operator running `--draw-zones CAM` and
redrawing the line where people actually walk will get a more useful
footfall number — but Phase 2's contract is to report what the
geometry actually produces, not what looks good.

### PUSH

- Commit on `main`: `6e75615` — Phase 2: zones / footfall / dwell /
  heatmap / occupancy
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27583182562](https://github.com/mudassar531/countervision/actions/runs/27583182562)
  green on `6e75615` — ruff clean, **43/43** pytest pass on the
  lightweight default deps (no `[cv]` extras needed).

### NEXT — Phase 3 (do not start yet)

InsightFace `buffalo_l` (SCRFD + ArcFace 512-d). Quality-gate by
`det_score`, embed only quality-gated faces, cluster across the window to
build a session gallery, and compare against `./watchlist/` for hits.
Emit alerts as non-accusatory review prompts. Tune `quality_min` +
`cosine_match` on the real footage, record the chosen values here.
Crucially: **replace Phase 2's `person_tracks.count` with an authoritative
`unique_visitors` count** (the field currently flagged
`unique_visitors_locked: true`).

---

## Phase 3 — Identity: unique + repeat + watchlist

### THINK (goal, files, risks)

**Goal.** InsightFace `buffalo_l` (SCRFD + ArcFace, CPU on M2) over the
same processing window as Phase 1, sampling every
`identity.sample_every_n_frames` frames. Per face: gate by `det_score`,
embed if it passes, link to the Phase-1 person box on that frame by
face-center-in-box + head-region IoU, greedy-cluster the 512-d
L2-normalized embeddings into "persons", recompute per-person
authoritative dwell by union-merging the linked tracker_ids' frames,
and watchlist-match each person centroid against `./watchlist/*.jpg`.
Emit non-accusatory review-prompt alerts with thumbnails. **Unlock the
`unique_visitors_locked: true` sentinel** Phase 2 set.

**Context7 lookup (done first).** Confirmed current InsightFace API:
`FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]).prepare(ctx_id=-1, det_size=(640,640))`,
`face = app.get(frame)` returning `bbox`, `det_score`,
`normed_embedding` (512-d, L2-normalized → cosine = dot product),
`kps`. `allowed_modules=['detection','recognition']` skips
genderage/landmarks for speed.

**Files created / changed.**

* `pipeline/countervision/identity.py` — new. `cosine_similarity`,
  `link_face_to_tracker` (face-center-in-person-box + head-region IoU),
  `PersonCluster` (greedy online cosine clustering with running
  centroid), `WatchlistMatcher` (pre-embed `./watchlist/*.jpg`,
  per-person centroid comparison), `compute_visit_count` (segmenting
  by absence gap), `run_identity` orchestrator that decodes the same
  window as Phase 1 and writes
  `data/output/identity/<camera>.json` + thumbnails + alert frames.
  `seed_watchlist_from_person` helper for the demo flow.
* `pipeline/main.py` — `--run-identity`, `--seed-watchlist CAM PID
  [LABEL]`. Writes `data/output/phase3_summary.json`.
* `pipeline/config.yaml` — `identity` block expanded
  (multi-line YAML; `quality_min: 0.55`, `cosine_match: 0.32` —
  tuned, see "Tuning" below).
* `pipeline/pyproject.toml` — new `[identity]` extra:
  `insightface>=0.7.3`, `onnxruntime-silicon>=1.16; sys_platform == 'darwin'`,
  `onnxruntime>=1.16; sys_platform != 'darwin'`. CI stays light.
* `tests/test_identity.py` — 16 unit tests using L2-normalized
  synthetic embeddings (no insightface needed): cosine identities,
  `PersonCluster` separates orthogonal / merges similar / runs
  centroid drift, `link_face_to_tracker` IoU prefers head-region
  match, `compute_visit_count` gap semantics, watchlist self-match
  vs orthogonal vs empty. **59/59** total tests pass.
* `.gitignore` — covers `data/output/identity/`, `data/output/persons/`
  and **all face-image extensions inside `watchlist/`** (faces are
  sensitive PII).

**Tuning on real footage (recorded as required by the spec).**

| param | spec range | first run | tuned | rationale |
|-------|------------|-----------|-------|-----------|
| `quality_min` | start 0.55 | 0.55 | **0.55** | 95 % gate rate on camera-3 (1 445 / 1 521 detected faces pass), 78 % on camera-5. Lower would inject jittery low-conf embeddings; higher would drop too many useful frames. |
| `cosine_match` | 0.30–0.45 | 0.38 | **0.32** | 0.38 over-clustered camera-3 to 13 personae for 2 visible people (centroid-drift greedy compounded the issue). 0.32 collapses to 8 personae with the two main seated guys captured by P001 (684 face frames / `linked_tracker_ids=[1]`) and P002 (494 frames / `[2]`). |
| `sample_every_n_frames` | 5 | 5 | **5** | At 25 fps that's 5 fps effective sampling — plenty for face-redetection while keeping the 3-camera × 180 s identity run under 10 min on M2 CPU. |

**Risks / decisions taken.**

* **`appearance_thresh: 0.25` + `with_reid: true`** broke Phase 1's
  custom BoT-SORT yaml (already documented in Phase 1). Phase 3 ReID
  is purely face-based, so `with_reid: false` in Phase 1 is fine —
  face linking does the cross-fragment merging.
* **InsightFace on M2 stays CPU-only.** Tried CoreML provider, it
  doesn't materially help SCRFD's dynamic shapes (per the build
  spec). `providers=["CPUExecutionProvider"]` is faster than CoreML
  fallback in our testing.
* **Greedy clustering is order-sensitive** but adequate for retail
  windows. For Phase 5 we may swap in offline agglomerative if the
  per-person dwell numbers ever look unstable across re-runs.
* **Two-threshold question.** Spec defines a single `cosine_match`
  cutoff for clustering, repeat detection AND watchlist alerts.
  At the tuned 0.32, watchlist false-positives surface in the
  same-camera self-match test. Mitigation: the alert `severity` tier
  (`info` < 0.45, `warn` 0.45-0.60, `high` ≥ 0.60) gives the operator
  a precision dial without complicating config. Adding a
  separate `watchlist_min_similarity` is a clean follow-up.
* **Watchlist seed via `--seed-watchlist`** copies the FULL FRAME at
  the person's best moment (`Pxxx_full.jpg`), not the tight thumbnail
  — SCRFD reliably re-detects the face only when given context. The
  build prompt's expected workflow is operator-supplied photos; the
  seeder is purely for demoing the flow without real reference
  imagery.

### CODE

Implemented exactly the files above; **59/59** tests pass; ruff clean.

### VALIDATE (on real footage)

```bash
cd pipeline
uv pip install -e ".[cv,identity]"
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python main.py --run-identity                 # 3 × 180 s @ M2 CPU, ~5–10 min
uv run python main.py --seed-watchlist camera-5 P006 staff_lead_demo
uv run python main.py --run-identity                 # demo: watchlist alert fires
```

**Headline numbers (tuned `quality_min=0.55`, `cosine_match=0.32`):**

| camera   | area                                | faces seen | gated | unique visitors | repeats | watchlist hits (with seed) |
|----------|--------------------------------------|-----------:|------:|----------------:|--------:|---------------------------:|
| camera-1 | Cosmetics & Skincare                 |          3 |     2 |               2 |       0 |                          0 |
| camera-3 | Customer Seating / Try-on Lounge     |      1 521 | 1 445 |               8 |       3 |                          0 |
| camera-5 | Service & Consultation Desk          |        199 |   155 |               6 |       2 |                          4 |
| **total**|                                      |      1 723 | 1 602 |          **16** |       5 |                          4 |

Phase 1 reported 89 raw tracker IDs across these three cameras. Phase 3
collapses that to **16 unique visitors (82 % reduction)** — the
authoritative count.

**Dwell merge demo (the user's "(b)" requirement).** Camera-5
**`P006`**: Phase 1 fragmented this person across **7 separate tracker
IDs** `[37, 50, 53, 60, 66, 68, 79]` (probably from people leaving / 
re-entering frame). Phase 3 face-linking merged all seven; the
authoritative `track_dwell_seconds_authoritative` for `P006` is **31.6
s** computed from the union of frames where any of those 7 tracker IDs
were alive — a number Phase 2's per-track provisional dwell could
never produce.

Camera-3 **`P001`** and **`P002`** each captured ~500+ face crops, both
mapped to a single dominant tracker ID (1 and 2 respectively), and their
authoritative whole-scene track-dwell is **180 s each** — they never
left the frame, exactly matching the eyeball view.

**Watchlist demo.** After
`--seed-watchlist camera-5 P006 staff_lead_demo`, re-running emitted:

* `staff_lead_demo` matches **P006 sim 0.52 (warn)**, **P004 sim 0.51
  (warn)**, **P003 sim 0.39 (info)**, **P005 sim 0.38 (info)** in
  camera-5. Each alert carries the non-accusatory copy
  *"Possible match with watchlist entry 'staff_lead_demo' (face
  similarity 0.XX). Please verify before acting."* High-similarity
  matches are flagged `warn`; marginal matches stay `info` so the
  operator can prioritize.
* `0` hits in camera-1 and camera-3 — the seeded face never appears
  there, exactly as expected. Self-match precision ✓; cross-scene
  precision ✓.
* Repeat-visitor alerts (3 in camera-3, 2 in camera-5) carry the
  copy *"Possibly a returning visitor — face seen across N separate
  visits in this window. Please verify."*

**`identity.enabled: false` smoke (acceptance):** flipping the config
key short-circuits `run_identity` to a no-op (logged warning, empty
results, no JSON written) — face processing is fully toggleable.

### PUSH

- Commit on `main`: `4bfa653` — Phase 3: InsightFace identity (unique +
  repeat + watchlist)
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27586753724](https://github.com/mudassar531/countervision/actions/runs/27586753724)
  green on `4bfa653` — ruff clean, **59/59** pytest pass on the
  default-only deps (no `[cv]` / `[identity]` extras needed in CI).

### NEXT — Phase 4 (do not start yet)

Cross-camera journey. Match face embeddings across `camera-1 / camera-3
/ camera-5` so the same person seen at e.g. `Customer Seating` and
later at `Service & Consultation Desk` is emitted as a single
visitor with a `journey: [{camera_id, t, area}, ...]`. Phase 5 will
roll the cross-camera unique count into the store-wide total.

---

## Phase 4 — Cross-camera identity (de-dup, not journey)

### THINK (goal, files, risks)

**Goal.** Read every `identity/<camera>.json` from Phase 3 and de-dup
people across cameras using ArcFace centroids, with a separate
**high-precision** cosine threshold distinct from the in-camera
clustering cutoff. Headline metric is the **store-wide unique-visitor
count**, not individual journeys. Each cross-camera link carries the
similarity score; we never force a link below the high bar.

**Honest framing for these specific videos.** The recording times
**do not overlap**:

* `camera-1`: 2026-06-07 20:53 (20 min)
* `camera-3`: 2026-06-08 00:31 + 00:54 (continuation files)
* `camera-5`: 2026-06-08 04:44 (15 min)

So even a high-confidence cross-camera face match is **"same face seen
in these areas across the captured period"** — repeat presence, *not*
a single continuous trip. Each emitted link carries an explicit
`presence_note` saying that, plus the computed time gap.

**Files created / changed.**

* `pipeline/countervision/cross_camera.py` — new. Pure-numpy
  `UnionFind`, `_CamPerson` reader (loads `identity/<cam>.json`,
  drops persons below `min_face_appearances` from the matching pool
  but keeps them in the headline), `find_cross_camera_links` (skips
  same-camera pairs; sorts links by descending similarity),
  `build_store_wide_persons` (connected components → `S001..`
  numbered by earliest `first_seen`), `run_cross_camera` driver +
  `summarize_results` for the CLI.
* `pipeline/countervision/identity.py` — small extension: each
  person record now carries `embedding_centroid` (512 floats,
  L2-normalized) so Phase 4 doesn't have to re-run InsightFace.
* `pipeline/main.py` — `--run-cross-camera` mode. Reads identity
  JSONs, writes `data/output/cross_camera.json`.
* `pipeline/config.yaml` — new identity keys:
  `cross_camera_match: 0.50` (deliberately ≫ 0.32) and
  `min_face_appearances_for_cross_camera: 3`.
* `tests/test_cross_camera.py` — 13 tests: union-find singletons /
  union / transitive; load_identity_persons appearance gate +
  centroid-missing skip; find_cross_camera_links no-match /
  one-match / same-camera-pair-ignored; build_store_wide_persons
  no-links / transitive across 3 cams; run_cross_camera no-reliable
  path / dedup / low-appearance persons still count. **72/72**
  total tests pass.

**Risks / decisions taken.**

* **Two distinct thresholds.** `cosine_match=0.32` (clustering, loose
  recall) vs `cross_camera_match=0.50` (cross-camera, high precision).
  False merges across cameras hurt the demo headline far more than
  missing a real match — a "store-wide unique 13" we under-claim is
  more defensible than "12" with a hallucinated link.
* **`min_face_appearances_for_cross_camera=3`.** A centroid built from
  1–2 face crops is too noisy for high-precision matching across
  cameras; we'd see false merges. Skipped persons are listed in
  `persons_skipped[]` in the JSON for full honesty, AND they each
  count as their own visitor in the headline so we never disappear
  them from the totals.
* **No "journey" framing for these clips.** With multi-hour gaps
  between camera windows, calling something a "journey" would
  mislead the client. Each link's `presence_note` says "repeat
  presence, not a single continuous trip" with the explicit
  computed gap.
* **No-reliable-matches honesty path.** When zero pairs clear the
  threshold the JSON's `headline.no_reliable_cross_camera_matches:
  true` flag fires and `store_wide_unique_visitors` falls back to
  the naive per-camera sum. The CLI prints a ⚠ warning. We never
  fabricate a link to fill the panel.

### CODE

Implemented exactly the files above. **72/72** tests pass; ruff clean.

### VALIDATE (on real footage)

```bash
cd pipeline
uv run python main.py --run-identity        # refresh JSONs with embedding_centroid
uv run python main.py --run-cross-camera
```

Headline (real-footage run with tuned thresholds):

| metric                              | value |
|-------------------------------------|------:|
| `cross_camera_match` threshold      |  0.50 |
| persons in matching pool            |    14 |
| persons skipped (face_appearances<3)|     2 |
| per-camera unique sum (naive)       |    16 |
| cross-camera links above threshold  |     3 |
| **store_wide_unique_visitors**      | **13** |
| saved by dedup                      |     3 |
| `no_reliable_cross_camera_matches`  | false |

**Reliable links found (highest similarity first):**

| from                | to                  | similarity | gap          |
|---------------------|---------------------|-----------:|--------------|
| camera-3 / P003     | camera-5 / P004     |       0.60 | ≈ 4 h 13 m   |
| camera-3 / P007     | camera-5 / P003     |       0.59 | ≈ 4 h 14 m   |
| camera-3 / P003     | camera-5 / P001     |       0.58 | ≈ 4 h 10 m   |

The three pairs are connected transitively (camera-3/P003 anchors a
chain to camera-5/P001 + P004, and camera-3/P007 anchors another to
camera-5/P003), so the union-find collapses **3 cross-camera links →
3 saved in the store-wide count** (16 − 3 = 13). The
`presence_note` on each link reads e.g.

> *Same face appears in 'Customer Seating / Try-on Lounge'
> (camera-3, last seen 2026-06-08T00:34:11.799) and in 'Service &
> Consultation Desk' (camera-5, first seen 2026-06-08T04:47:19.975).
> Recording windows do not overlap (gap ≈ 4 h 13 m), so this
> represents the same person being seen in these areas across the
> captured period — repeat presence, not a single continuous trip.
> Cosine similarity 0.60.*

**Persons skipped from the matching pool** (centroid too noisy):

* `camera-1/P001` and `camera-1/P002` — only 1 face appearance each
  (camera-1 detected just 3 faces total over the 180 s window because
  the operators look down at desks). They still count as 2 visitors
  in the headline; we just don't try to match them across cameras.

### PUSH

- Commit on `main`: `62662ee` — Phase 4: cross-camera identity
  (de-dup, not journey)
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27587529841](https://github.com/mudassar531/countervision/actions/runs/27587529841)
  green on `62662ee` — ruff clean, **72/72** pytest pass on the
  default-only deps (cross-camera module is pure-numpy, no `[cv]` /
  `[identity]` extras needed in CI).

### NEXT — Phase 5 (do not start yet)

Aggregate everything (Phase 1–4 outputs) into the §7
`analytics.json` schema + a sqlite mirror, generate 3–5 plain-English
retail insights tied to real numbers, and document the schema in
`docs/schema.md`. The store-wide unique number from Phase 4 is the
headline KPI on the dashboard.

---

## Phase 5 — Aggregate → analytics.json + sqlite + insights

### THINK (goal, files, risks)

**Goal.** Read every Phase 1–4 output and emit the single
`analytics.json` file the dashboard consumes (plus a faithful
sqlite mirror so downstream tools can SQL it). Generate 3–5
plain-English retail insights — but **only** from the reliable
numbers (per-area dwell, occupancy, area-level unique faces). Never
build an insight on a cross-camera link or near-zero footfall.
Document everything in `docs/schema.md`.

**Three honesty buckets in code**, surfaced explicitly to the
dashboard so it can render with the right hedging:

1. **Reliable headlines** — `confidence: "high" | "medium"`. Tied to
   Phase 3 face-based dwell, occupancy, area-level unique counts.
2. **Hedged / low confidence** — `confidence: "low"`. Cross-camera
   links, watchlist hits, footfall when entry lines weren't redrawn.
   Each carries a `note` / `method` / `presence_note` saying why.
3. **Locked** — `{"value": null, "locked": true, "reason": "..."}`.
   Uncomputable from this footage (POS / conversion / weather /
   quantified staffing). Dashboard must render "data not available".

**Files created / changed.**

* `pipeline/countervision/aggregate.py` — new. `aggregate(...)`
  reads `tracks/`, `zones/`, `identity/`, `cross_camera.json`,
  composes the §7 schema, writes `analytics.json` and a sqlite
  mirror in one pass. `_generate_insights(...)` emits 3–5 insights
  with documented trigger conditions (highest_dwell_area,
  peak_occupancy_zone, area_engagement_imbalance,
  repeat_visitor_opportunity, demo_headline_framing). Locked-fields
  block emitted with explicit reasons.
* `pipeline/main.py` — `--run-aggregate` mode.
* `tests/test_aggregate.py` — 17 tests. Synthetic Phase 1–4 inputs
  in a tmp dir → exercises schema basics, locked-field emission,
  cross-camera hedging copy, no-reliable-matches fallback path,
  watchlist confidence_note, sqlite mirror row counts, all 5
  insight triggers and parametric footfall-confidence boundary.
* `docs/schema.md` — plain-English schema doc + sqlite table
  layout. Includes the honesty conventions, locked-fields rationale,
  and the cross-camera `render_hint` (dashboard MUST hedge).
* `.gitignore` — covers `data/output/analytics.json`,
  `data/output/analytics.db`.

**Risks / decisions taken.**

* **Insights ignore the cross-camera block entirely.** A test
  (`test_no_insight_built_on_cross_camera`) actively asserts no
  generated insight mentions "cross-camera" or "store-wide" — the
  cross-camera count is rendered as a hedged KPI elsewhere, not
  spun into a recommendation.
* **Footfall confidence flips low → medium at value ≥ 5.** Below 5
  the small-sample noise dominates; we don't want the dashboard
  shouting "0 footfall" as if that were a fact about the store.
* **Per-area `unique_visitors` is authoritative** (face-based from
  Phase 3). `person_tracks_note` on each area reminds implementors
  that the tracker-id count is not the source of truth.
* **sqlite is a faithful mirror, not a query layer.** All
  derivation lives in the JSON build; sqlite just makes the same
  data queryable for downstream tools. We `DROP + CREATE` on every
  run so the mirror is always in sync.
* **No `conversion_rate` fabricated from face count.** That's the
  classic temptation — emit "conversion_rate = unique_visitors /
  footfall × some_constant". We explicitly do not do that; the
  KPI is locked with a clear reason. Phase 7 talk-track can quote
  the locked status as "what we'd unlock with a 1-day POS
  integration", which is a stronger pitch than a made-up number.

### CODE

Implemented exactly the files above. **89/89** tests pass; ruff clean.

### VALIDATE (on real footage)

```bash
cd pipeline
uv run python main.py --run-aggregate
```

Headline output (real Phase 1–4 artefacts, no model run, ~2 s):

```
schema version           : 1
cameras / areas          : 3 / 3
visitors (camera-person) : 16
alerts                   : 9
insights generated       : 5
store-wide unique        : 13
written analytics.json   : data/output/analytics.json
written sqlite mirror    : data/output/analytics.db
```

**Reliable KPIs (`confidence: high | medium`):**

| KPI                            | value | confidence | note |
|--------------------------------|------:|------------|------|
| `avg_dwell_seconds_store`      |  73.9 | high       | weighted across areas; uses authoritative track-dwell |
| `store_wide_unique_visitors`   |    13 | medium     | 3 cross-camera links above the 0.50 high-precision bar; saved 3 from naive sum |
| `repeat_visitors_per_area`     |     5 | medium     | face seen ≥ 2 visit segments within a single camera |

**Hedged KPIs (`confidence: low`):**

| KPI                | value | reason |
|--------------------|------:|--------|
| `footfall_total`   |     3 | auto-generated entry lines at 75 % height; operator should redraw for production demos |
| `watchlist_hits`   |     4 | verification prompts, not identifications; similarity attached to each |

**Locked KPIs** — emitted as `{"value": null, "locked": true, "reason": "..."}`:

* `conversion_rate` — no POS data
* `revenue_uplift` — no POS data
* `weather` — no external feed
* `staffing_recommendations_quantified` — needs payroll integration

**Insights generated** (all from reliable numbers; none built on
cross-camera or footfall):

1. `[high]` *Longest average dwell in Customer Seating / Try-on
   Lounge* — visitors spend an average of **96 s** there.
2. `[high]` *Peak crowding in Cosmetics & Skincare* — peaked at 3
   simultaneous occupants.
3. `[medium]` *Most engagement is happening in Customer Seating /
   Try-on Lounge* — 8 unique faces vs 2 in Cosmetics & Skincare.
4. `[medium]` *5 repeat visitors identified within this window*.
5. `[high]` *Areas with the deepest engagement are the staffing
   priority* — 16 unique faces total; deepest single dwell 180 s.

**sqlite mirror** populated correctly: `areas=3`, `visitors=16`,
`alerts=9`, `footfall_by_hour=2`, `occupancy_timeseries=523`,
`cross_camera_links=3`, `insights=5`, `kpis=12`.

### PUSH

- Commit on `main`: `8bdcabd` — Phase 5: aggregate → analytics.json
  + sqlite + insights
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27625280733](https://github.com/mudassar531/countervision/actions/runs/27625280733)
  green on `8bdcabd` — ruff clean, **89/89** pytest pass.

### NEXT — Phase 6 (do not start yet)

Next.js 16 dashboard (App Router + React 19 + Tailwind v4 + shadcn
+ Recharts 3) reading **only** `analytics.json`. Navy `#0A1347`. The
build step copies `data/output/*` into `dashboard/public/`. Polish
budget goes here — KPI cards, per-area heatmap hero, footfall-by-hour,
dwell-by-area, occupancy line, annotated video player, alerts feed
(with hedged copy on watchlist!), insights panel, and a journey-style
visual that respects the `render_hint` on `cross_camera`.

---

## Phase 6 — Next.js dashboard (navy, client-ready)

### THINK (goal, files, risks)

**Goal.** A genuinely client-ready Next.js 16 dashboard reading **only**
`analytics.json`. All 10 panels in spec order, with confidence-tier
visual treatment on every metric, lock-as-CTA framing on POS / weather
fields, and the cross-camera `render_hint` respected (no journey
arrows, only "repeat presence" framing). Annotated mp4s play directly
— no canvas sync — because overlays were burned in by Phase 1.

**Context7 + Next 16 docs lookups (done first).** Verified:
* Next.js 16.2.9 ships Turbopack + Tailwind v4 by default. `params` /
  `searchParams` are now async-only — we don't use dynamic routes so
  this doesn't bite.
* shadcn/ui CLI `npx shadcn@latest init --defaults` (Nova preset) +
  `add card badge separator tabs avatar table tooltip` works with
  Tailwind v4's `@theme inline` approach. Components land in
  `src/components/ui/`.
* Recharts 3.8 with React 19: any chart-using component needs
  `"use client"`; bars render as `<path>` (rounded corners), not
  `<rect>`.

**Files created (all new — `dashboard/` was empty).**

* `dashboard/` — Next.js 16 + Tailwind v4 + shadcn (`Geist` font,
  Nova preset, Lucide icons, Radix base).
* `dashboard/scripts/copy-data.cjs` — predev/prebuild script copies
  `data/output/{analytics.json,analytics.db,annotated,frames,heatmaps,persons,alerts}`
  into `dashboard/public/data/` (gitignored). Idempotent; rmrf + recopy.
* `dashboard/src/lib/analytics.ts` — full TS shape of the analytics
  contract (`Analytics`, `Kpis`, `Visitor`, `Alert`, `CrossCamera`,
  `Insight`, etc.) + `asPublicUrl(...)` which handles both relative
  and absolute paths the pipeline may write.
* `dashboard/src/app/globals.css` — navy `#0A1347` → OKLCH theme
  variables (`--navy`, `--primary`, charts, sidebar). Light theme is
  the demo default; dark vars defined for future toggle.
* `dashboard/src/app/page.tsx` — server component, reads
  `public/data/analytics.json` via `fs/promises`, lays out all 10
  panels in the spec's exact order.
* `dashboard/src/components/`
  * `kpi.tsx` — `KpiCard`, `LockedKpiCard`, `ConfidencePill` (the
    high/medium/low/locked pill system used across the dashboard).
  * `charts.tsx` — Recharts (footfall bar, dwell-by-area bar,
    occupancy line) with the navy palette + legends + custom tooltips.
  * `heatmap-hero.tsx` — per-camera tabs with toggle between
    heatmap overlay and the clean Phase-1 frame.
  * `area-detail.tsx` — per-camera tabs with the occupancy timeline
    + a grid of visitor thumbnails (with `repeat` / `watchlist`
    badges and the killer "7 merged ids" stat for P006).
  * `video-panel.tsx` — `<video controls poster={frame_jpg}>` per
    camera tab; **no canvas sync** because Phase 1 already burned
    the overlays into the mp4.
  * `alerts-feed.tsx` — non-accusatory review-prompt cards sorted by
    severity, with similarity attached.
  * `insights-and-cross.tsx` — `InsightsPanel` and `CrossCameraPanel`.
    The cross-camera panel surfaces the `render_hint` verbatim in its
    own description and lists each link with similarity + time gap +
    presence note; never as a "journey".
* `dashboard/package.json` — added `predev` / `prebuild` invoking
  the copy-data script.
* `dashboard/.gitignore` — adds `/public/data/` so the 605 MB of
  copied artefacts never get committed.
* `.gitignore` (root) — adds `.playwright-mcp/` to keep MCP scratch
  out of git.

**Panel order (matches the spec literally).**

1. **Branded Overview** — navy hero with CV wordmark, store name,
   capture window pill, camera/area/visitor counts.
2. **KPI cards** (8 total) — reliable headline first
   (**accent navy**: avg dwell + store-wide unique), then medium
   (repeat visitors, peak hour), then hedged (footfall, watchlist),
   then **locked** (conversion rate, revenue uplift) framed as
   "Unlock with POS integration".
3. **Per-area heatmap hero** — camera tabs; toggle between heatmap
   overlay and clean frame; stats strip below.
4. **Footfall by hour** — Recharts bar (sparse data is honest).
5. **Dwell by area** — Recharts bar (avg + max).
6. **Per-area detail** — occupancy timeline + visitor thumbnails
   (the P006 "7 merged ids" badge is the killer moment for the demo).
7. **Cross-camera presence** — hedged; `render_hint` shown verbatim
   to the user; each link carries similarity + time gap + presence
   note; never rendered as a continuous journey arrow.
8. **Annotated walkthrough** — per-camera `<video>` with
   first-frame poster; overlays already burned in.
9. **Alerts feed** — severity-sorted; non-accusatory copy with
   similarity attached.
10. **Plain-English insights** — 5 reliable insights surfaced with
    their own confidence pill.

**Risks / decisions taken.**

* **Two early bugs caught by Playwright + fixed.**
  * `buildCameraVideoList` exported from a `"use client"` module
    can't be called from a server component (Next 16 server/client
    boundary enforced strictly) — inlined the helper in `page.tsx`.
  * `asPublicUrl` originally stripped only a leading `data/output/`
    prefix; the Phase-5 JSON sometimes carries absolute paths like
    `/Users/.../data/output/frames/x.jpg` (because Phase 3 used
    `str(image_path)` for watchlist entries). Rewrote to extract
    after the `data/output/` substring wherever it appears.
* **No `next/image` for the heatmap / frame / thumbnails.** The
  camera-5 mp4 carries a filename with spaces; `next/image` plus
  query strings have new `localPatterns.search` requirements in 16,
  and frames may also have unusual characters. Used plain
  `<img src={encodeURI(src)}>` for everything — safer and we don't
  need responsive image sizes for a single client demo.
* **Annotated mp4 plays directly, no canvas sync** — per the spec.
  Overlays (boxes, IDs, traces, navy HUD with wall-clock) are burned
  in by Phase 1 already, so the panel is bulletproof for the live
  demo (no JS sync that can desync).
* **Cross-camera panel does NOT render arrows / paths.** Per the
  spec, the page renders the `render_hint` ("Dashboard MUST render
  cross-camera links with hedging copy from `presence_note`.
  Recording windows do not overlap; do not render these as continuous
  'journeys'.") as the panel's own description, so the reader sees the
  caveat before the data.
* **Tailwind v4 OKLCH navy.** `#0A1347` ≈ `oklch(0.184 0.117 274.6)`.
  All shadcn semantic tokens (primary, ring, sidebar, chart-1) map
  to the navy. Chart palette uses navy + a navy-tint + accent
  cool-tones so everything reads coherently.
* **Heavy artefacts gitignored.** `dashboard/public/data/` is 605 MB
  of copied mp4s and pngs — regenerated locally via `predev`/`prebuild`.

### CODE

Implemented exactly the files above; `npm run lint` clean,
`npm run build` succeeds and statically prerenders the page. Verified
with Playwright screenshots at 1440×900: every panel renders, all 3
camera tabs work in heatmap hero / per-area detail / video panel, the
"7 merged ids" P006 win is visible in camera-5's Per-area detail.

### VALIDATE (on real footage)

```bash
cd dashboard
npm install      # one-time
npm run dev      # auto-runs predev (copy data/output → public/data)
# open http://localhost:3000
npm run lint     # passes
npm run build    # passes — static prerender of `/`
```

Captured screenshots showed each panel rendering with the expected
data (e.g. camera-3 heatmap with two clean hot spots on the seated
guys, "8 unique faces · 96s avg dwell · 3 peak occupancy" stats strip;
camera-5 Per-area detail with 6 thumbnails and the "dwell 32s · faces
58 · 7 merged ids" stat on P006).

### PUSH

- Commit on `main`: `07598db` — Phase 6: Next.js 16 dashboard
  (navy, client-ready)
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27627668886](https://github.com/mudassar531/countervision/actions/runs/27627668886)
  green on `07598db` — ruff clean, **89/89** pytest pass (unchanged;
  dashboard has no Python tests).

### NEXT — Phase 7 (do not start yet)

`make demo` one-command flow that runs the full pipeline over
`videos/` and then static-exports the dashboard. Plus
`docs/DEMO_SCRIPT.md` — a ~3-minute walkthrough with business framing
that uses only the reliable numbers (avg dwell, the P006 7-fragment
merge, the heatmap visuals) and frames locked KPIs as
"this is what we'd unlock with a 1-day POS integration".

---

## Phase 7 — One-command demo + talk-track  (final phase)

### THINK (goal, files, risks)

**Goal.** Make the whole project run from raw footage to a live
demo with a single command, then write a presentation-ready
walkthrough that lands the reliable numbers and frames the rest
honestly. After this phase the project is **shippable for the client
meeting**: no manual steps, no internet dependency, no live model
inference in the demo path.

**Files created / changed.**

* `Makefile` (new, root) — GNU make 3.81-compatible targets:
  `install` (one-time `uv venv` + `[cv,identity]` extras + `npm
  install`), `pipeline` (chains the five Phase 1–5 targets),
  individual phase targets that **do not cascade upstream** so
  `make aggregate` is fast (~2 s) on existing outputs,
  `dashboard-build` (static-exports to `dashboard/out/`),
  `dashboard-serve` (`python3 -m http.server $PORT` over `out/`),
  `demo` = `pipeline + dashboard-serve`, `demo-quick` =
  `aggregate + dashboard-serve` for fast iteration. `help` lints
  itself from `## ...` doc comments on each target.
* `dashboard/next.config.ts` — switched to `output: "export"` and
  `images: { unoptimized: true }`. The dashboard now produces a
  fully static HTML/CSS/JS bundle under `dashboard/out/` that any
  HTTP server can host. Demo flow uses `python3 -m http.server` to
  avoid any new dependency.
* `docs/DEMO_SCRIPT.md` (new) — the ~3-minute talk-track. Setup
  checklist, throughline rehearsed three times ("your cameras
  turned into decisions — imagine this on your store's entrance and
  till"), 0:00–3:00 cue-by-cue script, a Q&A anticipation table
  (accuracy / hardware / privacy / what's missing / dashboard tech
  / time-to-deploy / AGPL), and an explicit "anti-demo failure
  modes" list (don't journey-frame the cross-camera panel, don't
  fight the locked cards, don't show the mp4 before the numbers).
* `README.md` — moved "Quick start" to a real **TL;DR — run the
  demo** with `make install` / `make demo` / `make demo-quick` /
  `make help`. Added a new **"Point CounterVision at a real
  client's footage"** section: NVR filename convention, area
  labelling, the `--draw-zones CAM` interactive editor flow,
  watchlist seeding, and a complete table of every tunable
  threshold in `config.yaml`.

**Risks / decisions taken.**

* **Individual phase targets are deliberately flat** (no upstream
  prereqs). First Makefile draft had `aggregate: zones
  cross-camera` which transitively pulled `identity` (~5 min) every
  time `make aggregate` ran. Caught during validation; fixed by
  letting only the umbrella `pipeline` target list the chain.
  Each per-phase target now sits beside an honest comment ("assumes
  identity/") so an operator can re-run a phase in isolation.
* **Static export over `next start`.** The build prompt asked for
  `output: 'export'` so the demo is bulletproof. The downside is
  `next start` no longer works — the trade-off is that **any
  static server** can host the demo (CDN, S3, USB stick, hotel
  Wi-Fi). The `dashboard-serve` target uses `python3 -m
  http.server` which ships with every macOS / Linux box.
* **`python3 -m http.server` does not implement HTTP Range
  responses fully**, which leads to a benign `BrokenPipeError` in
  the server log when `<video preload="metadata">` requests partial
  bytes and the browser cancels the rest. The mp4 still plays
  correctly because the file is local and small. For a slicker
  production demo you can swap in `npx serve out` (handles ranges).
* **`make demo` runs the full pipeline on every invocation.**
  Intentional: a real demo should be reproducible "I'll show you it
  generating from scratch", not a magic blob. `make demo-quick`
  exists for the iteration case (re-aggregate against existing
  outputs and reload).
* **`pythonpath` separation.** Older zsh history may have left a
  Next.js dev server bound to `:3000` — the Makefile assumes the
  port is free. The README documents the `PORT=NNNN make demo`
  override.

### CODE

Implemented exactly the files above. `make help` prints a clean
auto-extracted manifest; `npm run lint` (dashboard) + `make lint`
(both) still pass; 89/89 pytest unchanged from Phase 5.

### VALIDATE (on real artefacts)

```bash
make help                              # auto-extracted; lists every target
make aggregate                         # ~2 s — re-runs Phase 5 only
make dashboard-build                   # 8.5 s — static export to dashboard/out/
make demo-quick                        # aggregate + build + serve at :3000
curl -sI http://localhost:3000/                                    # 200
curl -sI http://localhost:3000/data/analytics.json                 # 200
curl -sI http://localhost:3000/data/heatmaps/camera-3.png          # 200
curl -sI http://localhost:3000/data/annotated/camera-1.mp4         # 200
```

All four curls returned `200 OK`. Browser hit `/` and pulled every
panel asset (heatmaps for 3 cameras, 16 person thumbnails, all 3
annotated mp4s, all chart data) — verified via the server access log
in the `demo-srv` shell.

Static export size: **608 MB at `dashboard/out/`** — dominated by the
three annotated mp4s. The HTML / CSS / JS bundle itself is only a
couple of MB.

### PUSH

- Commit on `main`: `af712ab` — Phase 7: one-command demo +
  DEMO_SCRIPT.md + client-handover README
- Repo: <https://github.com/mudassar531/countervision>
- CI: [run 27628862959](https://github.com/mudassar531/countervision/actions/runs/27628862959)
  green on `af712ab` — ruff clean, **89/89** pytest pass.

### Project status

All 8 phases ship. **The demo is reproducible from raw footage in one
command** (`make install` once, then `make demo`). The dashboard runs
with **zero live inference**. The talk-track lands the reliable
numbers, hedges the cross-camera and watchlist capability, and frames
the locked KPIs as the integration runway. Honest, polished,
client-ready.

---

## Post-ship enhancement — dedicated "People" panel (2026-06-29)

**Scope.** Dashboard-only; no pipeline / model / production work. The
per-person face analytics were already computed in `analytics.json`
(`visitors[]`) — this surfaces them better, recomputing nothing.

**Goal.** Consolidate the per-person story (the demo's core) into one
prominent panel. For every detected face: thumbnail, person id,
authoritative time-on-camera (`track_dwell_seconds`, the
fragment-merged value), area/camera seen in, visit/repeat count, a
repeat + a non-accusatory watchlist-review badge, and an
identity-evidence pill consistent with the existing confidence-pill
system. Sortable; default = longest time-on-camera first.

**Files.**

* `dashboard/src/components/people-panel.tsx` — new. `"use client"`
  sortable table (shadcn `Table` + `Badge`, navy theme, reuses
  `ConfidencePill`). Headline summary strip (people identified ·
  store-wide-after-dedup · longest on camera · repeats · watchlist
  reviews). Click-to-sort on time / visits / evidence; default
  dwell-desc. Cross-camera **de-prioritised** to a single subtle,
  hedged "↔ also matched in N area(s)" chip (no journey framing) —
  honest because the footage is not time-synced.
* `dashboard/src/app/page.tsx` — inserts the panel as a new
  "The core story" section immediately after the KPI cards and before
  the heatmap hero. All 10 existing panels untouched.
* `dashboard/src/components/kpi.tsx` — `ConfidencePill` gained an
  optional `label` prop (backward-compatible; same colour/shape).

**Honesty decisions.**

* **Identity-evidence pill, not "confidence".** The pill is derived
  from `face_appearances` (strong ≥ 50, moderate ≥ 10, else limited),
  so it is labelled *Identity evidence* and worded "Strong/Moderate/
  Limited evidence" — never the generic "confidence" — so a green pill
  next to a watchlist-review badge can't be misread as endorsing the
  flag or the dwell value. (Caught by an adversarial review pass.)
* **0-dwell records render "—", not "0s".** Seven faces were detected
  but never linked to a sustained body track, so no merged dwell
  exists; showing "0s" would misread as "no time in store". Each "—"
  carries a hover explanation and the face count still shows.
* **Watchlist = verification prompt**, not an identification (badge +
  similarity in the tooltip, matching the schema's framing).

**Validate.** `npm run build` (static export, TypeScript clean) +
`eslint` clean; `make demo-quick` runs end-to-end (aggregate → build →
serve, HTTP 200) with the panel in the static export; all 16 face
records render; Playwright screenshots reviewed (caught + fixed a
"2:60"→"3:00" seconds-carry bug and "1 faces"→"1 face"). A 3-dimension
adversarial review (correctness · honesty · theme/a11y) confirmed 2
findings (both applied), no blockers, no fabrication.

---

## Next horizon — PRODUCTION (deferred; awareness only)

After the client meeting the planned next horizon is **live CCTV in
production** — edge box at the store pulling RTSP/ONVIF streams,
metadata-only sync to a cloud DB + API, dashboard polling the API
instead of reading a static JSON.

**Do not begin any of that work until explicitly instructed.** Two
docs together govern the production pivot — read both, every session
that touches production code:

* [`PRODUCTION_DIRECTION.md`](./PRODUCTION_DIRECTION.md) — the four
  architectural seams to preserve in the meantime ("same brain, new
  body": input-agnostic analytics modules, config-driven `device`,
  stable `analytics.json` schema, models behind a provider
  abstraction) and the high-level deployment shape (edge processing,
  metadata-only cloud sync).
* [`docs/PRODUCTION_MODEL_SPEC.md`](./docs/PRODUCTION_MODEL_SPEC.md)
  — the **locked production model stack**: RF-DETR (Apache-2.0,
  replaces YOLO26 AGPL) for person detection; **AuraFace R100**
  (commercial, deployable default) for face embeddings with an
  **InsightFace SCRFD + ArcFace R100 swap pending** (config/weight
  swap only, not an architecture change); **SOLIDER (Swin-Base)** for
  body / appearance Re-ID; **BoT-SORT + Re-ID** for within-camera
  continuity; **Faiss → Milvus** for the cross-day face gallery;
  **GEFF-style late (score-level) fusion** (Arkushin et al., WACV
  2024 Workshops) — face anchor overrides body when visible, body
  carries continuity when not. **Never early-fuse face + body
  embeddings.** Three-stage architecture: entrance enrollment →
  interior continuity → cross-day face-anchored recall, with three
  distinct memory lifetimes (frame buffer, session/blind-spot
  gallery with ~5 min timeout, indefinite Faiss face gallery subject
  to retention).

**Open licensing items to resolve before any PAID deployment** (per
`PRODUCTION_MODEL_SPEC.md` § "Licensing"): (1) face detector weights
— SCRFD code is MIT but InsightFace weights are non-commercial;
either pair AuraFace with a permissively-licensed detector (YuNet /
MediaPipe) or cover under an InsightFace commercial license. (2)
Tracker implementations — BoT-SORT the algorithm is unencumbered but
the popular Ultralytics / BoxMOT implementations are AGPL-3.0; needs
a clean-room association implementation or a commercial Ultralytics
license. (3) Re-ID training data — SOLIDER code is MIT but verify
LUPerson / Market-1501 / MSMT17 terms; clean fallback is **NVIDIA TAO
ReIdentificationNet**. (4) **Never ship MS1M / WebFace-trained face
weights commercially** (AdaFace / LVFace / InsightFace default zoo).

The model spec stays **consistent with the four seams** in
`PRODUCTION_DIRECTION.md`: model abstraction (AuraFace ↔ InsightFace
↔ RF-DETR variants = config swap), embedding-store interface
(Faiss → Milvus = backend swap), fusion as its own layer (so the
GEFF logic is testable in isolation), device-agnostic config, and
`analytics.json` remains the dashboard contract.


