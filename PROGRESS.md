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
| 1 | Detect + track (YOLO26 MPS + BoT-SORT)              | ✅ done     | 27/27 tests; 3 cameras × 180 s validated on real footage (89 IDs, 38 ID-switch-proxy events, ID #28 survived a 2.20 s occlusion). |
| 2 | Zones / footfall / dwell / heatmap / occupancy      | ⏳ pending  | Awaits go-ahead. |
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
