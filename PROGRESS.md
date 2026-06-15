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
| 1 | Detect + track (YOLO26 MPS + BoT-SORT)              | ⏳ pending  | Awaits go-ahead. |
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
