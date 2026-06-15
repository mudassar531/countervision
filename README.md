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

# 4. Tests + lint
uv run ruff check ..
uv run pytest ../tests -q
```

The dry-run is what CI runs on every push (`.github/workflows/ci.yml`).

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
