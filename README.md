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

# 3. Tests + lint
uv run ruff check .
uv run pytest ../tests -q
```

The dry-run is what CI runs on every push (`.github/workflows/ci.yml`).

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
│   ├── config.yaml             # cameras → area, zones, lines, thresholds
│   ├── main.py                 # entrypoint (--dry-run today)
│   └── countervision/
│       ├── __init__.py
│       ├── discover.py         # camera + video discovery, probing
│       ├── logging_setup.py
│       └── timeparse.py        # YYYYMMDDHHMMSSmmm → datetime
├── tests/                      # pytest
├── data/output/                # pipeline artifacts (gitignored)
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
