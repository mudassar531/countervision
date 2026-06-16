# CounterVision — demo Makefile
#
#   make help          show this help
#   make install       one-time: uv venv + python deps + npm install
#   make demo          full pipeline + static dashboard + serve
#   make demo-quick    re-aggregate + dashboard (skips heavy phases)
#   make dashboard     build static export + serve
#
# Individual phases are also exposed: make detect-track / zones / identity /
# cross-camera / aggregate / dashboard-build / dashboard-serve.

.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

VENV    := pipeline/.venv
PYTHON  := $(VENV)/bin/python
UV      ?= uv
DATA    := data/output
PORT    ?= 3000
OUT_DIR := dashboard/out

# Required for any Ultralytics / supervision / insightface call on Apple Silicon.
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: help install install-pipeline install-dashboard \
        pipeline detect-track zones-default zones identity cross-camera aggregate \
        dashboard dashboard-build dashboard-serve dashboard-dev \
        demo demo-quick \
        test lint \
        clean clean-data clean-dashboard

# ----- help ------------------------------------------------------------------

help:
	@printf "\n\033[1mCounterVision — demo Makefile\033[0m\n"
	@printf "  Targets:\n"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
	   nb=index($$0,"##"); \
	   target=substr($$0,1,index($$0,":")-1); \
	   doc=substr($$0,nb+3); \
	   printf "    \033[36m%-22s\033[0m %s\n", target, doc; \
	 }' $(MAKEFILE_LIST)
	@printf "\n  Quick start (one-time):  make install\n"
	@printf "  Run the full demo:       make demo\n"
	@printf "  Re-aggregate + serve:    make demo-quick\n\n"

# ----- one-time install ------------------------------------------------------

install: install-pipeline install-dashboard  ## one-time: uv venv + Python + Node deps

install-pipeline:  ## set up the Python pipeline (uv venv, [cv,identity] extras)
	@command -v $(UV) >/dev/null 2>&1 || { echo "uv not found — install with: brew install uv"; exit 1; }
	cd pipeline && $(UV) venv --python 3.11
	cd pipeline && $(UV) pip install -e ".[cv,identity]"
	@echo ""
	@echo "Pipeline installed. First-run model downloads (one-time):"
	@echo "  - yolo26s.pt        (~20 MB, into pipeline/)"
	@echo "  - buffalo_l face    (~326 MB, into ~/.insightface/models/)"

install-dashboard:  ## install dashboard node_modules
	cd dashboard && npm install

# ----- pipeline phases -------------------------------------------------------

# Pipeline phases are phony — they regenerate the artefacts every time. Use
# `make demo-quick` if you only want to re-aggregate against existing per-phase
# outputs and reload the dashboard.

pipeline: detect-track zones identity cross-camera aggregate  ## run all phases (~25 min on M2 Pro)
	@echo ""
	@echo "Pipeline finished. Generated:"
	@ls -1 $(DATA)/*.json 2>/dev/null || echo "  (no JSON outputs found?)"

detect-track:  ## Phase 1: YOLO26 + BoT-SORT (per-camera, ~12 min)
	$(PYTHON) pipeline/main.py --run-detect-track

zones-default:  ## populate config.yaml with sane default zones + entry lines
	$(PYTHON) pipeline/main.py --draw-zones-default

zones:  ## Phase 2: footfall / dwell / heatmap / occupancy (assumes tracks/)
	$(PYTHON) pipeline/main.py --draw-zones-default
	$(PYTHON) pipeline/main.py --run-zones

identity:  ## Phase 3: InsightFace buffalo_l (per-camera, ~5 min on M2 CPU; assumes tracks/)
	$(PYTHON) pipeline/main.py --run-identity

cross-camera:  ## Phase 4: dedup people across cameras (assumes identity/)
	$(PYTHON) pipeline/main.py --run-cross-camera

aggregate:  ## Phase 5: build analytics.json + analytics.db (assumes zones/ + identity/ + cross-camera)
	$(PYTHON) pipeline/main.py --run-aggregate

# ----- dashboard -------------------------------------------------------------

dashboard-build: $(DATA)/analytics.json  ## static-export the dashboard to dashboard/out/
	cd dashboard && npm run build
	@echo ""
	@echo "Static export at $(OUT_DIR)/"
	@du -sh $(OUT_DIR) 2>/dev/null || true

$(DATA)/analytics.json:
	@if [ ! -f "$@" ]; then \
	  echo "ERROR: $@ missing. Run 'make aggregate' or 'make pipeline' first."; \
	  exit 1; \
	fi

dashboard-serve: dashboard-build  ## serve dashboard/out/ on $$PORT (default 3000)
	@echo ""
	@echo "Dashboard ready at http://localhost:$(PORT)/  (Ctrl-C to stop)"
	@echo ""
	cd $(OUT_DIR) && python3 -m http.server $(PORT)

dashboard: dashboard-serve  ## alias: build static + serve

dashboard-dev:  ## hot-reload dev server (skip if you just want the demo)
	cd dashboard && PORT=$(PORT) npm run dev

# ----- the demo --------------------------------------------------------------

demo: pipeline dashboard-serve  ## end-to-end: run the pipeline then serve the dashboard

demo-quick: aggregate dashboard-serve  ## re-aggregate against existing outputs, then serve

# ----- developer convenience -------------------------------------------------

test:  ## pytest the pipeline (no GPU / model deps required)
	cd pipeline && $(VENV)/bin/pytest ../tests -q

lint:  ## ruff (pipeline) + eslint (dashboard)
	cd pipeline && $(VENV)/bin/ruff check ..
	cd dashboard && npm run lint

# ----- clean -----------------------------------------------------------------

clean: clean-data clean-dashboard  ## remove all generated artefacts

clean-data:  ## remove pipeline outputs in data/output/
	rm -rf $(DATA)/annotated $(DATA)/frames $(DATA)/heatmaps $(DATA)/tracks \
	       $(DATA)/zones $(DATA)/identity $(DATA)/persons $(DATA)/alerts
	rm -f  $(DATA)/*.json $(DATA)/*.db

clean-dashboard:  ## remove dashboard build + copied data
	rm -rf dashboard/.next dashboard/out dashboard/public/data
