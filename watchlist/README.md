# Watchlist

Drop reference face JPGs here (one face per file). Phase 3 (identity) will
embed each one with InsightFace (ArcFace, 512-d) and emit a non-accusatory
**`watchlist`** alert when a matching face is seen in any camera.

- Filename = the watchlist label shown in the dashboard (e.g. `staff_lead.jpg`).
- One subject per image; frontal-ish, eyes visible, ≥ 112 px between eyes.
- Cosine cutoff is set by `identity.cosine_match` in `pipeline/config.yaml`
  (start 0.30–0.45 — tune on real footage in Phase 3).

Privacy: only the 512-d **embedding** and the cropped thumbnail are persisted
to `data/output/`. Raw watchlist source images stay here (gitignored) and
should not be redistributed.
