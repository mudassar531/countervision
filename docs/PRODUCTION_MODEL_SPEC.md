# PRODUCTION_MODEL_SPEC.md — the model stack for live CCTV. Read, do not build yet.

> Companion to `PRODUCTION_DIRECTION.md`. That file says *what* the production system
> is and to keep the seams clean. **This file says which models to use when we build it.**
> Same rule: **awareness/spec only — do NOT install, download, or build any of this until
> explicitly instructed.** The demo stays exactly as it is.

## The method (headline)
The production identity system is a **two-stage, face-anchored, late-fusion** pipeline:
a controlled **face enrollment at the entrance** gives a high-precision identity anchor;
**body/appearance Re-ID** carries that identity across interior cameras and blind spots
when the face is not visible. The fusion pattern is **GEFF-style gallery enrichment**
(Arkushin et al., WACV 2024 Workshops): face-confirmed identities enrich the body-Re-ID
gallery, and at match time **face overrides body when a face is visible; body provides
continuity when it is not.** This is the SOTA-validated approach for the exact problem we
have (overhead interior cameras where faces are usually unusable). **Never early-fuse /
average face and body embeddings** — that degrades both. Late (score-level) fusion only.

## Locked model stack

| Job | Model | License (code) | Commercial weights? | Role |
|---|---|---|---|---|
| Person detection | **RF-DETR Large** (Roboflow) | Apache 2.0 | Yes | Replaces YOLO26 (AGPL) |
| Face detection | **SCRFD** architecture | MIT | ⚠ see open item #1 | Detect+align faces at entrance |
| Face embedding | **AuraFace R100** (fal) | Commercial | **Yes** | Identity anchor (override) |
| Body/appearance Re-ID | **SOLIDER (Swin-Base)** | MIT (code) | ⚠ see open item #3 | Cross-camera continuity |
| Fusion | **GEFF-style late fusion** | (pattern, not a dep) | — | Face anchor + body continuity |
| Tracker | **BoT-SORT** + Re-ID on | ⚠ see open item #2 | — | Within-camera continuity |
| Vector DB | **Faiss** (Meta) → Milvus at scale | MIT / Apache | Yes | Cross-day face recall |

**Upgrade swap (pending):** we have emailed InsightFace for a commercial license on their
SCRFD + ArcFace R100 weights (buffalo_l/antelopev2 tier — higher accuracy than AuraFace on
low-quality CCTV faces). **If they reply with workable terms, swap AuraFace → InsightFace
weights. This is a config/model-file change only — architecture and fusion logic do not
change.** Until then, **AuraFace is the deployable default and there are no blockers.**

## Architecture (three stages)

**Stage 1 — Entrance (enrollment).** Detect face (SCRFD) + person (RF-DETR) on entry →
align → compute **face embedding (AuraFace)** AND **body embedding (SOLIDER)** on the same
person crop → look up face embedding in the Faiss gallery → tag **new / returning / staff**
→ write a visit record. The entrance is the only place we rely on a good frontal face.

**Stage 2 — Interior (continuity).** Per camera, in parallel: RF-DETR detection →
**BoT-SORT with Re-ID enabled, using SOLIDER embeddings** for within-camera continuity →
**cross-camera GEFF fusion**: SOLIDER body embedding is the primary signal inside (face
usually not visible); when a face *is* visible, AuraFace re-confirms and overrides.

**Stage 3 — Cross-day memory.** On each entrance event, face embedding → Faiss top-k
cosine search → above threshold = returning customer (increment visit count); below = new
(add to gallery). **Cross-day recall is face-anchored only** — body/clothing embeddings are
NOT stored across days (clothes change daily); they expire at session end.

## Identity lifecycle / "how long does it remember"
Three distinct memory layers, three different lifetimes:
1. **Frame tracker buffer** (BoT-SORT) — holds a lost track for N frames (~seconds) for
   short occlusions. Tune `track_buffer`; higher = longer occlusion tolerance but more ID
   switches.
2. **Session/blind-spot gallery** — when a person leaves all camera views, keep their
   identity in an "inactive" gallery with a **timeout (start ~5 min for retail, tune)**.
   On reappearance, re-associate by appearance (and face if visible) within the window.
   Too long → different people merge; too short → same person gets a new ID. (Mirrors
   NVIDIA Metropolis MTMC target / peer-target re-association.)
3. **Cross-day Faiss gallery** — face embeddings persist indefinitely (subject to the
   retention policy in the privacy section). This is what counts repeat visits across days.

Honest edge case: a shopper who enters, is enrolled, then **leaves through an unmonitored
exit** simply stops appearing — their session closes on timeout. If they return another day
**and pass the entrance camera**, the face gallery recognises them. If they avoid the
entrance camera, they look new. There is no way around this without a face capture.

## Staff vs customer
**Enrollment-based face exclusion is the primary method** (robust, works without uniforms):
enroll each staff member's face once into a dedicated `staff` partition of the Faiss gallery;
on entrance, a staff-gallery match tags them staff and excludes them from customer analytics.
Frequency/dwell heuristics (staff appear many times daily, dwell longer, use back-of-house
paths) are a secondary backstop, not the primary signal.

## Licensing — what's locked vs open (resolve before any PAID deployment)
**Clean / locked:** RF-DETR (Apache), AuraFace recognition (commercial), Faiss (MIT).
**Open items to resolve when production starts:**
1. **Face detector weights.** SCRFD *code* is MIT, but InsightFace's released SCRFD
   *weights* are non-commercial like the rest of their zoo. For a fully clean stack, pair
   AuraFace with a permissively-licensed detector (**YuNet / OpenCV Zoo**, or **MediaPipe**,
   Apache-2.0) or cover detection under the InsightFace commercial license. Verify what
   detector AuraFace bundles.
2. **Tracker implementation.** The BoT-SORT *algorithm* is unencumbered, but the popular
   *implementations* (Ultralytics, BoxMOT) are **AGPL-3.0** — unacceptable for a closed
   commercial product. Resolve by a clean-room association implementation (Kalman + IoU +
   cosine over the SOLIDER embeddings — not much code) or a commercial Ultralytics license.
3. **Re-ID training data.** SOLIDER code is MIT, but verify LUPerson and any Market-1501 /
   MSMT17 fine-tuning terms for commercial use. **Clean fallback: NVIDIA TAO
   ReIdentificationNet** (commercially licensed, ONNX-exportable, Metropolis-ready).
4. **Never ship MS1M / WebFace-trained weights commercially** (AdaFace/LVFace/InsightFace
   default zoo) — academic-only datasets. This is the rule that gates all face-model choices.

## Honest limits (tell the client these; build confidence scores accordingly)
- **Clothes-changing across days** — only the face bridges days, and only with a usable
  frontal capture. Even SOTA clothes-changing Re-ID drops sharply.
- **Crowds / occlusion** — body Re-ID degrades in dense crowds; occluded-Re-ID tops out
  ~70% Rank-1, not 100%.
- **Gallery scaling** — false-match rate rises as the gallery grows into tens of thousands;
  raise the face threshold as you scale, add a second factor.
- **No model is 100%.** Show confidence scores everywhere; never assert certainty. Same
  discipline as the demo's watchlist ("verify, not identify"). This is also legal protection.
- **Privacy/legal** — storing face templates for cross-day recall is biometric data. Build
  in consent, a retention/auto-purge policy, on-prem-only processing, and template
  protection from day one — not as an afterthought.

## Seams to preserve (consistency with PRODUCTION_DIRECTION.md)
- **Model abstraction** so AuraFace ↔ InsightFace (and RF-DETR variants) is a config swap.
- **Embedding store behind a clean interface** so Faiss → Milvus is a backend swap.
- **Fusion as its own layer** (not baked into the tracker) so the GEFF logic is testable.
- **Device-agnostic** (`cuda`/TensorRT in prod; the demo's `mps` path stays config-driven).
- **`analytics.json` schema stays the contract** — the dashboard must not need a rewrite.

## Operating rule
Do not build, install, or download any of this until explicitly instructed. When the pilot
is greenlit, implement in this order: (1) entrance enrollment + Faiss gallery, (2) RF-DETR +
SOLIDER Re-ID with the clean-room tracker, (3) GEFF late-fusion layer, (4) cross-day recall +
staff exclusion, (5) repoint the dashboard at the live API. Re-read this file and
`PRODUCTION_DIRECTION.md` at the start of any production session, and resolve the open
licensing items before charging a customer.
