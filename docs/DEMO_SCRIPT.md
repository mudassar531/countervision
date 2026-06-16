# DEMO_SCRIPT.md — CounterVision 3-minute walkthrough

> Audience: a retail prospect. Outcome we want: *"Yes, run this on my
> store."*
> Throughline: **your cameras turned into decisions — imagine this on
> your store's entrance and till.**

## Setup checklist (do this 5 minutes before the meeting)

1. `make demo-quick` — re-aggregates the analytics + serves the static
   dashboard at <http://localhost:3000>.
2. Open <http://localhost:3000> in your laptop browser, full-screen the
   tab, set zoom to 110 %.
3. Pre-click the **`camera-3`** tab in the *Per-area heatmap hero*
   panel — that's the visual everyone remembers (two bright hot spots
   on the seated customers).
4. Pre-click **`camera-5`** tab in the *Per-area detail* panel — scroll
   so the **`P006` thumbnail** is visible (this is the demo's killer
   moment: "7 merged ids").
5. Have a watchlist alert ready in the *Alerts* panel by running
   `python pipeline/main.py --seed-watchlist camera-5 P006
   staff_lead_demo` and then `make aggregate` before the meeting.
6. Mute your laptop — the annotated mp4 has no audio but some browsers
   warn anyway.

## 0:00 – 0:30  Hook

> **"What you're looking at is your standard CCTV feed turned into
> decisions. Same cameras. Same NVR. No new sensors. We added
> software."**

Hold on the *Branded Overview* hero for 8–10 seconds. Read the capture
window aloud:

> *"This is 9 minutes of footage from 3 cameras — recorded on June 7th
> and 8th 2026 — processed on a laptop in about 25 minutes. In your
> store this would run continuously on the same hardware you already
> have."*

The throughline lands once: **"Cameras you already own, turned into
the kind of report you currently pay a consultant to write."**

## 0:30 – 1:30  The reliable numbers (lead with these)

Move to the *Headline KPIs* row. Hit three numbers in order — each one
gets a sentence:

1. **`Avg dwell — store · 73.9 s`** (the navy accent card on the left).
   > *"This is how long the average customer actually spends in front
   > of a display. We compute it by **union-merging every camera
   > fragment a person produced** — so the number doesn't lie when
   > someone walks behind a column."*

2. **`Store-wide unique visitors · 13`** (the other navy card).
   > *"That's down from **16 raw per-camera counts**. Face
   > de-duplication found **3 people who appeared in more than one
   > area** in the captured window. We won't double-count Olivia just
   > because she was seen in Skincare and Consultation."*

3. **The `P006` visitor card** — scroll/click to the *Per-area detail*
   panel, `camera-5` tab.
   > *"This is the moment I want to land. **`P006` is one person**.
   > Phase-1 tracking fragmented them across **7 separate tracker
   > IDs** because of occlusion. Face matching merged the seven back
   > to one visitor with **32 seconds of authoritative dwell**. This
   > is the difference between analytics that's almost useful and
   > analytics that's actually trustworthy."*

Pivot to the *Per-area heatmap hero* — already on `camera-3`.

> *"And this is what 'where people actually stand' looks like in a
> 3-minute window. Two hot spots, two customers, exactly where they
> were sitting. Imagine this same picture on your skincare aisle, your
> till queue, your fragrance counter. **You'd know which display
> earns its rent.**"*

(60 seconds; this is the heart of the demo.)

## 1:30 – 2:15  The hedged capability (cross-camera + watchlist)

Scroll to the *Cross-camera presence* panel.

> *"Now look at the copy at the top of this panel — we deliberately
> render it with caveats. **'Repeat presence across the captured
> period, not a single continuous trip.'** The recordings in this
> footage don't overlap in time, so a face match means we saw the same
> person in two areas across the captured window — not that they
> walked a route. In a live store with overlapping camera windows,
> this becomes a real customer journey. We never invent links."*

> *"Three high-confidence pairs were found — that's where the
> store-wide 13 came from."*

Scroll to the *Alerts* panel.

> *"Every alert says **'Please verify before acting.'** Each one
> carries the cosine-similarity score. **This is a verification
> prompt, not an identification.** That's the difference between an
> operational tool and a liability."*

(45 seconds; the buyer needs to know we're not naïve about face data.)

## 2:15 – 2:45  Locked KPIs — the integration runway

Scroll back up to the KPI row and point at the **Unlock with POS
integration** cards.

> *"Conversion rate. Revenue uplift. We deliberately did not
> fabricate those numbers from face count times a constant. They're
> **locked**, with explicit 'data not available' badges. That's
> integrity — and it's also the **next conversation**."*

> *"**A one-day integration with your till system** unlocks real
> conversion rates per area, revenue per second of dwell, recommended
> product placements tied to actual sales. The same Phase 5
> aggregator joins on transaction timestamps. We've built the harness;
> you bring the till feed."*

(30 seconds; turns "I don't have a number for that" into "here's why
that's worth a follow-up call.")

## 2:45 – 3:00  Close

Scroll to the *Plain-English insights* panel (5 cards at the bottom).

> *"Five insights, all tied to the reliable numbers — none built on
> the cross-camera link or near-zero footfall."*

Click the headline insight: *Areas with the deepest engagement are the
staffing priority.*

> *"That's what 9 minutes of footage and a laptop produced. Imagine
> this exact dashboard, but at your **store's entrance** counting real
> footfall, at your **till** linking dwell to spend, and at your
> **consultation desks** measuring engagement. Same cameras. Same
> hardware. Just decisions you can act on by next Tuesday."*

Pause. Then:

> *"What would you want it to tell you first?"*

(That last question is the entire pitch — it gets them describing the
problem they'd want CounterVision to solve, in their own words.)

## Q&A — anticipated questions

| Question | Answer |
|---|---|
| **"How accurate is the face matching?"** | We use buffalo_l (SCRFD + ArcFace, the recognition model is ResNet50 trained on WebFace600K). For *clustering inside a single camera* we use cosine ≥ **0.32** — favours recall, then we use face linking to merge tracker-ID fragments. For *cross-camera de-dup* we use cosine ≥ **0.50** — a deliberately higher bar because a false cross-camera merge is a worse demo failure than a missed match. Both thresholds are in `config.yaml`. We can show you the tuning. |
| **"What hardware do I need?"** | The pipeline runs on a MacBook M2 Pro. For a live store, an NVIDIA T4 / RTX 4060 with 8 GB VRAM handles 4 × 1080p cameras at 25 fps in real time. Detection (YOLO26) runs ~10× faster on CUDA than the MPS we used here. |
| **"What about privacy?"** | We store **only embeddings + thumbnails** — never a raw-face database. Identity is fully toggleable (`identity.enabled: false` in `config.yaml`). Watchlist images stay on disk in `./watchlist/` and are gitignored. For UK GDPR / DPIA compliance you'd want signage, retention schedules, and an opt-out flow — we can advise. |
| **"What's not in this demo?"** | Anything that requires data we don't have. Real conversion rate needs your POS. Weather correlation needs an API. Quantified staffing needs payroll. The dashboard flags those as **Locked → Unlock with integration** rather than inventing numbers. |
| **"What did you build the dashboard with?"** | Next.js 16 (App Router, statically exported), React 19, Tailwind v4, shadcn/ui, Recharts 3. The dashboard reads **only** `analytics.json` — there's no Node server in the live demo path. It can be embedded in your existing BI portal or served from any CDN. |
| **"How long to deploy in our store?"** | Day 1: discover cameras + draw entry lines via `--draw-zones`. Day 2: tune face thresholds on a half-day's footage. Day 3: connect POS + run for a week. Day 8: present findings. |
| **"AGPL / licensing?"** | The detection model (YOLO26) is AGPL-3.0 — fine for this demo. For a shipped commercial product we'd swap to **RF-DETR Nano–Large (Apache-2.0)** or **YOLOX (Apache-2.0)**, both ONNX/CoreML-exportable. InsightFace pretrained weights are research-only — for production we'd license commercial face weights. None of this changes the dashboard. |

## Anti-demo failure modes (don't do these)

* **Don't claim more than the numbers show.** If footfall says "3"
  because the auto-entry-line missed the action, say so. The KPI card
  already labels it `hedged`.
* **Don't tell a journey story for the cross-camera panel.** Use the
  `presence_note` framing every time. The dashboard already says
  "repeat presence, not a single continuous trip"; reinforce that.
* **Don't fight the `Locked` cards.** They're a feature, not an
  apology. They become the second meeting.
* **Don't show the annotated mp4 until you've delivered the reliable
  numbers.** People glue to motion and stop listening. The mp4 is the
  proof; the numbers are the pitch.

## Run order for an offline / no-internet demo

```bash
make demo-quick         # 2 s aggregate + 8 s build + serve at :3000
open http://localhost:3000
```

If you have time to regenerate everything from raw footage in front of
the client (or want to be honest you're not faking it):

```bash
make pipeline           # ~25 min on M2 Pro — all 5 pipeline phases
make dashboard          # ~10 s static build + serve
```

**Zero live inference in the demo path** — the dashboard reads only
the pre-rendered `analytics.json` and the static heatmaps / mp4s.
Nothing in the room depends on a model running.
