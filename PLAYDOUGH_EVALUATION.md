# Quest passthrough play-dough evaluation

2026-09-24, RTX 3080 12 GB. These four user-supplied recordings are the
representative handled-material evaluation set. They show kitchen activity from
Quest 3 passthrough, including rolling, kneading, handling and camera motion.

## Footage inventory

| File under `Media/` | Frames | Duration | Resolution |
|---|---:|---:|---|
| play-dough_01.mp4 | 172 | 5.74 s | 1024 x 1024 |
| play-dough_02.mp4 | 2513 | 83.76 s | 1024 x 1024 |
| play-dough_03.mp4 | 1975 | 65.83 s | 1024 x 1024 |
| play-dough_04.mp4 | 3102 | 103.40 s | 1024 x 1024 |

All are approximately 30 fps, totalling 258.74 seconds. Initial contact-sheet
inspection: clip 1 is a short distant/setup view; clip 2 includes handling and
kneading; clip 3 includes looking away toward the stove and returning; clip 4
includes forming and arranging several small pieces. The recorded Quest menu,
hand outlines/rays and darkened scene are part of the input. No brightness
correction or cropping is applied to baseline inference.

Local artifacts: `perception/runs/playdough-20260924/`. `inventory.json` records
source hashes and exact frame rates; `play-dough_0*-contact.jpg` gives 12 evenly
spaced views per recording. Original videos and experiment outputs stay local.

## Protocol

- Use clip 2 for a small noun-prompt probe, then freeze the chosen prompt and
  thresholds before comparing clips 3 and 4.
- Compare identical frames and resolution through existing stream adapters.
  Include the current SAM3 image -> EdgeTAM hybrid, SAM3 video reference and
  SAM3 -> YOLOE fast alternative; test the available DARTF/SAM3.1 workers too.
- Inspect deformation, visible dough missed under hands, mask leakage onto
  hands/background, touching-piece merges, loss and return. Stable IDs and
  nonempty output alone are not quality scores.
- Record warm request mean/p50/p95 separately from startup and total processing.
  Retain overlays, masks, per-frame timings, source-frame mapping and environment.
- This is sequential replay, not a live Quest latency or sustained-memory test.
  Formal mask IoU and physical-ID accuracy require independent annotations.

## Prompt probe and frozen windows

SAM3 image FP16, threshold 0.4, clip 2 frames 228, 913, 1142 and 1599:
tested `play dough`, `dough`, `red dough` and `ball`. At frame 228 only `red dough`
returned masks (four); the other three returned none. At frame 1142, `dough`
missed all three visible pieces while the other prompts found them. Inspection
also found a duplicate proposal on the bowl piece for `red dough` at frame 1599.
The prompt is **`red dough`** for all following backends, selected on clip 2 only.
These counts are diagnostics, not independently annotated recall.
Raw scores/boxes and comparison images: `prompt-probe.json`, `prompt-frame-*.jpg`.

**Wider probe (`prompt_probe2.py`):** 14 prompts on the same four frames plus a
no-dough stove view (clip 3 frame 780), same threshold. Masks returned:

| Prompt | 228 | 913 | 1142 | 1599 | stove 780 |
|---|---:|---:|---:|---:|---:|
| red dough | 4 | 3 | 3 | 5 | 0 |
| food | 5 | 4 | 3 | 4 | **4** |
| clay | 4 | 0 | 3 | 4 | 0 |
| red ball | 0 | 2 | 3 | 5 | 0 |
| playdough / play dough | 0 | 0 | 3 | 4 | 0 |
| play-dough | 0 | 0 | 3 | 0 | 0 |
| dough | 0 | 0 | 0 | 4 | 0 |
| ball, dough ball | 0 | 0 | 3 / 0 | 4 | 0 |
| red clay, modeling clay | 0 | 0 | 3 | 0 | 0 |
| play doh, meatball | 0 | 0 | 0 | 0 | 0 |

`red dough` is the only prompt that finds dough on all four frames without
firing on the stove view. `food` also finds the dough but masks the pot and other
items on the stove view, so it is not usable as a dough prompt. The product
name (`play-dough`, any spelling) is unreliable: the darkened, low-contrast
views at 228/913 defeat it. Counts are still not recall; the sheets are
`prompt2-*.jpg`, raw scores `prompt-probe2.json`. The frozen prompt stays.

Frozen input windows preserve 1024x1024 pixels, taking every third source frame
(approximately 10 source samples/s). No brightness adjustment or crop:

| Window | Clip | Source frames (exclusive end) | Source time | Requests |
|---|---|---|---|---:|
| setup | 01 | 0:172:3 | 0–5.7 s | 58 |
| handling | 02 | 900:1260:3 | 30–42 s | 120 |
| return | 03 | 540:1260:3 | 18–42 s | 240 |
| pieces | 04 | 720:1080:3 | 24–36 s | 120 |

`cases.json` maps each lossless PNG to its source frame/time and SHA256.
The saved `replay.py` calls existing stream adapters and stores masks, videos,
frame CSVs and code/environment manifests. Backend defaults are retained:
hybrid detector 0.4/keep 0.2 and K=10 (one source second); YOLOE confidence 0.15;
SAM3.1 output threshold 0.5. Thresholds and internal resizing differ by backend;
this compares deployed configurations, not isolated architectures. FAST internally
resizes to 640x480. Each replay starts a fresh process/session; first 10 requests
are excluded from warm timing, but included in total processing.

## First completed batch: handling (clip 2, 30–42 seconds)

| Backend | Warm requests/s | Mean ms | p50 ms | p95 ms | Total processing s |
|---|---:|---:|---:|---:|---:|
| SAM3 image -> EdgeTAM | 4.83 | 207.2 | 161.4 | 509.6 | 43.1 |
| SAM3 video -> YOLOE | 18.25 | 54.8 | 48.3 | 52.2 | 22.1 |
| SAM3 video FP16 | 2.15 | 464.4 | 463.0 | 494.5 | 69.3 |

120 identical inputs per backend. Total includes model/session startup, frame
verification/decoding, mask writes, overlays and encoding; it is not live FPS.
The leading hybrid has expensive keyframe spikes. First visual review of source
frame 1140: EdgeTAM covers both palm-held pieces and the bowl piece; YOLOE covers
one palm-held piece and misses the other two. At frame 1257 the hybrid misses one
board piece, while YOLOE misses the bowl pieces. Fast requests alone do not pass.

Local `results/handling/<backend>/<timestamp>/` directories retain full overlays,
packed masks, frame timing/IDs and environment manifests. `comparison-handling.jpg`
contains matched review views. Sparse review points were placed on original frames
before inspecting results; they are assistant annotations, not human-validated masks.

## Installed worker checks: same handling window

| Backend | Warm requests/s | Mean ms | p50 ms | p95 ms | Total processing s |
|---|---:|---:|---:|---:|---:|
| DARTF FAST | 5.95 | 168.1 | 167.9 | 180.0 | 53.8 |
| DARTF native FP16 | 1.91 | 523.2 | 532.5 | 588.9 | 91.6 |
| SAM 3.1 | 1.72 | 580.7 | 586.7 | 613.8 | 130.7 |
| SAM 3.1 compiled | 2.09 | 477.9 | 483.7 | 506.7 | 134.3 |

FAST covers all three selected pieces at source frame 1140, with an extra small
mask beside the bowl. Its first output appears at request index 4 (0.4 nominal
source seconds after the window starts); this is not wall-clock startup latency.
Native FP16 also starts and completes successfully. Both SAM 3.1 modes complete;
compiled mode's first request takes 5.45 s, so its startup-inclusive total is not
lower. At source frame 1140, DARTF FP16 and both SAM 3.1 modes cover all three
selected pieces with no palm leakage at the two negative points (sparse check).

## Shared-seed propagation (candidate harness)

`perception/candidates-local/cases/playdough-20260924/20260924-174735-596375`:
120 **consecutive** clip-2 frames from source frame 1140 (about 4 s, no stride,
unlike the replay windows), `red dough`, three seeds: object 1 left-palm piece,
2 right-palm piece, 3 bowl piece (`seed-overlay.png`). No re-detection: this
isolates propagation. Two fresh passes each; ranges span both passes.

| Tracker | Mean ms | p95 ms | Peak GiB | Run (`candidates-local/runs/20260924-`) |
|---|---:|---:|---:|---|
| Native EdgeTAM + reshape fix | 47.5-48.5 | 52.6-54.3 | 0.54 | `181912-015668-edgetam` |
| Transformers EdgeTAM | 81.0-86.7 | 88.1-116.5 | 0.38 | `181948-978089-edgetam-hf` |
| SAM 2.1 tiny | 81.1-81.2 | 83.6-83.7 | 0.84 | `182037-625055-sam21tiny` |

In this window the palm pieces are put down: one on the board, one into the bowl
next to the existing bowl piece. `candidate-review.jpg` and
`candidate-agreement.json` (from `candidate_review.py`, pass 1) show:
- Object 2 (right palm -> board) is followed by all three; pairwise mask IoU
  averages 0.92-0.97.
- Object 3 (bowl piece) **absorbs the neighbouring piece** in both EdgeTAM
  implementations: its area roughly doubles (3303 -> ~7100 px) and it has two
  separate components in 46 (native) / 39 (HF) of 120 frames. SAM 2.1 tiny keeps
  it to one piece (3303 -> 3730 px, one multi-component frame).
- Object 1 (left-palm piece) is lost once it is set down: empty on 35 (native),
  19 (HF) and 45 (SAM 2.1) frames. Native EdgeTAM and SAM 2.1 later put it back
  on a piece; whether it is the right physical piece needs a human check.
- Native and Transformers EdgeTAM disagree on object 1 (mean IoU 0.34), so the
  two implementations do not make the same tracking decisions here.

Native EdgeTAM is about 1.7x faster than the Transformers version used in the
hybrid. SAM 2.1 tiny is as slow as Transformers EdgeTAM but was the only one
that did not merge touching pieces. Agreement is not accuracy; no ground truth.

## Multiple-piece baseline (clip 4, 24–36 seconds)

| Backend | Warm requests/s | Mean ms | p50 ms | p95 ms | Total processing s |
|---|---:|---:|---:|---:|---:|
| SAM3 image -> EdgeTAM | 4.13 | 242.4 | 220.0 | 548.8 | 45.7 |
| SAM3 video -> YOLOE | 18.10 | 55.2 | 55.7 | 61.2 | 22.4 |
| SAM3 video FP16 | 1.47 | 681.8 | 725.9 | 927.2 | 94.4 |
| DARTF FAST | 6.10 | 164.0 | 163.7 | 173.0 | 41.7 |
| DARTF native FP16 | 1.64 | 609.1 | 599.7 | 738.7 | 101.5 |
| SAM 3.1 | 1.68 | 594.6 | 598.2 | 639.8 | 132.5 |
| SAM 3.1 compiled | 1.99 | 502.2 | 504.9 | 548.2 | 137.4 |

All four workers also cover the four points at frame 840 (5-6 masks: the held
piece is being pinched in two). **SAM 3.1 (both modes) assigns one ID to two
separate places**: at frame 840 the bowl-piece mask has a second component about
300 px away (box 297x200 px for a ~3000 px mask); compiled mode repeats it at
frame 960. At source frame 840, the hybrid covers all four selected interior points (bowl,
two board balls, held material); YOLOE misses the held material. SAM3 video
produces a large background false mask at source frame 960. These are sparse
spot checks, not full mask-quality scores. Backend ID churn remains visible in
the CSVs; totals are not counted as identity errors because pieces can split and
rejoin and physical identities have not been annotated.

## Return-window memory failure

SAM3 video was stopped after 208/240 requests on clip 3. Device-wide sampled
memory reached about 11.9 GB (includes Windows/desktop usage); request time rose
from roughly 614 ms on indices 10–49 to a maximum **28.5 seconds**. This is a
memory-saturation/latency failure, not a caught CUDA OOM or a completed FPS result.
Per-frame masks, CSV and sampled JPEGs were retained; abrupt process termination
can leave the MP4 unfinished. The manifest explicitly marks this run aborted.

Hybrid and YOLOE completed all 240 return requests at 8.03 and 11.14 requests/s,
respectively. Empty parts of this window make those rates inappropriate as
handled-object speed estimates. At inspected source frame 1257, the hybrid
misses the left-palm piece. YOLOE covers it but also emits a false mask at the far-right bowl edge
on stove-only source frame 780. Full-frame inspection corrected an initial
cropped-preview interpretation; source frame 1020 has no output mask. Both findings need event-level annotation
before being turned into overall recovery or false-positive rates.

Workers on the same return window (all 240 requests completed, no memory alarm;
device memory stayed under 11 GB):

| Backend | Warm requests/s | Mean ms | p50 ms | p95 ms | Total processing s |
|---|---:|---:|---:|---:|---:|
| DARTF FAST | 6.52 | 153.3 | 153.2 | 166.9 | 55.9 |
| DARTF native FP16 | 2.05 | 486.7 | 478.4 | 707.0 | 148.3 |
| SAM 3.1 | 1.76 | 569.1 | 561.4 | 640.1 | 199.8 |
| SAM 3.1 compiled | 2.16 | 462.1 | 446.6 | 523.1 | 193.2 |

At frame 1257, DARTF FP16 and both SAM 3.1 modes cover the left-palm piece;
FAST misses it like the hybrid. None of the four puts a mask on the stove-only
frames 780 and 1020. Unlike SAM3 video, SAM 3.1 finished this window without
memory growth becoming a problem.

## Setup window (clip 1, 58 requests)

| Backend | Warm requests/s | Mean ms | p95 ms | Total s |
|---|---:|---:|---:|---:|
| SAM3 video -> YOLOE | 18.63 | 53.7 | 54.6 | 13.6 |
| SAM3 image -> EdgeTAM | 8.51 | 117.5 | 349.4 | 20.3 |
| DARTF FAST | 6.33 | 158.0 | 162.9 | 20.7 |
| SAM3 video FP16 | 2.92 | 341.9 | 356.7 | 30.1 |
| DARTF native FP16 | 2.55 | 392.3 | 443.8 | 46.4 |
| SAM 3.1 compiled | 2.34 | 426.6 | 448.1 | 92.8 |
| SAM 3.1 | 1.86 | 536.7 | 570.3 | 80.0 |

No review points were placed on this window; timing only.

## Summary so far

- Nothing reaches 8-10 requests/s with full coverage while dough is handled.
  YOLOE is the only one above 10 and it misses held pieces.
- FAST is steady at 6.0-6.5 requests/s (p95 <= 180 ms) and covered the pieces
  checked in handling/pieces, but missed the left-palm piece after return.
- The SAM3 image -> EdgeTAM hybrid covers well but has 300-550 ms keyframe
  spikes and misses the left-palm piece after return. Native EdgeTAM would cut
  its propagation cost; both EdgeTAM versions merge touching pieces.
- DARTF FP16 and SAM 3.1 cover the most checked points but run at 1.6-2.2/s.
  SAM 3.1 can spread one ID over two places. SAM3 video's memory grows until it
  collapses on the 24 s return window.
- Spot checks are sparse assistant-placed points. Event-level human annotation
  (hidden, lost, recovered, merged, split) is the next step before §1 changes.

No backend has passed product acceptance.

## Event annotation (human, pending)

Label sheets show raw frames only, brightened 1.8x for viewing, with each
thumbnail titled by its source frame. They are in
`perception/runs/playdough-20260924/annotation/` (`handling-p1..2`,
`return-p1..4`, `pieces-p1..2`), built by `annotation_sheets.py`.

Write labels in [`perception/annotations/playdough-events.csv`](perception/annotations/playdough-events.csv),
one row per event, using the source frame where it starts:

- `window`: `handling`, `return` or `pieces`.
- `piece`: any stable name you choose, such as `A`, `bowl-1` or `left-palm`.
  Use the same name for the same physical piece across its window.
- `event`: one of
  - `appears`: enters the view, or is formed
  - `hidden`: fully covered by a hand or out of view while still present
  - `visible`: back in view after `hidden`
  - `leaves`: gone from the scene for good
  - `merge`: two pieces become one; list both in `notes`
  - `split`: one piece becomes two; name the new piece in `notes`
  - `touch`: pieces are in contact but still separate
- `notes`: optional free text.

Example: `pieces,816,held,split,"pinched into held + held-2"`.

Model outputs are then scored against these events. A piece that is `hidden`
should keep its identity without a mask. After `visible` it should come back
under the same identity. A `touch` must not be merged into one mask.
