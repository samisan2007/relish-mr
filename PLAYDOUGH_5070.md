# Quest play-dough replay on RTX 5070

2026-09-25, `experiment` at `8c9c250`, RTX 5070 12 GB. This repeats the [RTX 3080 protocol](PLAYDOUGH_EVALUATION.md) on the local Quest clips. The replay script is [`perception/playdough_replay.py`](perception/playdough_replay.py); local masks, overlays, per-frame timings, source hashes and manifests are under `perception/runs/playdough-5070-20260925/`. `results-summary.csv` lists every run. Tests were sequential, one GPU job at a time, at the original 1024 x 1024 input resolution, source-frame stride 3, with prompt `red dough`. Each window starts a new tracker. Warm rates exclude the first ten requests and do not include decoding, drawing or file writes. They are not Quest capture-to-display rates. The local replay saves packed per-frame masks; its wall-clock totals are not directly comparable to the 3080 harness's totals.

| Backend | Handling 5070 / 3080 requests/s | Pieces 5070 / 3080 requests/s | Return 5070 / 3080 requests/s |
|---|---:|---:|---:|
| SAM3 image → EdgeTAM hybrid | 6.57 / 4.83 | 5.61 / 4.13 | 9.19 / 8.03 |
| SAM3 → YOLOE | 24.45 / 18.25 | 26.08 / 18.10 | 14.38 / 11.14 |
| SAM3 video FP16 | 2.03 / 2.15 | 1.27 / 1.47 | Aborted / aborted |
| DARTF native FP16 | 2.11 / 1.91 | 1.83 / 1.64 | 2.17 / 2.05 |
| SAM 3.1 | 1.84 / 1.72 | 1.77 / 1.68 | 1.97 / 1.76 |
| SAM 3.1 compiled | 2.19 / 2.09 | 2.01 / 1.99 | 2.33 / 2.16 |

All six backends completed setup (58 requests), handling (120) and pieces (120). Five completed return (240). SAM3 video stopped after 151 return requests when sampled device memory reached 11,710 MB. The last request was 846 ms; this is a preventive memory stop, **not** a measured 5070 latency collapse. The 3080 run stopped after 208 requests when memory approached 11.9 GB and request latency reached 28.5 s. This PC had roughly 2 GB of desktop GPU memory in use before inference. The sampled peak can miss spikes between ten-request samples. DARTF FAST was not run: its existing worker and TensorRT engines are explicitly SM86/RTX 3080 only, and there is no 5070 FAST build.

The hybrid's handling/pieces request p95 is 374/414 ms, so its keyframe spikes remain despite the higher average rate. YOLOE's p95 is 43 ms in both loaded windows, but it misses held pieces. The full p95 and sampled memory numbers are in `results-summary.csv`.

The setup window is mostly an empty or distant view. In particular, YOLOE skipped many requests during its retry cooldown, so its setup-window rate is not a useful object-tracking speed. Return-window rates also include long empty stretches; handling and pieces are the better loaded comparisons.

## Matched-frame visual review

- Handling source frame 1140: the hybrid, SAM3 video, DARTF and SAM 3.1 cover the three selected pieces; YOLOE covers only one. See local `compare-handling-1140.jpg`.
- Pieces source frame 840: the hybrid and DARTF cover the bowl, board and held dough; YOLOE misses the held material. Both SAM 3.1 modes spread one ID across widely separated places. At frame 960 SAM3 video produces a large background mask. See `compare-pieces-840.jpg` and `compare-pieces-960.jpg`.
- Return source frame 1257: the hybrid misses the left-palm dough; YOLOE, DARTF and both SAM 3.1 modes cover it. On stove-only frame 780 YOLOE emits a false mask at the right edge; the other completed backends do not. See `compare-return-1257.jpg` and `compare-return-780.jpg`.

These are sparse visual checks, not annotated recall or physical-ID accuracy. The event-label sheet at `perception/annotations/playdough-events.csv` still has only its header. The 5070 is faster for the hybrid, but it does not remove the quality failures or reach 8–10 requests/s while the dough is handled.

## Shared-seed propagation

`perception/candidates-local/cases/20260925-113650-046188` contains 120 **consecutive** frames from clip 2 beginning at source frame 1140, seeded with the same three SAM3 proposals as the 3080 comparison. The seed overlay was inspected. Two passes per tracker completed; results are under `perception/candidates-local/runs/20260925-*`.

| Tracker | 5070 mean ms, passes 1–2 | 3080 mean ms, passes 1–2 | Object 1 empty frames | Object 3 first → last mask area |
|---|---:|---:|---:|---:|
| Native EdgeTAM | 36.2–36.4 | 47.5–48.5 | 35 | 3303 → 7103 px |
| Transformers EdgeTAM | 62.2–62.6 | 81.0–86.7 | 19 | 3303 → 7158 px |
| SAM 2.1 tiny | 80.4–83.7 | 81.1–81.2 | 45 | 3303 → 3730 px |

The bowl-piece mask doubles in area with both EdgeTAM implementations, reproducing the nearby-piece absorption. SAM 2.1 tiny keeps its area close to the seed but also loses the left-palm piece. These are fixed-seed propagation runs with no re-detection; they do not measure the full hybrid pipeline.

## Prompt and validation checks

The local `prompt_probe.py` repeated the fourteen prompts on clip-2 frames 228, 913, 1142 and 1599, plus stove-only clip-3 frame 780 at threshold 0.4. `red dough` yielded 4, 3, 3, 5 and 0 masks respectively, identical to the 3080 run. `food` returned four false proposals on the stove. Most counts match; `play doh` returned two masks on frame 1142 here versus none in the 3080 report. Raw scores are in `prompt-probe.json`.

The repository's 60 regression tests pass. Neither they nor these sparse frame checks establish product acceptance. The next useful step is human event labels for loss, return, touching and splitting, followed by the planned same-frame association and occluded-ID experiments. A paced replay and live Quest capture-to-display measurement remain separate work.

Git: this report and `perception/playdough_replay.py` are committed and pushed on `experiment`. Video, checkpoint and full replay outputs remain ignored local artifacts.
