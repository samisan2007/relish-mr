# Real-time food tracking — findings, progress, options

## Review correction — 2026-09-07

The observations below predate two implementation fixes: YOLOE received RGB
NumPy frames where it expects BGR, and Hybrid reapplied seed boxes to each
new frame without keeping the seed image. Hybrid now preserves that image,
extracts visual embeddings from it once, and reuses them for later frames.
The earlier explanation of motion failures as an architectural ceiling is
therefore provisional; live rotation and occlusion need retesting with the
corrected pipeline. The webcam's tuned config now also forwards both tuning
options and reapplies the global cuDNN flag when switching cached configs.

Validation after the fixes: seven automated regressions pass; the tuned
SAM3 config detects 13 meatballs with both tuning options applied. A real
SAM3-to-YOLOE run on a translated local image seeds 13 instances and detects
10 after translation, keeps YOLOE IDs across the three follow-up frames,
extracts visual embeddings only once, and installs a fresh reference on
restart. This synthetic translation check does not validate live pose
changes, occlusion, or sustained performance. Run it from `perception/` with
`python tests/smoke_hybrid.py <image> <prompt>` using the project venv.

The remaining sections preserve the earlier findings and proposed directions.

*(Temp file at repo root — a devlog.md exists on another machine and will be
merged in later. Kept separate and tracked so it survives a pull/merge,
rather than living in the gitignored Documentation/ folder.)*

## What's built

- SAM3 image tester, SAM3 video tracker (file + live webcam tabs).
- `bench_realtime.py` — CLI benchmark, replays one captured clip through several configs for a fair comparison.
- Webcam tab: **Model** dropdown (SAM3 / YOLOE text-prompt / Hybrid), SAM3 **config** dropdown, and a run log of the last 8 runs.
- `yoloe_runner.py` — YOLOE text-prompt tracker, and a Hybrid tracker (SAM3 seeds once, YOLOE tracks after).

## Findings

**SAM3** — the only backend that's actually *reliable* (tracks through rotation/angle changes, understands niche food nouns like "meatball"). But:
- Per-frame cost **ramps for ~7-8 frames** as its memory bank fills, then plateaus — short benchmarks were measuring the ramp, not real speed. Fixed the benchmark's warmup to account for this.
- **Confirmed steady-state**: fp32 0.42 fps (2387ms/frame) vs fp16 (via `torch.autocast`) 0.96 fps (1038ms/frame) — fp16 is the right choice, ~2.3x faster.
- **No further speed lever survived testing**: resolution override breaks (hardcoded token-grid buffers), `num_maskmem` reduction breaks (tied to a fixed-shape learned weight), `torch.compile` is dead on Windows (no Triton), device_map/mask_size/cudnn.benchmark/conditioning-frame-cap all made no meaningful difference. **~1fps is the practical ceiling** with cheap levers.

**YOLOE (text prompt)** — fast (~100-500ms/frame) but its MobileCLIP vocabulary is too weak for specific food nouns ("meatball" maxes ~0.17 confidence even on the largest checkpoint). Not usable alone for this project's actual prompts.

**YOLOE (visual exemplar)** — excellent in isolation (0.9+ confidence given a good example box) but it's re-detection-by-similarity to a static snapshot, not real tracking with memory.

**Hybrid (SAM3 seeds once -> YOLOE tracks)** — works as code, but **live testing (moving pen, moving eye) showed it degrades to plain YOLOE's exact weakness** once seeded: loses the object on angle/pose change, because it inherited YOLOE's re-detection mechanism, not SAM3's temporal memory. Currently just reproduces YOLOE's ceiling with extra steps.

**Bugs fixed along the way**: SAM3 dtype crash (full weight-cast broke on real detections, fixed via autocast), ByteTrack `None`-ID crash in the run log, a `.gitignore` gap that almost committed a 599MB file, camera buffer lag (`CAP_PROP_BUFFERSIZE=1` — cheap fix applied, not yet confirmed it holds on Windows).

## The core tension

The only reliable backend (SAM3) is slow (~1fps). Every fast alternative tried breaks under motion because it lacks real temporal memory — it re-detects per frame by static similarity rather than tracking. That's a property of the model, not a config bug, so no amount of hybrid wiring around YOLOE fixes it by itself.

## Options

| # | Option | What changes | Tradeoff |
|---|--------|--------------|----------|
| 1 | **Periodic-refresh hybrid v2** | SAM3 re-grounds on a cadence (e.g. every 300-500ms) as authoritative correction; a classical CV tracker (CSRT/KCF) — motion/appearance continuity, not semantic matching — fills the gap between corrections | Structural fix, doesn't depend on YOLOE's weak spot at all. More to build; gives a box not a mask between corrections (approximate diameter); can drift/fail under occlusion |
| 2 | **Adaptive multi-exemplar Hybrid** | Keep the current Hybrid, but periodically add newly-confirmed detections as more visual exemplars so YOLOE's matching set covers more angles over time | Smallest code change. Still the same re-detection mechanism — may still fail on a genuinely novel pose; risk of polluting the exemplar set with a bad detection |
| 3 | **Accept ~1fps, design around it** | Use SAM3 alone at its real rate | No further engineering. Worth checking first whether ~1fps is an actual proven blocker for real cook/HMD movement speed, or an untested assumption |
| 4 | **Push SAM3's raw speed further** | Attempt ONNX/TensorRT export, or check for a smaller/distilled SAM3 checkpoint | Bigger, uncertain-payoff investment — every cheap lever is already exhausted |

**Option 1 vs 2, precisely**: Option 1 changes *what does the tracking* — swaps YOLOE's semantic re-detection for a fundamentally different algorithm (motion/appearance continuity) that doesn't care what the object is, so it doesn't break on unseen angles. Option 2 keeps the exact same mechanism and just gives it more reference photos — a mitigation, not a structural fix.

No direction has been chosen yet.
