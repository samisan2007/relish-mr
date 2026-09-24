# Relish — next steps

Updated 2026-09-24. Requirements: [SPECS.md](SPECS.md).
Measurements and completed work: [DEVLOG.md](DEVLOG.md).

## Direction

Keep SAM3 image -> EdgeTAM as the leading experiment, SAM3 video as a quality
reference, and YOLOE hybrids as fast alternatives. No production backend is chosen.
The target remains useful masks and identities at 8-10 fps while food is handled.

YOLOE preserves IDs between detections; installing new exemplars resets its tracker.
EdgeTAM shares image features but runs object-specific tracking sequentially.
Batching may improve GPU utilization, not make work independent of object count.

## 0. Baseline and small fixes

Implemented in this working tree:
- Replace obsolete assistant instructions and reconcile the active spec.
- Pin Transformers, Ultralytics and timm to the installed adapter-compatible
  versions. Other dependencies are not fully locked yet.
- YOLOE: ground immediately at startup and retry on the next request after a new
  loss. Continuing failures wait 500 ms after an attempt finishes. Existing
  exemplars keep searching between retries; before the first seed, YOLOE cannot
  search. `retry_seconds=0` restores retry-every-request for comparison.
- Replay: unique directories, clip/source hashes, settings, host packages, GPU,
  Git state when available, completion/failure state and p95 timing. Clean up the
  camera, writer and worker on failure as well as success.

Checkpoint the reviewed tree before comparing changes. Dirty-state flags and hashes
identify runs but cannot reconstruct uncommitted code. The manifest does not yet
capture every resolved checkpoint revision or worker package; record those for
model comparisons. Existing artifacts without provenance remain historical evidence.

## 1. Lost-object diagnosis — next implementation task

Preserve baseline replays of `../Media/pen_test_vid.mp4` and
`../Media/pen_vid_test_x3.mp4`. Annotate failure windows by physical object, not by
SAM3 detections: the detector is not ground truth.

Record why a track disappeared: detector miss, tracker absence, keyframe hiding,
duplicate suppression, failed association or compaction. Test one change at a time:
1. Match detections and propagated tracks at the same frame time. Current matching
   uses the previous processed frame's masks before propagation.
2. Test a bounded occluded state: retain identity without displaying a stale mask.
   Currently an absent object drops out of `last_masks`; recovery may mint a new ID.
3. Test a short grace period for a blurred keyframe without restoring ghosts.
   `retire_after=1` currently hides a track after one unsupported keyframe.
4. Test an early keyframe after a visible track disappears, with a cooldown to
   prevent repeated expensive detection on every empty frame.

Acceptance: improve annotated failure windows without introducing swaps, duplicates,
merged objects or persistent ghosts elsewhere in either clip. Report recovery time,
mask quality and ID switches alongside throughput/p95. ID totals alone do not pass.
Keep EdgeTAM execution unchanged during this diagnosis.

## 2. Timing and retry-policy validation

Compare `retry_seconds=0` and 0.5 on initial empty views, loss, return and a false
exemplar. Include SAM 3.1's stateful seeder: sparse inputs may affect its recovery.
The 500 ms delay is provisional. Measure recovery latency and GPU duty cycle;
faster empty frames do not establish successful tracking.

Keep sequential replay for quality. Add timestamp-paced replay with a latest-frame
queue for live behavior, then camera capture timestamps and a one-frame background
capture buffer. Log dropped frames and frame age; fresh frames have larger motion
gaps, which can hurt association. Ten processed frames at 10 fps is about one second.

Report request p50/p95, total processing, capture-to-display age, recovery delay,
visible-object coverage and VRAM. Separate startup from warm processing. Run at least
10 minutes to check sustained memory. SAM3 video's OOM at frame 942 remains open;
bound its memory before relying on it for long live use. Choose the display-latency
budget from the first Quest measurements; FPS is not sufficient.

## 3. Profile, then consider EdgeTAM batching

Use PyTorch Profiler with 1, 3 and about 12 visible objects. Separate shared image
features, per-object memory/decoder work, CPU mask operations and transfers.
Do not reimplement the existing image-feature cache.

If object-specific work dominates, batch compatible ordinary propagation steps.
Group by memory/token and object-pointer layouts, not just memory length. Keep
newly seeded/reseeded and incompatible objects sequential first. Preserve each
object's state and output mapping.

Acceptance: compare masks, scores, identities and subsequent memory behavior through
reseeding, occlusion and compaction against sequential execution. Allow small numeric
differences; investigate changed tracking decisions. Record speed, p95 and VRAM at
each object count on each GPU. Keep batching only if it materially helps without
quality regression. Pin any private APIs it uses.

Keep SAM synchronous for this experiment. Background grounding comes later only if
stalls remain material; it requires timestamps, stale-result handling and propagating
delayed masks to now. Two workers sharing one GPU can contend.

## 4. Food evaluation and the Quest loop

Collect repeatable food footage before selecting a backend: kneading/shape change,
crossing pieces, full hand occlusion, touching pieces and a plate with many portions.
Hand-check selected masks and identities; keep separate tuning and evaluation clips.
Pens remain regression fixtures, not the product acceptance set.

Build the smallest Quest camera -> PC -> widget loop using existing Meta samples.
Carry frame/session IDs, capture timestamps, capture-time camera pose and intrinsics;
link returned results to that frame. Reject stale results, clear widgets across
session resets and measure display age. Start this alongside food evaluation once
the baseline is stable; do not wait for every model experiment.

Test head motion on stationary food, then handled food. Correct capture pose removes
head-pose mismatch, not food motion during inference. Hand joints are hints for
association/occlusion, never a blanket mask-overlap rejection. Validate depth,
calibration and size stability before reporting metric portions.

## 5. Conditional model and product experiments

Earn a UI integration with a bounded offline comparison first:
- [Official EdgeTAM](https://github.com/facebookresearch/EdgeTAM): compare the
  implementation using identical seeds; mobile FPS does not predict our performance.
- [SAM 2.1 tiny](https://github.com/facebookresearch/sam2): try propagation only
  if EdgeTAM remains weak after association fixes.
- [EfficientSAM3 EV-M](https://github.com/SimonZeng7108/efficientsam3): test food-noun
  recall and image masks before trying it as a cheaper keyframe detector.
- Keep DARTF available; further engine work needs evidence from food footage.
- [EOVSAM](https://arxiv.org/abs/2608.02284) is lower priority: its headline gain
  concerns large vocabularies, not our single-prompt temporal workload.

For portion UX, first explore relative size when sufficiently visible, showing
uncertainty during occlusion. This is a proposal, not a grams/calories commitment.
Pixel bounding-box diameter is not calibrated physical size. Defer nutrition/action
VLMs, stereo reconstruction and on-device inference until the core loop is useful.
