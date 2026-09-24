# Working on Relish

Relish is a Quest mixed-reality cooking prototype. Moving, deforming food in
the cook's hands is the primary case, including crossing and hand occlusion.

- Read `SPECS.md` for requirements, `PLAN.md` for ordered work, and the newest
  entry in `DEVLOG.md` for progress. `temp-devlog.md` contains historical results.
- Unity lives in `unity/`; PC perception lives in `perception/`.
  There is no current remote-mentor/audio/FastAPI hub to implement.
- Keep inference on the PC. World anchoring and hand joints complement food
  perception; they do not replace it. Do not reject food just for overlapping hands.
- Preserve existing uncommitted work. Keep changes small, using the current
  start_stream/track_frame adapters and standard-library tools where possible.
- Use `perception/.venv/Scripts/python.exe` on Windows. From `perception/`, run
  `.venv/Scripts/python.exe -m unittest discover -s tests -v` for regression checks.
  These tests do not establish tracking quality; GPU smokes and replay are separate.
- Compare the same clip, settings and GPU; record revision/dirty state and retain
  results. Distinguish request throughput, total processing and capture-to-display
  age. Empty output, ID totals and synthetic translation are not quality metrics.
- Run one GPU experiment at a time. Do not download models or rebuild TensorRT
  engines for routine unit checks. Engine and compile artifacts are machine-specific.
- Record measurements and limitations in `DEVLOG.md`; update `PLAN.md` as work
  completes. Keep settled requirements separate from research ideas.
