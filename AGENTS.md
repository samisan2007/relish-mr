# Working on Relish

Relish is a Quest mixed-reality cooking prototype. Moving, deforming food in
the cook's hands is the primary case, including crossing and hand occlusion.

- Read `SPECS.md` for requirements, `PLAN.md` for ordered work, and the newest
  entry in `DEVLOG.md` for progress. Older DEVLOG entries are history: later
  entries supersede them.
- Unity lives in `unity/`; PC perception lives in `perception/`.
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
- After every completed batch or task, update the relevant Markdown documentation
  immediately, even when more work remains. This is a standing user instruction;
  do not defer it until the end of a session or until usage/context is nearly exhausted.
- Record completed work, checks/results, limitations, artifact paths, unresolved
  issues and the exact next step in `DEVLOG.md`; keep `PLAN.md` current. Include
  commit/push status in handoffs so uncommitted work is not mistaken for a saved
  Git checkpoint. Update setup READMEs when commands or prerequisites change.
  Keep settled requirements in `SPECS.md` separate from research ideas.
