```markdown
# FUTURE — Phase 2+ (Vision Grounding, RAM/SAM3 Notes)
**Last updated:** 2026-01-15
**Status:** Not required for Phase 1 MVP. Keep here to avoid scope creep.

---

## 1) Why this file exists
Phase 1 is audio-first with MR suggestions (task board, timers, checks) and does **not** depend on vision.

Phase 2 adds optional vision grounding to enable:
- object-aware MR overlays (e.g., highlight a bowl/pan/ingredient)
- spatial anchoring of instructions (attach a suggestion to the right place)

This file stores:
- your RAM setup progress
- a minimal “how we integrate vision later” contract
- optional SAM3 notes if/when you want segmentation

---

## 2) Grounding integration contract (non-negotiable)
To keep the architecture simple and resilient:

1) **Only ground after acceptance**
- Vision is invoked only when:
  - the cook accepts a suggestion, AND
  - the suggestion is marked `needs_grounding=true`.

2) **Never block the UX on vision**
- If grounding is unavailable or fails:
  - the suggestion remains a normal MR card (no highlight).

3) **No “mentor command UI” model**
- We are not building “mentor points/highlights” workflows as a default path.
- Vision grounding supports the cook-side MR layer, driven by accepted suggestions.

4) **Keep ML deps isolated**
- Vision models live in separate envs/services (do not pollute the Phase 1 hub env).

---

## 3) RAM service notes (Recognize Anything) — your Phase 2 building block
These notes reflect the RAM steps you started (and are safe to continue later).

### 3.1 Environment (Python 3.10 + CUDA 12.6 wheels)
```powershell
conda create -n relish-ram python=3.10 -y
conda activate relish-ram
python -m pip install -U pip

```

Install torch CUDA wheels (cu126):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```

### 3.2 Clone and install RAM

```powershell
cd C:\RELiSH\relish-mr\services\ram
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything

pip install -r requirements.txt
pip install -e .

```

### 3.3 Weights + tag embeddings

Create a predictable folder:

```powershell
mkdir C:\RELiSH\relish-mr\services\ram\weights
cd C:\RELiSH\relish-mr\services\ram\weights

```

Download checkpoint + embedding (example PowerShell pattern):

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth" -OutFile "ram_swin_large_14m.pth"
Invoke-WebRequest -Uri "https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_tag_embedding_class_4585.pth" -OutFile "ram_tag_embedding_class_4585.pth"

```

Expected:

```
services\ram\weights\
  ram_swin_large_14m.pth
  ram_tag_embedding_class_4585.pth

```

### 3.4 RAM quick test (static image)

Create: `services/ram/test_ram.py`

```python
import torch
from PIL import Image

from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

CKPT = r"C:\RELiSH\relish-mr\services\ram\weights\ram_swin_large_14m.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = ram(pretrained=CKPT, image_size=384, vit="swin_l")
model.eval().to(device)

img = Image.open(r"C:\RELiSH\relish-mr\services\ram\test_kitchen.jpg").convert("RGB")
transform = get_transform(image_size=384)

x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    tags = inference(x, model)

print("tags:", tags[0])

```

Run:

```powershell
conda activate relish-ram
cd C:\RELiSH\relish-mr\services\ram
python test_ram.py

```

### 3.5 RAM microservice stub (optional scaffolding)

Only do this when you actually want to call RAM from the hub.

Create: `services/ram/app.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="relish-ram")

class TagRequest(BaseModel):
    image_b64: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/tags")
def tags(req: TagRequest):
    # TODO: decode image and run RAM inference
    return {"tags": [], "note": "wire model later"}

```

Run:

```powershell
conda activate relish-ram
pip install fastapi uvicorn[standard]
uvicorn app:app --host 0.0.0.0 --port 8010

```

---

## 4) SAM3 notes (optional, only if you need segmentation)

If you decide you need pixel masks (not just tags), SAM3 is a candidate.

**Strong guideline:** keep it in a separate environment from RAM and from the hub.

High-level skeleton (do later):

- env: `relish-sam3` (Python 3.12)
- run a simple image segmentation test
- expose a stub service on port 8011:
    - `GET /health`
    - `POST /segment` → `{ masks:[], boxes:[] }` until wired

---

## 5) How Phase 2 connects back to Phase 1 (no architectural churn)

Phase 1 hub stays the coordinator.

Phase 2 adds optional calls from hub → vision services:

- hub receives accepted suggestion with `needs_grounding=true`
- hub requests grounding from vision service
- hub emits an `overlay_render` event to the Quest
- if grounding fails, hub emits nothing and the card remains ungrounded

This preserves the Phase 1 “always usable” MR experience.