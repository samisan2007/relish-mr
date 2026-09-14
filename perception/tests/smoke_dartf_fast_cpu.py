"""Check upstream GPTQ compatibility on one image/block inside the FAST image.

Requires fast.py's download stage. Writes only /assets/cpu-smoke; these partial
caches cannot be mistaken for a completed FAST build or used by that build.
python /app/tests/smoke_dartf_fast_cpu.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path('/opt/dart/dartf')
out = Path('/assets/cpu-smoke')
out.mkdir(exist_ok=True)
source = (root / 'quant/gptq_actaware.py').read_text()
anchor = 'for b, layer in enumerate(bb.layers):'
assert source.count(anchor) == 1
script = out / 'gptq_smoke.py'
script.write_text(source.replace(anchor, 'for b, layer in enumerate(bb.layers[:1]):'))
ids = json.loads((root / 'calib/calib_ids.json').read_text())[:1]
(out / 'ids.json').write_text(json.dumps(ids))
os.environ.update(HF_HUB_OFFLINE='1', LQ_HEAD_ROT='1')
subprocess.run([sys.executable, str(script), '--images', '/assets/images',
    '--ids', str(out / 'ids.json'), '--out', str(out / 'gptq'), '--blocks', '0-0',
    '--act-blocks', '0-0', '--smooth', '0', '--hadamard', '--head-rot',
    '--act-scales', str(root / 'calib/act_scales.json'), '--threads', '8'], check=True)
with np.load(out / 'gptq_cache.npz', allow_pickle=False) as cache:
    for family, shape in {'qkv': (1024, 3072), 'proj': (1024, 1024),
                          'fc1': (1024, 4736), 'fc2': (4736, 1024)}.items():
        codes, scale = cache[f'block0.{family}::codes'], cache[f'block0.{family}::scale']
        assert codes.shape == shape and codes.dtype == np.int8
        assert scale.shape == (shape[1],) and np.isfinite(scale).all() and (scale > 0).all()
print('PASS: one-block CPU GPTQ smoke; full 32-block quantization still requires the build')
