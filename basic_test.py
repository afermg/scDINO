"""Standalone smoke test for scDINO ViT.

Loads ``vit_small`` (patch_size=16, in_chans=3, num_classes=0) directly from the
local ``pyscripts/vision_transformer.py`` with random init and runs a forward
pass on a 1x3x224x224 random input. Should print ``torch.Size([1, 384])``.

Run from the repo root:

    nix develop --impure --command python basic_test.py
"""

import os
import sys

import numpy
import torch

# Make the local pyscripts package importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyscripts"))

import vision_transformer as vits  # noqa: E402

# CPU is plenty for a 1x3x224x224 smoke test on a randomly-initialised ViT-S/16
# and avoids contending with whatever else is on the box's GPUs.
device = torch.device("cpu")

model = vits.vit_small(patch_size=16, num_classes=0, in_chans=3).to(device).eval()

tile_size = 224
data = numpy.random.random_sample((1, 3, tile_size, tile_size))
torch_tensor = torch.from_numpy(data).float().to(device)

with torch.no_grad():
    out = model(torch_tensor)

print(out.shape)
