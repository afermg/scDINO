"""Nahual server for scDINO.

Loads a DINO-style ViT (`vit_small`, `vit_base`, `vit_tiny`) from
`pyscripts/vision_transformer.py` and optionally a checkpoint, then runs the
model on incoming `NCZYX` tensors. The Z dimension is dropped before forward.

Run with:
    nix run . -- ipc:///tmp/scdino.ipc
or:
    python server.py ipc:///tmp/scdino.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# Make pyscripts importable so we can use the local `vision_transformer` module.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyscripts"))

import vision_transformer as vits  # noqa: E402

address = sys.argv[1]


def setup(
    arch: str = "vit_small",
    patch_size: int = 16,
    in_chans: int = 3,
    weights: str | None = None,
    checkpoint_key: str = "teacher",
    device: int | None = None,
    expected_tile_size: int = 16,
) -> tuple[Callable, dict]:
    """Build a scDINO ViT and (optionally) load a checkpoint.

    Parameters
    ----------
    arch : str
        Architecture name: ``vit_tiny``, ``vit_small``, ``vit_base``.
    patch_size : int
        Patch size (typically 16 or 8).
    in_chans : int
        Number of input channels expected by the model.
    weights : str | None
        Path to a ``.pth`` checkpoint. If None, the model uses random init —
        useful for smoke tests.
    checkpoint_key : str
        Key inside the checkpoint dict to load (e.g. ``teacher`` or ``student``).
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    if arch not in vits.__dict__:
        raise ValueError(
            f"Unknown arch {arch!r}; available: vit_tiny, vit_small, vit_base"
        )
    model = vits.__dict__[arch](
        patch_size=patch_size, num_classes=0, in_chans=in_chans
    )

    if weights is not None and os.path.exists(weights):
        state_dict = torch.load(weights, map_location="cpu")
        if isinstance(state_dict, dict) and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        # Strip common DINO prefixes.
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_info = {"missing": len(missing), "unexpected": len(unexpected)}
    else:
        load_info = {"missing": 0, "unexpected": 0, "weights": "random"}

    model.to(torch_device).eval()

    info = {
        "device": str(torch_device),
        "arch": arch,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "load": load_info,
    }
    processor = partial(
        process,
        model=model,
        device=torch_device,
        expected_tile_size=expected_tile_size,
        expected_channels=in_chans,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    expected_tile_size: int,
    expected_channels: int,
) -> numpy.ndarray:
    """Forward an NCZYX numpy array through the scDINO ViT, returning CLS features."""
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    pixels = pad_channel_dim(pixels, expected_channels)
    torch_tensor = torch.from_numpy(pixels.copy()).float().to(device)

    with torch.no_grad():
        feats = model(torch_tensor)  # CLS token, shape (N, embed_dim)
    return feats


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"scDINO server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
