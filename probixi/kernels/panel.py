"""Accelerated tiled panel mean, preserving the existing panel map and masks.

Finite CUDA float32/float64 frames; fixed panel layout. Mask counts are cached
using tensor identity and PyTorch's in-place version counter. Tree reductions
avoid long FP32 atomic accumulation chains. FP32 is the default; FP64 is an
optional accuracy reference.
"""

from itertools import pairwise

import numpy as np
import torch
import triton
import triton.language as tl

from . import tensor_key


@triton.jit
def partials(
    F,
    V,
    M,
    START,
    LENGTH,
    SUMS,
    COUNTS,
    HAS_MASK: tl.constexpr,
    COUNT: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    n = tl.load(LENGTH + tile)
    pos = tl.load(START + tile) + j
    valid = tl.load(V + pos, j < n, other=0)
    if HAS_MASK:
        valid = valid & tl.load(M + pos, j < n, other=0)
    x = tl.load(F + pos, j < n, other=0)
    x = x.to(tl.float64) if FP64 else x.to(tl.float32)
    value = x * valid.to(x.dtype)
    tl.store(SUMS + tile, tl.sum(value, 0))
    if COUNT:
        tl.store(COUNTS + tile, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def finish(
    SUMS,
    COUNTS,
    PTR,
    CACHE,
    OUT,
    COUNT: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    panel = tl.program_id(0)
    lo = tl.load(PTR + panel)
    hi = tl.load(PTR + panel + 1)
    j = lo + tl.arange(0, BLOCK)
    total = tl.sum(tl.load(SUMS + j, j < hi, other=0), 0)
    if COUNT:
        count = tl.sum(tl.load(COUNTS + j, j < hi, other=0).to(tl.int64), 0)
        tl.store(CACHE + panel, count)
    else:
        count = tl.load(CACHE + panel)
    tl.store(OUT + panel, total / tl.maximum(count, 1).to(total.dtype))


class TiledPanelProjector:
    def __init__(self, panel, block=4096, fp64=False):
        self.fp64 = fp64
        self.panel = panel
        self.block = block
        ids = panel._pid_flat.detach().cpu().numpy()
        bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, ids.size]
        runs = [[] for _ in range(panel.n_panels)]
        for lo, hi in pairwise(bounds):
            runs[int(ids[lo])].append((int(lo), int(hi)))
        starts, lengths, ptr = [], [], [0]
        for group in runs:
            for lo, hi in group:
                for start in range(lo, hi, block):
                    starts.append(start)
                    lengths.append(min(block, hi - start))
            ptr.append(len(starts))
        device = panel.mean_.device
        self.starts = torch.tensor(starts, device=device, dtype=torch.int64)
        self.lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
        self.ptr = torch.tensor(ptr, device=device, dtype=torch.int32)
        self.max_tiles = int(max(np.diff(ptr)))
        self.sums = torch.empty(
            len(starts), device=device, dtype=torch.float64 if fp64 else torch.float32
        )
        self.counts = torch.empty(len(starts), device=device, dtype=torch.int32)
        self.cached_counts = torch.empty(
            panel.n_panels, device=device, dtype=torch.int64
        )
        self.mask_key = None
        self.mask_ref = None

    @torch.no_grad()
    def __call__(self, frame, mask=None):
        p = self.panel
        assert frame.is_cuda and frame.is_contiguous()
        assert frame.dtype in (torch.float32, torch.float64)
        assert tuple(frame.shape) == p.frame_size
        if mask is not None:
            assert (
                mask.dtype == torch.bool
                and mask.is_contiguous()
                and mask.device == frame.device
            )
        key = (
            tensor_key(p.valid_mask),
            tensor_key(mask) if mask is not None else None,
        )
        count = key != self.mask_key
        out = torch.empty_like(p.mean_)
        partials[(self.starts.numel(),)](
            frame,
            p.valid_mask,
            mask if mask is not None else p.valid_mask,
            self.starts,
            self.lengths,
            self.sums,
            self.counts,
            HAS_MASK=mask is not None,
            COUNT=count,
            FP64=self.fp64,
            BLOCK=self.block,
            enable_fp_fusion=False,
        )
        finish[(p.n_panels,)](
            self.sums,
            self.counts,
            self.ptr,
            self.cached_counts,
            out,
            COUNT=count,
            FP64=self.fp64,
            BLOCK=triton.next_power_of_2(max(1, self.max_tiles)),
            enable_fp_fusion=False,
        )
        self.mask_key = key
        self.mask_ref = (p.valid_mask, mask)
        return out
