"""Utilities for batched :class:`BSplineConstraintPack` handling.

The migrated ``bspline_constraints.build_constraint_pack_from_regions`` builds ONE
pack from a shared ``anchors[R]`` vector, i.e. it assumes every sample of the
batch uses the same region progresses.  In this project the number of regions
and their anchors differ per sample:

* a local corridor inserts bridge cells, so a sample has ``Q + bridges`` rows;
* the collision intervals differ, so the anchors differ too.

Rather than change the migrated builder, each sample is packed on its own
(same function, ``B = 1``) and the resulting packs are merged here with padding
on the piece / face axes.  Padding rows carry ``A = 0, b = +inf`` and
``piece_mask = False``, so they can never contribute a constraint.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch

from .bspline_constraints import BSplineConstraintPack

__all__ = ["merge_packs", "pack_to_device", "empty_pack"]


def empty_pack(num_controls: int, dtype=torch.float32, device=None) -> BSplineConstraintPack:
    """A pack with no piece and no face (used when nothing can be constrained)."""
    return BSplineConstraintPack(
        extraction=torch.zeros(1, 0, 4, int(num_controls), dtype=dtype, device=device),
        piece_A=torch.zeros(1, 0, 0, 2, dtype=dtype, device=device),
        piece_b=torch.zeros(1, 0, 0, dtype=dtype, device=device),
        face_mask=torch.zeros(1, 0, 0, dtype=torch.bool, device=device),
        piece_mask=torch.zeros(1, 0, dtype=torch.bool, device=device),
        piece_region_id=torch.full((1, 0), -1, dtype=torch.long, device=device),
        intervals=torch.zeros(1, 0, 2, dtype=dtype, device=device),
        num_pieces=torch.zeros(1, dtype=torch.long, device=device))


def merge_packs(packs: Iterable[Optional[BSplineConstraintPack]],
                num_controls: int,
                device=None) -> Optional[BSplineConstraintPack]:
    """Concatenate per-sample packs along the batch axis with padding.

    ``None`` entries become fully masked rows.  Returns ``None`` when every entry
    is ``None`` (nothing to constrain at all).
    """
    packs = list(packs)
    if not packs or all(p is None for p in packs):
        return None
    ref = next(p for p in packs if p is not None)
    dtype = ref.extraction.dtype
    device = device if device is not None else ref.extraction.device

    max_pieces = max(p.extraction.shape[1] for p in packs if p is not None)
    max_faces = max(p.piece_A.shape[2] for p in packs if p is not None)
    max_pieces = max(max_pieces, 1)
    max_faces = max(max_faces, 1)
    batch = len(packs)

    extraction = torch.zeros(batch, max_pieces, 4, int(num_controls),
                             dtype=dtype, device=device)
    piece_A = torch.zeros(batch, max_pieces, max_faces, 2, dtype=dtype, device=device)
    piece_b = torch.zeros(batch, max_pieces, max_faces, dtype=dtype, device=device)
    face_mask = torch.zeros(batch, max_pieces, max_faces, dtype=torch.bool, device=device)
    piece_mask = torch.zeros(batch, max_pieces, dtype=torch.bool, device=device)
    piece_region_id = torch.full((batch, max_pieces), -1, dtype=torch.long, device=device)
    intervals = torch.zeros(batch, max_pieces, 2, dtype=dtype, device=device)
    num_pieces = torch.zeros(batch, dtype=torch.long, device=device)

    for row, pack in enumerate(packs):
        if pack is None:
            continue
        pieces = int(pack.extraction.shape[1])
        faces = int(pack.piece_A.shape[2])
        if pieces == 0:
            continue
        extraction[row, :pieces] = pack.extraction[0].to(dtype=dtype, device=device)
        piece_A[row, :pieces, :faces] = pack.piece_A[0].to(dtype=dtype, device=device)
        piece_b[row, :pieces, :faces] = pack.piece_b[0].to(dtype=dtype, device=device)
        face_mask[row, :pieces, :faces] = pack.face_mask[0].to(device=device)
        piece_mask[row, :pieces] = pack.piece_mask[0].to(device=device)
        piece_region_id[row, :pieces] = pack.piece_region_id[0].to(device=device)
        intervals[row, :pieces] = pack.intervals[0].to(dtype=dtype, device=device)
        num_pieces[row] = int(pack.num_pieces[0].item())

    return BSplineConstraintPack(
        extraction=extraction, piece_A=piece_A, piece_b=piece_b,
        face_mask=face_mask, piece_mask=piece_mask,
        piece_region_id=piece_region_id, intervals=intervals,
        num_pieces=num_pieces)


def pack_to_device(pack: Optional[BSplineConstraintPack],
                   device=None, dtype=None) -> Optional[BSplineConstraintPack]:
    """``pack.to`` that tolerates ``None`` (used inside the training loop)."""
    if pack is None:
        return None
    return pack.to(device=device, dtype=dtype)


def pack_active_rows(pack: BSplineConstraintPack) -> torch.Tensor:
    """Per-sample number of pieces that actually carry at least one face."""
    if pack is None:
        return torch.zeros(0, dtype=torch.long)
    active = (pack.piece_mask & pack.face_mask.any(dim=-1))
    return active.sum(dim=1).long()
