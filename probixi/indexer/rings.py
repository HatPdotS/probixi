"""Circular integration and frame-wide geometric overlap exclusion."""

import math

import torch


def snap_positions(positions, observed, radius):
    """Snap predictions to observed centroids.

    Parameters
    ----------
    positions, observed : Tensor
        Predicted and observed (row, column) coordinates in pixels.
    radius : float
        Maximum snapping distance in pixels.

    Returns
    -------
    tuple of Tensor
        Updated positions and the boolean snapped mask.
    """
    snapped = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    if len(positions) and len(observed):
        distance, index = torch.cdist(positions, observed.to(positions)).min(1)
        snapped = distance < radius
        positions = torch.where(
            snapped[:, None], observed.to(positions)[index], positions
        )
    return positions, snapped


def keep_non_overlapping(positions, integration_radius_px, panels=None):
    """Keep centres at least 1.5 signal radii from every same-panel neighbour.

    Parameters
    ----------
    positions : Tensor
        Finite (N, 2) detector coordinates in pixels.
    integration_radius_px : float
        Positive radius of the signal circle.
    panels : Tensor, optional
        Integer panel identifiers aligned with positions.

    Returns
    -------
    Tensor
        Boolean keep mask; both members of a close pair are excluded.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not math.isfinite(integration_radius_px) or integration_radius_px <= 0:
        raise ValueError("integration radius must be finite and positive")
    keep = torch.ones(len(positions), dtype=torch.bool, device=positions.device)
    positions = positions.float()
    cutoff2 = (1.5 * integration_radius_px) ** 2
    for start in range(0, len(positions), 512):
        block = positions[start : start + 512]
        distance2 = (block[:, None] - positions[None]).square().sum(-1)
        row = torch.arange(len(block), device=positions.device)
        distance2[row, start + row] = float("inf")
        if panels is not None:
            distance2.masked_fill_(
                panels[start : start + 512, None] != panels[None], float("inf")
            )
        keep[start : start + 512] = distance2.amin(1) >= cutoff2
    return keep


@torch.no_grad()
def integrate_rings(
    pred_positions,
    excess,
    var,
    obs_positions,
    snap_radius=5,
    box_radius=3,
    mean=None,
    pixel_valid=None,
    adu_per_photon=1,
    n_bg=None,
    *,
    radii,
):
    """Integrate signed signal above a reflection-centred annular background.

    Parameters
    ----------
    pred_positions, excess, var, obs_positions, snap_radius, box_radius, mean, pixel_valid, adu_per_photon, n_bg
        As for ``integrate_predicted``. ``var``, ``box_radius`` and ``n_bg`` are
        unused: the annulus supplies the background variance and sample count.
    radii : tuple of float
        Signal radius and inner/outer background radii, in pixels.

    Returns
    -------
    tuple of Tensor
        Snapped positions, intensities, sigmas, snapped mask, peaks, backgrounds.
    """
    if mean is None:
        raise ValueError("circular integration requires the background mean image")
    positions, snapped = snap_positions(pred_positions, obs_positions, snap_radius)
    if not len(positions):
        empty = excess.new_empty(0)
        return positions, empty, empty, snapped, empty, empty
    raw = excess + mean
    extent = math.ceil(radii[2])
    off = torch.arange(-extent, extent + 1, device=raw.device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    dr = dr.flatten()
    dc = dc.flatten()
    radius = dr**2 + dc**2
    center = positions.round().long()
    rr = center[:, 0, None] + dr
    cc = center[:, 1, None] + dc
    ok = (rr >= 0) & (rr < raw.shape[0]) & (cc >= 0) & (cc < raw.shape[1])
    flat = rr.clamp(0, raw.shape[0] - 1) * raw.shape[1] + cc.clamp(0, raw.shape[1] - 1)
    if pixel_valid is not None:
        ok &= pixel_valid.flatten()[flat]
    pixels = raw.flatten()[flat]
    bgmask = ok & (radius >= radii[1] ** 2) & (radius <= radii[2] ** 2)
    nb = bgmask.sum(1)
    background = torch.where(bgmask, pixels, 0).sum(1) / nb.clamp_min(1)
    bgvariance = torch.where(bgmask, (pixels - background[:, None]) ** 2, 0).sum(1) / (
        nb - 1
    ).clamp_min(1)
    use = ok & (radius <= radii[0] ** 2)
    # Sparse nearest-owner reduction prevents double counting overlapping disks.
    unique, inverse = torch.unique(flat.flatten(), return_inverse=True)
    inverse = inverse.reshape_as(flat)
    distance = (rr - positions[:, 0, None]) ** 2 + (cc - positions[:, 1, None]) ** 2
    nearest = torch.full_like(unique, float("inf"), dtype=raw.dtype)
    nearest.scatter_reduce_(
        0,
        inverse.flatten(),
        torch.where(use, distance, float("inf")).flatten(),
        reduce="amin",
    )
    wins = use & (distance <= nearest[inverse] + 1e-6)
    cid = torch.arange(len(positions), device=raw.device)[:, None].expand_as(flat)
    owner = torch.full_like(unique, len(positions))
    owner.scatter_reduce_(
        0,
        inverse.flatten(),
        torch.where(wins, cid, len(positions)).flatten(),
        reduce="amin",
    )
    use = wins & (owner[inverse] == cid)
    n = use.sum(1)
    intensity = torch.where(use, pixels - background[:, None], 0).sum(1)
    totalvar = (
        n * bgvariance * (1 + n / nb.clamp_min(1))
        + intensity.clamp_min(0) * adu_per_photon
    )
    sigma = totalvar.clamp_min(1e-12).sqrt()
    sigma = torch.where((n > 0) & (nb >= 10), sigma, torch.zeros_like(sigma))
    peak = torch.where(use, pixels - background[:, None], float("-inf")).amax(1)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    return positions, intensity, sigma, snapped, peak, background
