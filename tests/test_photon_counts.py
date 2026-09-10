"""Photon counts for detected peaks, and how they reach the DB.

Only searched peaks carry these: a predicted position has no measured blob, so
there is no honest pixel set to sum a background over.
"""

from __future__ import annotations

import torch

from probixi.peakfinding.peaks.blobs import compute_blob_stats


def _maps():
    labels = torch.zeros(9, 9, dtype=torch.long)
    labels[2:4, 2:4] = 1  # 4-pixel blob
    labels[6, 6] = 2  # 1-pixel blob
    excess = torch.zeros(9, 9)
    excess[2:4, 2:4] = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    excess[6, 6] = 7.0
    mean = torch.full((9, 9), 3.0)
    return labels, excess, mean


def _stats(mean=None):
    labels, excess, m = _maps()
    return compute_blob_stats(
        labels,
        2,
        excess=excess,
        z=excess,
        log_bf=excess,
        posterior=excess,
        var=torch.ones(9, 9),
        mean=m if mean is None else mean,
    )


def test_background_sum_totals_the_noise_model_over_the_blob_pixels():
    stats = _stats()
    assert float(stats.background_sum[0]) == 12.0  # 4 px x mean 3.0
    assert float(stats.background_sum[1]) == 3.0  # 1 px x mean 3.0


def test_background_sum_pairs_with_size_not_a_bounding_box():
    # the sum is over the blob's own pixels, so it tracks `size` exactly
    stats = _stats()
    assert [int(v) for v in stats.size] == [4, 1]
    assert torch.allclose(stats.background_sum, stats.size.float() * 3.0)


def test_observed_counts_are_excess_plus_background():
    stats = _stats()
    observed = stats.intensity_sum + stats.background_sum
    assert float(observed[0]) == 112.0  # 100 excess + 12 background
    assert float(observed[1]) == 10.0  # 7 excess + 3 background


def test_background_sum_is_zero_when_no_mean_map_is_given():
    labels, excess, _ = _maps()
    stats = compute_blob_stats(
        labels,
        2,
        excess=excess,
        z=excess,
        log_bf=excess,
        posterior=excess,
        var=torch.ones(9, 9),
    )
    assert torch.all(stats.background_sum == 0.0)


def test_background_sum_survives_blob_selection():
    from probixi.peakfinding.peaks.blobs import select_blobs

    stats = _stats()
    kept = select_blobs(stats, torch.tensor([False, True]))
    assert float(kept.background_sum[0]) == 3.0


def test_peaks_only_gain_falls_back_to_geometry(tmp_path):
    from probixi.io.db import DuckDBOffloader

    off = DuckDBOffloader(tmp_path / "x.db", {"adu_per_photon": 2.0})
    assert off._gain() == 2.0
    assert off._gain(None) == 2.0


def test_indexed_gain_prefers_the_value_the_frame_was_processed_with(tmp_path):
    from probixi.io.db import DuckDBOffloader

    class R:
        adu_per_photon = 0.671

    off = DuckDBOffloader(tmp_path / "x.db", {"adu_per_photon": 12960.0})
    assert off._gain(R()) == 0.671  # measured gain wins over the geometry's
    assert off._gain() == 12960.0


def test_gain_rejects_nonsense_values(tmp_path):
    from probixi.io.db import DuckDBOffloader

    for bad in ({}, {"adu_per_photon": None}, {"adu_per_photon": 0.0}):
        off = DuckDBOffloader(tmp_path / "x.db", bad)
        assert off._gain() == 1.0
