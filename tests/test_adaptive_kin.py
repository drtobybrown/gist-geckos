"""
Tests for the adaptive coarse-to-fine template grid (KIN.ADAPTIVE_GRID).

- Unit tests: grid index builders and that coarse/fine logic is consistent.
- Integration: run KIN with ADAPTIVE_GRID and assert outputs are valid (optional, needs data).
"""

import os
import sys

import numpy as np

# Project root
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def test_build_coarse_idx():
    """Coarse index list has correct size and ordering (age fastest, then metal, then alpha)."""
    from ngistPipeline.stellarKinematics.ppxf_kin_wrapper import _build_coarse_idx

    nAges, nMetal, nAlpha = 8, 4, 2
    # step 2,2,1 -> 4*2*2 = 16 coarse points
    idx = _build_coarse_idx(nAges, nMetal, nAlpha, 2, 2, 1)
    assert idx.ndim == 1
    assert len(idx) == (4 * 2 * 2)
    assert np.all(idx >= 0) and np.all(idx < nAges * nMetal * nAlpha)
    assert len(np.unique(idx)) == len(idx)

    # step 1,1,1 -> full grid
    full = _build_coarse_idx(nAges, nMetal, nAlpha, 1, 1, 1)
    assert len(full) == nAges * nMetal * nAlpha
    np.testing.assert_array_equal(np.sort(full), np.arange(nAges * nMetal * nAlpha))


def test_build_fine_window_idx():
    """Fine window is a box around (j0,k0,i0) and stays in bounds."""
    from ngistPipeline.stellarKinematics.ppxf_kin_wrapper import _build_fine_window_idx

    nAges, nMetal, nAlpha = 10, 5, 3
    j0, k0, i0 = 5, 2, 1
    radius_age, radius_metal, radius_alpha = 2, 1, 0

    idx = _build_fine_window_idx(
        nAges, nMetal, nAlpha, j0, k0, i0,
        radius_age, radius_metal, radius_alpha,
    )
    assert idx.ndim == 1
    # Box [3,4,5,6,7] x [1,2,3] x [1] = 5*3*1 = 15
    assert len(idx) == 5 * 3 * 1
    assert len(np.unique(idx)) == len(idx)

    # Center (j0,k0,i0) must be in the set
    t0 = j0 + nAges * k0 + nAges * nMetal * i0
    assert t0 in idx

    # All indices must be valid
    assert np.all(idx >= 0) and np.all(idx < nAges * nMetal * nAlpha)


def test_adaptive_returns_finite_when_coarse_succeeds():
    """With synthetic data, _run_ppxf_adaptive returns finite kinematics (smoke test)."""
    from ngistPipeline.stellarKinematics.ppxf_kin_wrapper import _run_ppxf_adaptive

    np.random.seed(42)
    # ppxf with velscale_ratio=2 needs templates at least 2x galaxy length; use ratio 1 to keep test small
    npix = 200
    nAges, nMetal, nAlpha = 6, 4, 1
    ntemplates = nAges * nMetal * nAlpha

    templates = np.random.randn(npix, ntemplates).astype(np.float64) * 0.1 + 1.0
    t_best = 10
    gal = templates[:, t_best] + np.random.randn(npix) * 0.05
    noise = np.ones(npix) * 0.05
    logLam = np.linspace(np.log(4800), np.log(7000), npix)
    c_km_s = 299792.458
    velscale = float(c_km_s * (logLam[1] - logLam[0]))
    goodPixels = np.arange(10, npix - 10)
    start = np.array([0.0, 100.0])
    offset = 0.0

    out = _run_ppxf_adaptive(
        templates,
        nAges, nMetal, nAlpha,
        gal, noise,
        velscale, start, None, goodPixels,
        2, 4, 0, None, True, logLam, offset,
        1, 0, 1, 0,
        [0],
        [3, 2, 1],
        [1, 1, 0],
    )
    assert len(out) == 9
    sol, _, bestfit, optimal, _, _, _, _, _ = out
    assert np.all(np.isfinite(sol)), "adaptive should return finite kinematics"
    assert np.all(np.isfinite(bestfit)), "bestfit should be finite"
    assert optimal is None or np.all(np.isfinite(optimal))


def test_adaptive_vs_full_agreement_synthetic():
    """When the true template lies on the coarse grid, adaptive and full-grid agree closely."""
    from ngistPipeline.stellarKinematics.ppxf_kin_wrapper import run_ppxf, _run_ppxf_adaptive

    np.random.seed(123)
    npix = 150
    nAges, nMetal, nAlpha = 8, 4, 1
    ntemplates = nAges * nMetal * nAlpha

    templates = np.random.randn(npix, ntemplates).astype(np.float64) * 0.08 + 1.0
    # Use template 4: with coarse steps [4,2,1], index 4 is on the coarse grid (j=4,k=0,i=0)
    gal = templates[:, 4] + np.random.randn(npix) * 0.04
    noise = np.ones(npix) * 0.04
    logLam = np.linspace(np.log(4800), np.log(7000), npix)
    c_km_s = 299792.458
    velscale = float(c_km_s * (logLam[1] - logLam[0]))
    goodPixels = np.arange(20, npix - 20)
    start = np.array([0.0, 80.0])
    velscale_ratio = 1

    full_out = run_ppxf(
        templates, gal, noise, velscale, start, None, goodPixels,
        2, 4, 0, None, True, logLam, 0.0, velscale_ratio, 0, 1, 0, [0],
    )
    full_sol = full_out[0]

    ad_out = _run_ppxf_adaptive(
        templates, nAges, nMetal, nAlpha,
        gal, noise, velscale, start, None, goodPixels,
        2, 4, 0, None, True, logLam, 0.0, velscale_ratio, 0, 1, 0, [0],
        [4, 2, 1], [2, 1, 0],
    )
    ad_sol = ad_out[0]

    assert np.all(np.isfinite(full_sol)) and np.all(np.isfinite(ad_sol)), "both fits should be finite"
    np.testing.assert_allclose(ad_sol[:2], full_sol[:2], atol=20.0, rtol=0.25,
                               err_msg="adaptive V, sigma should be close to full when true template is on coarse grid")


if __name__ == "__main__":
    test_build_coarse_idx()
    test_build_fine_window_idx()
    test_adaptive_returns_finite_when_coarse_succeeds()
    test_adaptive_vs_full_agreement_synthetic()
    print("All adaptive KIN tests passed.")
