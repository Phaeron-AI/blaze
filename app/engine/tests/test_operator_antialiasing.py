"""Regression tests for the anti-aliased BlurDownsampleOperator.

These lock in the operator fix for the checkerboard artifact (a scale-mismatched
PSF under-filtered before downsampling, folding aliased frequencies back as a
grid that the prior amplified). They assert three properties across the scales
the product uses:

  1. the scale-matched PSF suppresses the checkerboard in the floor,
  2. it does so far better than the old fixed PSF (the fix is real, not marginal),
  3. the operator still passes full range-null certification (adjoint, alignment,
     shape), so the anti-aliasing did not break the guarantee.

Even scales (2, 4) are covered. Odd scales are a known limitation: the
phase-centered sampling leaves a half-pixel asymmetry that fails the alignment
certification, so odd-scale support is deferred rather than silently shipped.
"""

from __future__ import annotations

import pytest
import torch

from engine.src.decomposition.range_null import RangeNullDecomposition
from engine.src.diagnostics_report import checkerboard_energy
from engine.src.operators.blur_downsample import (
  BlurDownsampleOperator,
  gaussian_psf,
  matched_psf,
)
from engine.src.operators.diagnostics import adjoint_test

SCALES = [2, 4]
SIZE = 128

# The old fixed PSF that caused the artifact, kept as the regression baseline.
FIXED_PSF_SIZE = 5
FIXED_PSF_SIGMA = 1.5

# The floor of a smooth input must be smooth; anything above this is grid energy.
# Chosen to pass the matched PSF comfortably while failing the old fixed PSF.
CHECKERBOARD_CEILING = 0.04

# The matched PSF must beat the fixed PSF by at least this factor, so the test
# fails if a future change quietly reverts to under-filtering.
MIN_IMPROVEMENT_FACTOR = 2.0


@pytest.fixture
def smooth_image() -> torch.Tensor:
  # A smooth gradient has no genuine high-frequency content, so any grid in its
  # reconstruction is an artifact rather than recovered detail.
  yy, xx = torch.meshgrid(
    torch.linspace(0, 1, SIZE), torch.linspace(0, 1, SIZE), indexing="ij"
  )
  image = (0.5 + 0.3 * xx + 0.2 * yy).unsqueeze(0).unsqueeze(0)
  return image.repeat(1, 3, 1, 1)


def _floor_grid_energy(kernel: torch.Tensor, scale: int, image: torch.Tensor) -> float:
  operator = BlurDownsampleOperator(kernel, scale=scale)
  measurement = operator.A(image)
  decomposition = RangeNullDecomposition(
    operator, measurement, tuple(image.shape), cg_iters=600
  )
  return checkerboard_energy(decomposition.pinv_reconstruction)


@pytest.mark.parametrize("scale", SCALES)
def test_matched_psf_floor_below_ceiling(scale: int, smooth_image: torch.Tensor) -> None:
  energy = _floor_grid_energy(matched_psf(scale), scale, smooth_image)
  assert energy < CHECKERBOARD_CEILING, (
    f"scale {scale}: floor grid energy {energy:.4e} exceeds ceiling "
    f"{CHECKERBOARD_CEILING}; the checkerboard fix may have regressed"
  )


@pytest.mark.parametrize("scale", SCALES)
def test_matched_psf_beats_fixed(scale: int, smooth_image: torch.Tensor) -> None:
  matched = _floor_grid_energy(matched_psf(scale), scale, smooth_image)
  fixed = _floor_grid_energy(
    gaussian_psf(FIXED_PSF_SIZE, FIXED_PSF_SIGMA), scale, smooth_image
  )
  assert matched * MIN_IMPROVEMENT_FACTOR < fixed, (
    f"scale {scale}: matched PSF grid {matched:.4e} is not at least "
    f"{MIN_IMPROVEMENT_FACTOR}x better than fixed PSF grid {fixed:.4e}"
  )


@pytest.mark.parametrize("scale", SCALES)
def test_matched_operator_adjoint_exact(scale: int) -> None:
  operator = BlurDownsampleOperator(matched_psf(scale), scale=scale)
  _, _, rel_error = adjoint_test(operator, (1, 3, SIZE, SIZE), "cpu")
  assert rel_error < 1e-4, (
    f"scale {scale}: adjoint relative error {rel_error:.2e} too large; "
    "A and A_T are no longer a transpose pair"
  )


@pytest.mark.parametrize("scale", SCALES)
def test_matched_operator_passes_certification(
  scale: int, smooth_image: torch.Tensor
) -> None:
  # RangeNullDecomposition runs adjoint, alignment, and shape certification and
  # raises if any fails, so a successful build is a full-certification pass.
  operator = BlurDownsampleOperator(matched_psf(scale), scale=scale)
  measurement = operator.A(smooth_image)
  RangeNullDecomposition(operator, measurement, tuple(smooth_image.shape), cg_iters=400)


@pytest.mark.skip(reason="odd-scale sampling is not yet supported (alignment drift)")
def test_odd_scale_support() -> None:
  # Odd scales leave a half-pixel sampling asymmetry whose alignment behaviour
  # is size-dependent, so they are not supported. This placeholder marks the
  # gap; unskip and implement when odd-scale sampling lands.
  pass