"""Public API for the engine's core certified components.

Explicit re-exports (no star imports) so the exported surface is visible to
readers and to static analysis. This is the interface downstream code and the
web layer import from; keep it curated rather than a dumping ground.
"""

from __future__ import annotations

from .decomposition import RangeNullDecomposition
from .operators import (
  ConjugateGradientInverse,
  _psnr,
  adjoint_test,
  assert_adjoint,
  assert_alignment,
  consistency_test,
  impulse_alignment_test,
  reconstruction_floor,
)
from .protocols import ForwardOperator

__all__ = [
  "ConjugateGradientInverse",
  "ForwardOperator",
  "RangeNullDecomposition",
  "_psnr",
  "adjoint_test",
  "assert_adjoint",
  "assert_alignment",
  "consistency_test",
  "impulse_alignment_test",
  "reconstruction_floor",
]