from __future__ import annotations

from torch import Tensor
from typing import Any

from .base import VelocityPrior


class NormalizedVelocityPrior(VelocityPrior):
  def __init__(self, inner: VelocityPrior, scale: float = 2.0, shift: float = -1.0) -> None:
    if scale == 0:
      raise ValueError("scale must be non-zero")
    self.inner = inner
    self.scale = float(scale)
    self.shift = float(shift)

  def to_model_space(self, x: Tensor) -> Tensor:
    return self.scale * x + self.shift

  def from_model_space(self, x_model: Tensor) -> Tensor:
    return (x_model - self.shift) / self.scale

  def velocity(self, x_t: Tensor, t: float, **cond: Any) -> Tensor:
    x_model = self.to_model_space(x_t)
    v_model = self.inner.velocity(x_model, t, **cond)

    return v_model / self.scale
