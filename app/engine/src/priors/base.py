from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class VelocityPrior(ABC):
  # Flow time is a scalar in [0, 1]; every implementation treats it as such
  # (see the float(t) uses below). The contract is float, not Tensor | float,
  # so the declared type matches what the code can actually accept.
  @abstractmethod
  def velocity(self, x_t: Tensor, t: float, **cond: Any) -> Tensor: ...

  def __call__(self, x_t: Tensor, t: float, **cond: Any) -> Tensor:
    return self.velocity(x_t, t, **cond)


class LinearTargetStub(VelocityPrior):
  def __init__(self, x1: Tensor, t_eps: float = 1e-4) -> None:
    self.x1 = x1
    self.t_eps = t_eps

  def velocity(self, x_t: Tensor, t: float, **cond: Any) -> Tensor:
    denom = max(1.0 - t, self.t_eps)
    return (self.x1 - x_t) / denom


class ConstantVelocityStub(VelocityPrior):
  def __init__(self, x0: Tensor, x1: Tensor) -> None:
    self.v = x1 - x0

  def velocity(self, x_t: Tensor, t: float, **cond: Any) -> Tensor:
    return self.v.expand_as(x_t)