from __future__ import annotations

from abc import ABC, abstractmethod
from torch import Tensor

class VelocityPrior(ABC):
  @abstractmethod
  def velocity(self, x_t: Tensor, t: Tensor | float, **cond)-> Tensor: ...
  def __call__(self, x_t: Tensor, t, **cond)-> Tensor:
    return self.velocity(x_t, t, **cond)
  
class LinearTargetStub(VelocityPrior):
  def __init__(self, x1: Tensor, t_eps: float = 1e-4)-> None:
    self.x1 = x1
    self.t_eps = t_eps

  def velocity(self, x_t: Tensor, t, **cond) -> Tensor:
    t_val = float(t)
    denom = max(1.0 - t_val, self.t_eps)
    return (self.x1 - x_t) / denom

class ConstantVelocityStub(VelocityPrior):
  def __init__(self, x0: Tensor, x1: Tensor)-> None:
    self.v = x1-x0
  
  def velocity(self, x_t: Tensor, t: Tensor | float, **cond) -> Tensor:
    return self.v.expand_as(x_t)