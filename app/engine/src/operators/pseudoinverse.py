from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from .base import ForwardOperator

@dataclass
class ConjugateGradientInverse:
  operator: ForwardOperator
  max_iters: int = 600
  tol: float = 1e-10
  damping: float = 1e-6

  def normal_matvec(self, v: Tensor)-> Tensor:
    return self.operator.A_T(self.operator.A(v)) + self.damping * v
  
  def solve(self, y: Tensor, image_shape: torch.Size | tuple, x0: Optional[Tensor] = None)-> Tensor:
    b = self.operator.A_T(y)                      # rhs = A^T y, lives in image space
    x = torch.zeros(image_shape, device=b.device, dtype=b.dtype) if x0 is None else x0.clone()
    r = b - self.normal_matvec(x)
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(self.max_iters):
      Ap = self.normal_matvec(p)
      denom = (p * Ap).sum()
      if denom.abs() < 1e-20:
        break
      alpha = rs / denom
      x = x + alpha * p
      r = r - alpha * Ap
      rs_new = (r * r).sum()
      if rs_new.sqrt() < self.tol:
        break
      p = r + (rs_new / (rs + 1e-20)) * p
      rs = rs_new
    return x