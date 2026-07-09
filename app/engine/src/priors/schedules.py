from __future__ import annotations

import torch
from torch import Tensor

from typing import Any
class DiscreteLinearSchedule:
  def __init__(
    self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2
  ) -> None:
    self.T = num_steps
    self.beta_start = beta_start
    self.beta_end = beta_end

    self._betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    alphas = 1.0 - self._betas
    self._alpha_bars = torch.cumprod(alphas, dim=0)

  def _tau_to_index(self, tau: float) -> tuple[int, int, float]:
    s = max(0.0, min(1.0, float(tau))) * (self.T - 1)
    lo = int(s)
    hi = min(lo + 1, self.T - 1)
    frac = s - lo
    return lo, hi, frac

  def beta(self, tau: Any) -> Tensor:
    lo, hi, frac = self._tau_to_index(float(tau))
    b = self._betas[lo] * (1 - frac) + self._betas[hi] * frac

    return b.to(torch.float32)

  def alpha_bar(self, tau: Any) -> Tensor:
    lo, hi, frac = self._tau_to_index(tau)

    log_lo = torch.log(self._alpha_bars[lo])
    log_hi = torch.log(self._alpha_bars[hi])

    log_ab = log_lo * (1 - frac) + log_hi * frac

    return torch.exp(log_ab).to(torch.float32)

  def beta_at_step(self, t: int) -> float:
    return float(self._betas[t])

  def alpha_bar_at_step(self, t: int) -> float:
    return float(self._alpha_bars[t])
