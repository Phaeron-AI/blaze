from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import torch
from torch import Tensor

from .base import VelocityPrior

@dataclass
class NoiseSchedule:
  beta_min: float = 0.1
  beta_max: float = 20.0

  def beta(self, tau: Tensor | float)-> Tensor:
    tau = torch.as_tensor(tau, dtype=torch.float32)
    return self.beta_min + tau * (self.beta_max - self.beta_min)
  
  def alpha_bar(self, tau: Tensor | float)-> Tensor:
    tau = torch.as_tensor(tau, dtype=torch.float32)
    integral = self.beta_min * tau + 0.5 * (self.beta_max - self.beta_min) * tau **2
    return torch.exp(-integral)


class ScoreToVelocity(VelocityPrior):
  def __init__(
    self, 
    model_fn: Callable[[Tensor, Tensor], Tensor], 
    schedule: Optional[NoiseSchedule] = None, 
    parameterization: str= "eps", 
    flow_time_is_clean_at_one: bool = True
  )-> None:
    if parameterization not in ("eps", "score"):
      raise ValueError("parameterization must be 'eps' or 'score'")
    self.model_fn = model_fn
    self.schedule = schedule or NoiseSchedule()
    self.parameterization = parameterization
    self.flow_clean_at_one = flow_time_is_clean_at_one
  
  def _score(self, x_tau: Tensor, tau: float)-> Tensor:
    out = self.model_fn(x_tau, torch.as_tensor(tau, device=x_tau.device))
    if self.parameterization == "score":
      return out
    # eps -> score:  s = -eps_hat / sqrt(1 - alpha_bar)
    ab = self.schedule.alpha_bar(tau).to(x_tau.device)
    return -out / torch.sqrt(1.0 - ab).clamp(min=1e-8)
  
  def velocity(self, x_t: Tensor, t, **cond) -> Tensor:
    t_val = float(t)

    tau = (1.0 - t_val) if self.flow_clean_at_one else t_val
    beta = self.schedule.beta(tau).to(x_t.device)
    score = self._score(x_t, tau)

    v_diffusion = -0.5 * beta * (x_t + score)

    return -v_diffusion if self.flow_clean_at_one else v_diffusion
 