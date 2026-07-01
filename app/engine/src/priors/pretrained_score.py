from __future__ import annotations

import torch
from torch import Tensor

from typing import Optional

from .score_to_velocity import ScoreToVelocity
from .schedules import DiscreteLinearSchedule

IMAGENET256_UNCOND_FLAGS = dict(
  image_size=256,
  num_channels=256,
  num_res_blocks=2,
  num_head_channels=64,
  attention_resolutions="32,16,8",
  resblock_updown=True,
  use_scale_shift_norm=True,
  learn_sigma=True,
  class_cond=False,
  diffusion_steps=1000,
  noise_schedule="linear",
)

class PretrainedScorePrior:
  def __init__(
    self,
    checkpoint_path: str,
    device: str = "cuda",
    use_fp16: bool = False,
    num_diffusion_steps: int = 1000,
    flow_time_is_clean_at_one: bool = True,
  ):
    from guided_diffusion.script_util import (  # type: ignore[import]
      create_model_and_diffusion, model_and_diffusion_defaults,
    )

    args = model_and_diffusion_defaults()
    args.update(IMAGENET256_UNCOND_FLAGS)
    args["use_fp16"] = use_fp16
    model, _ = create_model_and_diffusion(**args)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
      state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
      # surface any mismatch loudly — a silent partial load is a silent bug.
      raise RuntimeError(
        f"checkpoint/architecture mismatch: "
        f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
        f"first missing: {missing[:3]} first unexpected: {unexpected[:3]}"
      )
    if use_fp16:
      model.convert_to_fp16()
    model.to(device).eval()

    self.model = model
    self.device = device
    self.T = num_diffusion_steps
    self.flow_clean_at_one = flow_time_is_clean_at_one

  def _flow_t_to_step(self, t: float) -> float:
    tau = (1.0 - t) if self.flow_clean_at_one else t   # diffusion time in [0,1]
    return max(0.0, min(1.0, tau)) * (self.T - 1)

  @torch.no_grad()
  def eps_at_step(self, x: Tensor, step: int | float) -> Tensor:
    ts = torch.full((x.shape[0],), float(step), device=x.device, dtype=torch.float32)
    out = self.model(x, ts)              # (N, 6, H, W)
    return out[:, :3]                    # eps; discard the learned-variance half

  @torch.no_grad()
  def eps(self, x: Tensor, t) -> Tensor:
    return self.eps_at_step(x, self._flow_t_to_step(float(t)))

  def as_velocity_prior(self, schedule: Optional[DiscreteLinearSchedule] = None) -> ScoreToVelocity:
    sched = schedule or DiscreteLinearSchedule(
      num_steps=self.T, beta_start=1e-4, beta_end=2e-2,
    )
    return ScoreToVelocity(
      model_fn=lambda xt, tt: self.eps(xt, tt),
      schedule=sched, # type: ignore[arg-type]
      parameterization="eps",
      flow_time_is_clean_at_one=self.flow_clean_at_one,
    )