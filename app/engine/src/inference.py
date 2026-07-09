from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .eval.metrics import psnr, ssim
from .priors.schedules import DiscreteLinearSchedule
from .protocols import DiffusionPrior, ForwardOperator
from .reconstruct import DDNMReconstructor, ReconstructionConfig


@dataclass
class ReconstructionResult:
  reconstruction: Tensor
  floor: Tensor
  measurement: Tensor
  consistency_residual: float
  recon_psnr: float | None = None
  floor_psnr: float | None = None
  gap_db: float | None = None
  recon_ssim: float | None = None
  floor_ssim: float | None = None

  @property
  def has_metrics(self) -> bool:
    return self.recon_psnr is not None


def reconstruct(
  operator: ForwardOperator,
  prior: DiffusionPrior,
  measurement: Tensor,
  image_shape: tuple[int, ...],
  *,
  ground_truth: Tensor | None = None,
  config: ReconstructionConfig | None = None,
  schedule: DiscreteLinearSchedule | None = None,
  data_range: float = 1.0,
  generator: torch.Generator | None = None,
) -> ReconstructionResult:
  cfg = config or ReconstructionConfig()
  sched = schedule or DiscreteLinearSchedule(1000, 1e-4, 2e-2)

  reconstructor = DDNMReconstructor(operator, prior.eps_at_step, sched, cfg)
  out = reconstructor.reconstruct(measurement, image_shape, generator=generator)

  x_hat = out["x_hat"].clamp(0, data_range)
  floor = out["pinv_floor"].clamp(0, data_range)

  result = ReconstructionResult(
    reconstruction=x_hat,
    floor=floor,
    measurement=measurement,
    consistency_residual=float(out["consistency_residual"]),
  )

  if ground_truth is not None:
    gt = ground_truth.clamp(0, data_range)
    result.recon_psnr = psnr(x_hat, gt, data_range)
    result.floor_psnr = psnr(floor, gt, data_range)
    result.gap_db = result.recon_psnr - result.floor_psnr
    result.recon_ssim = ssim(x_hat, gt, data_range)
    result.floor_ssim = ssim(floor, gt, data_range)

  return result


def reconstruct_demo(
  operator: ForwardOperator,
  prior: DiffusionPrior,
  clean_image: Tensor,
  *,
  config: ReconstructionConfig | None = None,
  schedule: DiscreteLinearSchedule | None = None,
  data_range: float = 1.0,
  generator: torch.Generator | None = None,
) -> ReconstructionResult:
  measurement = operator.A(clean_image)
  return reconstruct(
    operator,
    prior,
    measurement,
    image_shape=tuple(clean_image.shape),
    ground_truth=clean_image,
    config=config,
    schedule=schedule,
    data_range=data_range,
    generator=generator,
  )


def to_display_array(x: Tensor) -> Tensor:
  if x.dim() == 4:
    x = x[0]
  x = x.clamp(0, 1).detach().cpu()
  return (x.permute(1, 2, 0) * 255).round().to(torch.uint8)