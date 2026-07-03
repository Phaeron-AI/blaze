from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .eval.metrics import psnr, ssim
from .eval.run_log import RunLogger
from .operators.base import ForwardOperator
from .operators.blur_downsample import BlurDownsampleOperator, gaussian_psf
from .priors.base import VelocityPrior
from .reconstruct import ReconstructionConfig, SuperResReconstructor


@dataclass
class EvalResult:
  recon_psnr: float
  recon_ssim: float
  floor_psnr: float
  floor_ssim: float
  gap_db: float  # recon_psnr - floor_psnr : the prior's contribution
  consistency_residual: float
  config: dict

  def summary(self) -> str:
    verdict = (
      "prior ADDS signal" if self.gap_db > 0.3 else "prior neutral/negative — likely out-of-domain"
    )
    return (
      f"  recon : {self.recon_psnr:6.2f} dB  SSIM {self.recon_ssim:.3f}\n"
      f"  floor : {self.floor_psnr:6.2f} dB  SSIM {self.floor_ssim:.3f}  (A_dag y, no prior)\n"
      f"  gap   : {self.gap_db:+6.2f} dB   -> {verdict}\n"
      f"  A x_hat=y residual: {self.consistency_residual:.2e}"
    )


class ReconstructionEvaluator:
  def __init__(
    self,
    operator: ForwardOperator,
    prior_factory: Callable[[], VelocityPrior],
    config: ReconstructionConfig | None = None,
    logger: RunLogger | None = None,
    data_range: float = 1.0,
  ):
    self.operator = operator
    self.prior_factory = prior_factory
    self.config = config or ReconstructionConfig()
    self.logger = logger
    self.data_range = data_range

  @torch.no_grad()
  def evaluate(
    self, ground_truth: Tensor, *, seed: int | None = None, note: str = ""
  ) -> EvalResult:
    if seed is not None:
      torch.manual_seed(seed)

    y = self.operator.A(ground_truth)
    reconstructor = SuperResReconstructor(self.operator, self.prior_factory(), self.config)
    out = reconstructor.reconstruct(y, image_shape=tuple(ground_truth.shape))
    x_hat = out["x_hat"].clamp(0, self.data_range)
    floor = out["pinv_floor"].clamp(0, self.data_range)

    recon_psnr = psnr(x_hat, ground_truth, self.data_range)
    floor_psnr = psnr(floor, ground_truth, self.data_range)
    result = EvalResult(
      recon_psnr=recon_psnr,
      recon_ssim=ssim(x_hat, ground_truth, self.data_range),
      floor_psnr=floor_psnr,
      floor_ssim=ssim(floor, ground_truth, self.data_range),
      gap_db=recon_psnr - floor_psnr,
      consistency_residual=out["consistency_residual"],
      config=asdict(self.config),
    )

    if self.logger is not None:
      self.logger.record(
        config={**asdict(self.config), "seed": seed, "shape": list(ground_truth.shape)},
        metrics={
          "recon_psnr": result.recon_psnr,
          "recon_ssim": result.recon_ssim,
          "floor_psnr": result.floor_psnr,
          "gap_db": result.gap_db,
          "consistency_residual": result.consistency_residual,
        },
        note=note,
      )
    return result


def default_sr_operator(
  scale: int = 2, psf_size: int = 5, psf_sigma: float = 1.0, device: str = "cpu"
) -> BlurDownsampleOperator:
  return BlurDownsampleOperator(gaussian_psf(psf_size, psf_sigma).to(device), scale=scale)


def load_image(path: str, device: str = "cpu") -> Tensor:
  import numpy as np
  from PIL import Image

  img = Image.open(path).convert("RGB")
  arr = torch.from_numpy(np.asarray(img)).float() / 255.0  # (H,W,3)
  return arr.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)


def to_multiple(x: Tensor, k: int) -> Tensor:
  _, _, h, w = x.shape
  nh, nw = (h // k) * k, (w // k) * k
  top, left = (h - nh) // 2, (w - nw) // 2
  return x[:, :, top : top + nh, left : left + nw]
