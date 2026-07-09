from __future__ import annotations

import argparse

import torch
from torch import Tensor
import torch.nn.functional as F

from ..diagnostics_report import compute_diagnostics
from ..evaluate import load_image
from ..inference import reconstruct_demo, to_display_array, ReconstructionResult
from ..operators.blur_downsample import BlurDownsampleOperator, gaussian_psf, matched_psf
from ..priors.pretrained_score import PretrainedScorePrior
from ..priors.schedules import DiscreteLinearSchedule
from ..reconstruct import ReconstructionConfig

UNET_DOWNSAMPLE = 32


def _downscale_to_max(x: torch.Tensor, max_size: int) -> torch.Tensor:
  _, _, h, w = x.shape
  longest = max(h, w)
  if longest <= max_size:
    return x
  ratio = max_size / longest
  new_h, new_w = round(h * ratio), round(w * ratio)
  return F.interpolate(x, size=(new_h, new_w), mode="area")


def _crop_to_multiple(x: torch.Tensor, k: int) -> torch.Tensor:
  _, _, h, w = x.shape
  nh, nw = (h // k) * k, (w // k) * k
  top, left = (h - nh) // 2, (w - nw) // 2
  return x[:, :, top : top + nh, left : left + nw]


def _save_panel(result: ReconstructionResult, clean: Tensor, difference: Tensor, out_path: str) -> None:
  import os

  import numpy as np
  from PIL import Image

  parent = os.path.dirname(out_path)
  if parent:
    os.makedirs(parent, exist_ok=True)

  measurement = torch.nn.functional.interpolate(
    result.measurement, size=clean.shape[-2:], mode="nearest"
  )
  # Scale the difference map to [0, 1] so faint prior contributions are visible.
  diff_vis = difference / (difference.max() + 1e-8)
  tiles = [clean, measurement, result.floor, result.reconstruction, diff_vis]
  arrays = [to_display_array(t).numpy() for t in tiles]
  panel = np.concatenate(arrays, axis=1)
  Image.fromarray(panel).save(out_path)


def main() -> None:
  torch.backends.cudnn.enabled = False
  parser = argparse.ArgumentParser(
    description="Validate the engine on a natural image (Phase A)."
  )
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--image", required=True)
  parser.add_argument("--scale", type=int, default=2)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--cg-iters", type=int, default=800)
  parser.add_argument("--psf-sigma", type=float, default=1.0)
  parser.add_argument(
    "--psf-mode",
    default="matched",
    choices=["matched", "fixed"],
    help="matched scales the anti-alias blur with the scale factor (fixes the "
    "checkerboard); fixed uses a constant 5x5 psf (the old behaviour).",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-size", type=int, default=512)
  parser.add_argument("--save-panel", default=None)
  parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  print(f"device: {args.device}")
  clean = load_image(args.image, device=args.device)
  clean = _downscale_to_max(clean, args.max_size)

  # The UNet halves the resolution several times and its skip connections
  # require matching sizes, so both dimensions must be divisible by the
  # downsample factor. Cropping to a multiple of scale * factor keeps both
  # the SR operator and the UNet path integer-sized.
  multiple = args.scale * UNET_DOWNSAMPLE
  clean = _crop_to_multiple(clean, multiple)
  print(f"image: {tuple(clean.shape)} (cropped to multiple of {multiple})")

  # A matched PSF scales the anti-alias blur with the downsampling factor, which
  # suppresses the checkerboard artifact a fixed narrow PSF leaves at high scale.
  if args.psf_mode == "matched":
    kernel = matched_psf(args.scale)
  else:
    kernel = gaussian_psf(5, args.psf_sigma)
  operator = BlurDownsampleOperator(kernel.to(args.device), scale=args.scale)
  prior = PretrainedScorePrior(args.checkpoint, device=args.device, use_fp16=False)
  schedule = DiscreteLinearSchedule(1000, 1e-4, 2e-2)
  config = ReconstructionConfig(num_steps=args.steps, cg_iters=args.cg_iters)
  generator = torch.Generator(device=args.device).manual_seed(args.seed)

  print(f"psf mode: {args.psf_mode}")
  print("reconstructing...")
  result = reconstruct_demo(
    operator,
    prior,
    clean,
    config=config,
    schedule=schedule,
    generator=generator,
  )

  diagnostics = compute_diagnostics(result.reconstruction, result.floor, clean)

  print("")
  print(f"  floor : {result.floor_psnr:6.2f} dB  SSIM {result.floor_ssim:.3f}")
  print(f"  recon : {result.recon_psnr:6.2f} dB  SSIM {result.recon_ssim:.3f}")
  print(f"  gap   : {result.gap_db:+6.2f} dB")
  print(f"  A x_hat = y residual: {result.consistency_residual:.2e}")
  print(f"  floor grid energy : {diagnostics.floor_checkerboard:.4e}")
  print(f"  recon grid energy : {diagnostics.recon_checkerboard:.4e}")
  if diagnostics.prior_added_artifacts:
    print("  WARNING: reconstruction is markedly more gridded than the floor "
          "(prior amplifying artifacts).")

  if result.gap_db > 0.3: # type: ignore
    print("\n  Prior ADDS signal above the floor on this in-domain image.")
  else:
    print("\n  Gap is small; check that the image is natural (in-domain).")

  if args.save_panel:
    _save_panel(result, clean, diagnostics.difference, args.save_panel)
    print(
      "  saved panel (clean | degraded | floor | recon | difference) -> "
      f"{args.save_panel}"
    )


if __name__ == "__main__":
  main()