"""CLI: evaluate a prior on a reconstruction task.

Examples
--------
Evaluate the pretrained ImageNet prior on a natural image (validate the machine):
    python -m engine.src.scripts.run_reconstruction \
        --checkpoint models/256x256_diffusion_uncond.pt \
        --image samples/natural.png --scale 2 --steps 50

Same on a microscopy image (expect a smaller/negative gap until Phase-3 tuning):
    python -m engine.src.scripts.run_reconstruction \
        --checkpoint models/256x256_diffusion_uncond.pt \
        --image samples/em.png --scale 2 --steps 50

The gap (recon - floor, in dB) is the answer to "does the generative prior add
signal above pure linear algebra." Every run is appended to the runs log.
"""

from __future__ import annotations

import argparse

import torch

from typing import Callable

from ..priors.score_to_velocity import ScoreToVelocity
from ..eval.run_log import RunLogger
from ..evaluate import (
  ReconstructionEvaluator,
  default_sr_operator,
  load_image,
  to_multiple,
)
from ..reconstruct import ReconstructionConfig


def build_prior_factory(checkpoint: str, device: str)-> Callable[[], ScoreToVelocity]:
  from ..priors.pretrained_score import PretrainedScorePrior
  from ..priors.schedules import DiscreteLinearSchedule

  def factory()-> ScoreToVelocity:
    prior = PretrainedScorePrior(
      checkpoint,
      device=device,
      use_fp16=False,  # fp32 for correctness-first
    )
    return prior.as_velocity_prior(DiscreteLinearSchedule(1000, 1e-4, 2e-2))

  return factory


def main() -> None:
  torch.backends.cudnn.enabled = False
  p = argparse.ArgumentParser(description="Evaluate a prior on SR reconstruction.")
  p.add_argument("--checkpoint", required=True, help="path to the prior checkpoint")
  p.add_argument("--image", required=True, help="ground-truth image (will be degraded)")
  p.add_argument("--scale", type=int, default=2, help="SR downsampling factor")
  p.add_argument("--steps", type=int, default=50, help="ODE integration steps")
  p.add_argument("--method", default="heun", choices=["euler", "heun"])
  p.add_argument("--cg-iters", type=int, default=800)
  p.add_argument("--psf-sigma", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--runs-log", default="runs/reconstruction.jsonl")
  p.add_argument("--check-consistency", action="store_true")
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = p.parse_args()

  print(f"device: {args.device}")
  gt = load_image(args.image, device=args.device)
  gt = to_multiple(gt, args.scale)  # SR needs H,W divisible by scale
  print(f"image: {tuple(gt.shape)} (cropped to multiple of {args.scale})")

  operator = default_sr_operator(
    scale=args.scale,
    psf_sigma=args.psf_sigma,
    device=args.device,
  )
  config = ReconstructionConfig(
    num_steps=args.steps,
    method=args.method,
    cg_iters=args.cg_iters,
    check_consistency=args.check_consistency,
  )
  evaluator = ReconstructionEvaluator(
    operator=operator,
    prior_factory=build_prior_factory(args.checkpoint, args.device),
    config=config,
    logger=RunLogger(args.runs_log),
  )

  print("reconstructing...")
  result = evaluator.evaluate(gt, seed=args.seed, note=args.image)
  print("\n" + result.summary())
  print(f"\nlogged to {args.runs_log}")


if __name__ == "__main__":
  main()
