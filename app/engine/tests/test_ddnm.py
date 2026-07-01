"""DDNM reconstruction test on the REAL prior. Run with the checkpoint path.

    python -m engine.tests.test_ddnm <checkpoint> [image.png]

Gates (both must pass before the real-prior number is trustworthy):
  GATE  — pure oracle (knows the truth): proves the sampler SKELETON is correct.
  GATE2 — semi-oracle (shifted truth, so the projection is NON-trivial): proves
          the eps/x0 consistency coupling is correct. The pure oracle cannot test
          this, because projection is a no-op on the true image.
  REAL  — pretrained ImageNet prior. Positive gap expected on in-domain (natural)
          images; small/negative on out-of-domain (synthetic/EM) is a domain
          signal, not a bug.
"""

import sys
import math
import torch

from ..src.operators.blur_downsample import BlurDownsampleOperator, gaussian_psf
from ..src.decomposition.range_null import RangeNullDecomposition
from ..src.priors.schedules import DiscreteLinearSchedule
from ..src.samplers.ddnm import DDNMSampler
from ..src.reconstruct import DDNMReconstructor, ReconstructionConfig
from ..src.eval.metrics import psnr, ssim
from ..src.eval.syn_images import make_test_image

ckpt = sys.argv[1] if len(sys.argv) > 1 else None
if not ckpt:
  print("usage: pass checkpoint path as arg 1")
  sys.exit(1)
image_path = sys.argv[2] if len(sys.argv) > 2 else None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

torch.backends.cudnn.enabled = False

# --- test image: real photo if given, else synthetic 'shapes' (mixed structure) ---
if image_path:
  from ..src.evaluate import load_image, to_multiple
  x_true = to_multiple(load_image(image_path, device), 2)
else:
  x_true = make_test_image("shapes", size=256, device=device)
print(f"image: {tuple(x_true.shape)}")

op = BlurDownsampleOperator(gaussian_psf(5, 1.0).to(device), scale=2)
y = op.A(x_true)
sched = DiscreteLinearSchedule(1000, 1e-4, 2e-2)
shape = tuple(x_true.shape)


def report(tag, out, dec):
  fl = psnr(dec.pinv_reconstruction.clamp(0, 1), x_true)
  rc = psnr(out.clamp(0, 1), x_true)
  print(f"  {tag}: floor {fl:.2f} dB | recon {rc:.2f} dB | gap {rc - fl:+.2f} dB "
        f"| consistency {dec.consistency_residual(out):.2e}")
  return rc - fl


# === GATE: pure oracle -> proves the sampler skeleton ===
print("[GATE] oracle prior (knows truth) -> must give large positive gap")
dec = RangeNullDecomposition(op, y, shape, cg_iters=800)
x0_model_true = 2 * x_true - 1


def oracle_eps(x_t, t):
  ab = float(sched.alpha_bar_at_step(t))
  return (x_t - math.sqrt(ab) * x0_model_true) / max(math.sqrt(1 - ab), 1e-8)


sampler = DDNMSampler(oracle_eps, dec, sched, num_steps=100, check_consistency=True)
g = torch.Generator(device=device).manual_seed(0)
gate_gap = report("oracle", sampler.sample(generator=g), dec)
assert gate_gap > 5.0, "GATE FAILED: sampler skeleton is not correct on this machine"
print("  -> GATE PASS: sampler skeleton is correct.\n")

# === GATE2: semi-oracle -> projection is NON-trivial, exercises eps/x0 coupling
# that the pure oracle (no-op projection on the true image) structurally cannot. ===
print("[GATE2] semi-oracle (shifted truth -> projection matters) -> positive gap")
x0_model_shifted = 2 * (x_true * 0.9 + 0.05) - 1


def semi_oracle_eps(x_t, t):
  ab = float(sched.alpha_bar_at_step(t))
  return (x_t - math.sqrt(ab) * x0_model_shifted) / max(math.sqrt(1 - ab), 1e-8)


sampler2 = DDNMSampler(semi_oracle_eps, dec, sched, num_steps=100, check_consistency=True)
g = torch.Generator(device=device).manual_seed(0)
gate2_gap = report("semi-oracle", sampler2.sample(generator=g), dec)
assert gate2_gap > 3.0, "GATE2 FAILED: eps/x0 consistency coupling is broken"
print("  -> GATE2 PASS: projection/eps coupling is correct.\n")

# === REAL prior ===
print("[REAL] pretrained ImageNet prior")
from ..src.priors.pretrained_score import PretrainedScorePrior

prior = PretrainedScorePrior(ckpt, device=device, use_fp16=False)
recon = DDNMReconstructor(op, prior.eps_at_step, sched,
                          ReconstructionConfig(num_steps=100, cg_iters=800))
g = torch.Generator(device=device).manual_seed(0)
out = recon.reconstruct(y, shape, generator=g)
dec_real = RangeNullDecomposition(op, y, shape, cg_iters=800)
real_gap = report("real", out["x_hat"], dec_real)

print("\n" + "=" * 60)
if real_gap > 0.3:
  print("  Prior ADDS signal above the floor. Pipeline + prior both working.")
elif image_path:
  print("  Neutral/negative on a real image — check in-domain-ness; may need")
  print("  domain fine-tuning (Phase 3).")
else:
  print("  Neutral/negative on SYNTHETIC (out-of-domain for ImageNet) — expected.")
  print("  Try a real natural photo as arg 2 to see the in-domain positive gap.")