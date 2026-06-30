"""Validate NullSpaceFlowSampler against EXACT ground truth, using a non-stiff
stub so solver stiffness can't be mistaken for a loop bug.

ConstantVelocityStub: v = x1 - x0 (constant). Under null-space projection the
integrated state is
    x_hat(1) = A^dagger y + P_N z0 + P_N (x1 - x0),
so its null component is exactly  P_N (z0 + x1 - x0).
"""

import torch

from ..src.operators.blur_downsample import BlurDownsampleOperator, gaussian_psf
from ..src.decomposition.range_null import RangeNullDecomposition
from ..src.priors.base import ConstantVelocityStub
from ..src.samplers.null_space_flow import NullSpaceFlowSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}\n")

torch.backends.cudnn.enabled = False

op = BlurDownsampleOperator(gaussian_psf(5, 1.0).to(device), scale=2)
H = W = 64
yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
x_true = (0.5 + 0.5 * torch.sin(6 * xx) * torch.cos(5 * yy)).view(1, 1, H, W).to(device)
y = op.A(x_true)
dec = RangeNullDecomposition(op, y, image_shape=(1, 1, H, W), cg_iters=800)

x0 = torch.zeros(1, 1, H, W, device=device)
x1 = torch.randn(1, 1, H, W, device=device)
stub = ConstantVelocityStub(x0, x1)
z0 = torch.randn(1, 1, H, W, device=device)

def rel(a, b): return ((a - b).norm() / (b.norm() + 1e-12)).item()

sampler = NullSpaceFlowSampler(stub, dec, num_steps=50, method="euler",
                               check_consistency=True, consistency_rtol=1e-2)
out = sampler.sample(z0)

print("[A] Endpoint consistency: A x_hat == y")
cons = dec.consistency_residual(out)
print(f"    ||A out - y||/||y|| = {cons:.2e}  -> {'PASS' if cons < 1e-3 else 'FAIL'}\n")

print("[B] Null component == exact prediction P_N(z0 + x1 - x0)")

predicted_null = dec.project_null(z0 + x1 - x0)
r = rel(dec.project_null(out), predicted_null)
print(f"    rel(P_N out, prediction) = {r:.2e}  -> {'PASS' if r < 1.5e-2 else 'FAIL'}\n")

print("[C] Per-step consistency invariant held (no AssertionError above)")
print(f"    -> PASS (sampler ran with check_consistency=True)\n")

print("[D] Determinism: same z0 -> identical output")
out2 = sampler.sample(z0.clone())
d = (out - out2).abs().max().item()
print(f"    max|out - out2| = {d:.2e}  -> {'PASS' if d < 1e-6 else 'FAIL'}\n")

print("=" * 60)
print("If A-D pass: integration loop correct against exact ground truth,")
print("consistency holds every step, output reproducible. Ready for a real prior.")