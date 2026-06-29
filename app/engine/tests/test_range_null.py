"""Verify RangeNullDecomposition against the theorems it claims to implement.
Each test maps to a result in the research doc §3/§4."""

import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/claude")

from ..src.operators import BlurDownsampleOperator, gaussian_psf
from ..src.operators import ForwardOperator
from engine.src.decomposition.range_null import RangeNullDecomposition

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}\n")
torch.backends.cudnn.enabled = False

# --- a certified operator and a measurement ---
op = BlurDownsampleOperator(gaussian_psf(5, 1.0).to(device), scale=2)
H = W = 64
yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
x_true = (0.5 + 0.5 * torch.sin(6 * xx) * torch.cos(5 * yy)).view(1, 1, H, W).to(device)
y = op.A(x_true)

dec = RangeNullDecomposition(op, y, image_shape=(1, 1, H, W), cg_iters=800)

# === Test 1: §3 theorem — A x_hat = y for ARBITRARY z ===
print("[1] Consistency theorem: A·reconstruct(z) == y for arbitrary z")
worst = 0.0
for i in range(5):
    z = torch.randn(1, 1, H, W, device=device) * (i + 1)  # wildly different z's
    x_hat = dec.reconstruct(z)
    res = dec.consistency_residual(x_hat)
    worst = max(worst, res)
    print(f"    z-scale {i+1}:  ||A x_hat - y||/||y|| = {res:.2e}")
print(f"    -> {'PASS' if worst < 5e-4 else 'FAIL'} (worst {worst:.2e}); "
      f"model output is annihilated by A as proven\n")

# === Test 2: projectors are idempotent (P_N^2 = P_N) ===
print("[2] Projector idempotency: P_N(P_N z) == P_N z")
z = torch.randn(1, 1, H, W, device=device)
pn = dec.project_null(z)
pnpn = dec.project_null(pn)
idem = (pn - pnpn).norm().item() / (pn.norm().item() + 1e-12)
print(f"    rel diff = {idem:.2e}  -> {'PASS' if idem < 5e-4 else 'FAIL'}\n")

# === Test 3: complementarity — P_R z + P_N z == z ===
print("[3] Complementarity: P_R z + P_N z == z")
comp = (dec.project_range(z) + dec.project_null(z) - z).norm().item() / (z.norm().item() + 1e-12)
print(f"    rel diff = {comp:.2e}  -> {'PASS' if comp < 5e-4 else 'FAIL'}\n")

# === Test 4: orthogonality — <P_R a, P_N b> ~ 0 ===
print("[4] Subspace orthogonality: <P_R a, P_N b> == 0")
a = torch.randn(1, 1, H, W, device=device)
b = torch.randn(1, 1, H, W, device=device)
ortho = (dec.project_range(a) * dec.project_null(b)).sum().item()
scale = (dec.project_range(a).norm() * dec.project_null(b).norm()).item() + 1e-12
print(f"    normalized inner product = {ortho/scale:.2e}  "
      f"-> {'PASS' if abs(ortho/scale) < 5e-4 else 'FAIL'}\n")

# === Test 5: null_velocity lands in the null space (A P_N v ~ 0) ===
print("[5] Null velocity: A·null_velocity(v) == 0")
v = torch.randn(1, 1, H, W, device=device)
anv = op.A(dec.null_velocity(v)).norm().item() / (op.A(v).norm().item() + 1e-12)
print(f"    ||A P_N v|| / ||A v|| = {anv:.2e}  -> {'PASS' if anv < 5e-4 else 'FAIL'}\n")

# === Test 6: the constructor gate — a broken operator must be REFUSED ===
print("[6] Constructor gate: broken operator must raise, not build silently")

class BrokenOp(ForwardOperator):
  """A^T deliberately not the adjoint of A (wrong padding) -> must be rejected."""
  def __init__(self, base): self.base = base
  def A(self, x): return self.base.A(x)
  def AT(self, y):  # corrupt the adjoint: scale it, breaking <Ax,y>=<x,ATy>
    return self.base.AT(y) * 1.7

broken = BrokenOp(op)
try:
  RangeNullDecomposition(broken, y, image_shape=(1, 1, H, W))
  print("    -> FAIL (built on a broken operator!)\n")
except ValueError as e:
  print(f"    -> PASS (refused): {str(e)[:60]}...\n")

print("=" * 60)
print("If all six pass, the module faithfully implements §3/§4 and")
print("the operator gate is live. Ready to feed a real flow prior.")
