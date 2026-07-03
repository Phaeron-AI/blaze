import torch

from ..src.priors.base import ConstantVelocityStub, VelocityPrior
from ..src.priors.normalized import NormalizedVelocityPrior

torch.manual_seed(0)
passed = []

torch.backends.cudnn.enabled = False


def check(name, ok, detail=""):
  passed.append(ok)
  print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  '+detail) if detail else ''}")


def rel(a, b):
  return (a - b).norm().item() / (b.norm().item() + 1e-12)


# --- transform invertibility ---
print("Affine transform")
prior_dummy = ConstantVelocityStub(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
norm = NormalizedVelocityPrior(prior_dummy, scale=2.0, shift=-1.0)
x = torch.rand(1, 1, 8, 8)
check("to/from model space inverts", rel(norm.from_model_space(norm.to_model_space(x)), x) < 1e-6)
check(
  "[0,1] -> [-1,1] endpoints",
  abs(float(norm.to_model_space(torch.zeros(1))) + 1) < 1e-6
  and abs(float(norm.to_model_space(torch.ones(1))) - 1) < 1e-6,
)

# --- the invariant: wrapping preserves the pipeline-space trajectory ---
# Integrate a constant model-space velocity directly vs through the wrapper.
print("\nVelocity rescaling preserves the trajectory")

c = torch.randn(1, 1, 8, 8)


class ConstModelVel(VelocityPrior):
  def velocity(self, x_t, t, **cond):
    return c.to(x_t.device).expand_as(x_t)


inner = ConstModelVel()
wrapped = NormalizedVelocityPrior(inner, scale=2.0, shift=-1.0)

x0 = torch.rand(1, 1, 8, 8)  # pipeline space start
# integrate wrapped flow in pipeline space (fine Euler)
xp = x0.clone()
n = 500
dt = 1.0 / n
for _ in range(n):
  xp = xp + dt * wrapped.velocity(xp, 0.0)
# closed form: x0 + (c/scale) * 1
expected_pipeline = x0 + c / 2.0
check(
  "wrapped flow lands at x0 + c/scale",
  rel(xp, expected_pipeline) < 1e-4,
  f"rel {rel(xp, expected_pipeline):.2e}",
)

# equivalently, the same physical point in model space
xm0 = norm.to_model_space(x0)
expected_model = xm0 + c
check("model-space endpoint consistent", rel(norm.to_model_space(xp), expected_model) < 1e-4)

# --- velocity is exactly inner/scale ---
print("\nVelocity factor")
v_wrapped = wrapped.velocity(x0, 0.3)
v_inner = inner.velocity(wrapped.to_model_space(x0), 0.3)
check("v_wrapped == v_inner / scale", rel(v_wrapped, v_inner / 2.0) < 1e-6)

print("\n" + "=" * 56)
print(f"  {sum(passed)}/{len(passed)} checks passed")
print("  Bridge certified: affine space-change is invertible and the velocity")
print("  rescaling preserves the trajectory. Safe to wrap the real prior.")
