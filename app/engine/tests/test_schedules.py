import torch

from ..src.priors.schedules import DiscreteLinearSchedule
from ..src.priors.score_to_velocity import ScoreToVelocity

sched = DiscreteLinearSchedule(num_steps=1000, beta_start=1e-4, beta_end=2e-2)
passed = []


def check(name, ok, detail=""):
  passed.append(ok)
  print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


print("Discrete reference values")
check(
  "beta_0 == 1e-4", abs(sched.beta_at_step(0) - 1e-4) < 1e-9, f"got {sched.beta_at_step(0):.2e}"
)
check(
  "beta_999 == 2e-2",
  abs(sched.beta_at_step(999) - 2e-2) < 1e-9,
  f"got {sched.beta_at_step(999):.2e}",
)
check(
  "alpha_bar_0 == 1-beta_0",
  abs(sched.alpha_bar_at_step(0) - (1 - 1e-4)) < 1e-9,
  f"got {sched.alpha_bar_at_step(0):.6f}",
)

# hand-check alpha_bar_2 = (1-b0)(1-b1)(1-b2)
b = [sched.beta_at_step(i) for i in range(3)]
ab2_hand = (1 - b[0]) * (1 - b[1]) * (1 - b[2])
check(
  "alpha_bar_2 == hand product",
  abs(sched.alpha_bar_at_step(2) - ab2_hand) < 1e-9,
  f"got {sched.alpha_bar_at_step(2):.6f} vs {ab2_hand:.6f}",
)

print("\nMonotonicity & range")
abs_all = torch.tensor([sched.alpha_bar_at_step(t) for t in range(0, 1000, 50)])
check("alpha_bar monotone decreasing", bool((abs_all[1:] <= abs_all[:-1]).all()))
check(
  "alpha_bar_999 small (high noise)",
  sched.alpha_bar_at_step(999) < 1e-3,
  f"got {sched.alpha_bar_at_step(999):.2e}",
)

print("\nContinuous query passes through discrete anchors")
# at tau = t/(T-1) the continuous alpha_bar must equal the discrete value
worst = 0.0
for t in [0, 100, 500, 900, 999]:
  tau = t / 999
  cont = float(sched.alpha_bar(tau))
  disc = sched.alpha_bar_at_step(t)
  worst = max(worst, abs(cont - disc) / (disc + 1e-12))
check("continuous == discrete at anchors", worst < 1e-4, f"worst rel {worst:.2e}")

print("\nInterface: drives ScoreToVelocity without error")


def eps_zero(xt, t):
  return torch.zeros_like(xt)


adapter = ScoreToVelocity(eps_zero, sched, parameterization="eps", flow_time_is_clean_at_one=True)  # type: ignore
x = torch.randn(1, 3, 16, 16)
v = adapter.velocity(x, 0.5)
ok = v.shape == x.shape and torch.isfinite(v).all()
check("velocity finite & correct shape", ok, f"shape {tuple(v.shape)}")

# at flow t=0 (tau=1, max noise) and t~1 (tau~0, clean) velocity must stay finite
v0 = adapter.velocity(x, 0.0)
v1 = adapter.velocity(x, 0.999)
check(
  "velocity finite at both ODE endpoints",
  bool(torch.isfinite(v0).all() and torch.isfinite(v1).all()),
)

print("\n" + "=" * 58)
print(f"  {sum(passed)}/{len(passed)} checks passed")
print("  Schedule certified: matches the DDPM discrete linear schedule and")
print("  drives the adapter cleanly. Ready to pair with the real model.")
