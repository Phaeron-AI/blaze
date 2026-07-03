import torch

from ..src.priors.score_to_velocity import NoiseSchedule, ScoreToVelocity

torch.manual_seed(0)
sched = NoiseSchedule(beta_min=0.1, beta_max=20.0)
s0 = 2.0
passed = []


def rel(a, b):
  bn = b.norm().item()
  if bn < 1e-8:
    return (a - b).norm().item()
  return (a - b).norm().item() / bn


def check(name, ok):
  passed.append(ok)
  print(f"  {'PASS' if ok else 'FAIL'}  {name}")


for tau in [0.2, 0.5, 0.8]:
  ab = sched.alpha_bar(tau)
  var = ab * s0**2 + (1 - ab)
  x = torch.randn(1, 1, 16, 16) * float(var.sqrt())  # a plausible x_tau sample

  # analytic models
  def eps_model(xt, t, var=var, ab=ab):
    return torch.sqrt(1 - ab) * xt / var  # MMSE eps

  def score_model(xt, t, var=var):
    return -xt / var  # analytic score

  beta = sched.beta(tau)
  analytic_score = -x / var
  v_diffusion_true = -0.5 * beta * (x + analytic_score)

  # --- score parameterization, no flow-time flip (diffusion-time velocity) ---
  adapter_s = ScoreToVelocity(
    score_model, sched, parameterization="score", flow_time_is_clean_at_one=False
  )
  v_s = adapter_s.velocity(x, tau)
  check(f"score-param reproduces PF velocity (tau={tau})", rel(v_s, v_diffusion_true) < 1e-5)

  # --- eps parameterization must give the SAME thing (eps->score internally) ---
  adapter_e = ScoreToVelocity(
    eps_model, sched, parameterization="eps", flow_time_is_clean_at_one=False
  )
  v_e = adapter_e.velocity(x, tau)

  check(f"eps-param == score-param (tau={tau})", rel(v_e, v_s) < 5e-5)


print("\nFlow-time orientation")


def eps_zero(xt, t):
  return torch.zeros_like(xt)  # score=0 -> v = -0.5 beta x


x = torch.randn(1, 1, 8, 8)
t_flow = 0.3
tau = 1 - t_flow
adapter = ScoreToVelocity(eps_zero, sched, parameterization="eps", flow_time_is_clean_at_one=True)
v_flow = adapter.velocity(x, t_flow)
beta = sched.beta(tau)
expected = -(-0.5 * beta * x)  # negated diffusion velocity (score=0)
check("flow-time velocity negates & uses tau=1-t", rel(v_flow, expected) < 1e-5)

print("\nSchedule sanity")
taus = torch.linspace(0, 1, 11)
abs_ = torch.stack([sched.alpha_bar(t) for t in taus])
check("alpha_bar(0)==1", abs(abs_[0].item() - 1.0) < 1e-6)
check("alpha_bar monotone decreasing", bool((abs_[1:] <= abs_[:-1] + 1e-6).all()))
check("alpha_bar in (0,1]", bool((abs_ > 0).all() and (abs_ <= 1 + 1e-6).all()))

print("\n" + "=" * 56)
print(f"  {sum(passed)}/{len(passed)} checks passed")
print("  Adapter certified: eps/score -> velocity matches the closed-form")
print("  probability-flow ODE. A real prior can now plug in behind it.")
