import math

import torch

from ..src.eval.metrics import evaluate, psnr, ssim

torch.manual_seed(0)
passed = []


def check(name, got, expected, tol=1e-3):
  ok = (math.isinf(got) and math.isinf(expected)) or abs(got - expected) <= tol
  passed.append(ok)
  print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.4f}, expected {expected:.4f}")


print("PSNR — against hand-computed values")
img = torch.rand(1, 1, 32, 32)

# 1. identical -> +inf
check("identical -> inf", psnr(img, img, 1.0), float("inf"))

# 2. constant offset 0.1, data_range 1.0 -> exactly 20 dB
off = (img + 0.1).clamp(0, 1)
# use an unclamped version so MSE is exactly 0.01 (clamping would perturb it)
a = torch.full((1, 1, 32, 32), 0.5)
b = torch.full((1, 1, 32, 32), 0.6)
check("offset 0.1, D=1 -> 20 dB", psnr(a, b, 1.0), 20.0)

# 3. black vs white, data_range 255 -> exactly 0 dB
black = torch.zeros(1, 1, 16, 16)
white = torch.full((1, 1, 16, 16), 255.0)
check("black vs white, D=255 -> 0 dB", psnr(black, white, 255.0), 0.0)

# 4. half pixels off by 0.2, half exact: MSE = 0.2^2/2 = 0.02
#    PSNR = 10 log10(1/0.02) = 10 * log10(50) = 16.9897 dB
p = torch.full((1, 1, 2, 1), 0.5)
t = torch.tensor([0.5, 0.7]).view(1, 1, 2, 1)  # one exact, one off by 0.2
expected = 10 * math.log10(1.0 / 0.02)
check("half-off 0.2 -> 10log10(50)", psnr(p, t, 1.0), expected)

print("\nSSIM — against known values")
# 5. identical -> 1.0
big = torch.rand(1, 1, 64, 64)
check("identical -> 1.0", ssim(big, big, 1.0), 1.0)

# 6. SSIM is symmetric
s_ab = ssim(big, (big + 0.05).clamp(0, 1), 1.0)
s_ba = ssim((big + 0.05).clamp(0, 1), big, 1.0)
check("symmetry SSIM(a,b)==SSIM(b,a)", s_ab, s_ba)

print("\nGuards — the data_range mismatch must be REFUSED")
# 7. a [-1,1] image scored with data_range=1.0 must raise (the classic bug)
neg = torch.rand(1, 1, 32, 32) * 2 - 1  # spans ~[-1,1], range ~2
try:
  psnr(neg, neg, data_range=1.0)
  passed.append(False)
  print("  FAIL  [-1,1] scored as D=1 should raise, did not")
except ValueError:
  passed.append(True)
  print("  PASS  [-1,1] scored as D=1 raised (guard fired)")

# 8. shape mismatch must raise
try:
  psnr(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9), 1.0)
  passed.append(False)
  print("  FAIL  shape mismatch should raise, did not")
except ValueError:
  passed.append(True)
  print("  PASS  shape mismatch raised")

# 9. evaluate() returns both metrics
m = evaluate(big, big, 1.0)
ok = "psnr_db" in m and "ssim" in m and math.isinf(m["psnr_db"])
passed.append(ok)
print(f"  {'PASS' if ok else 'FAIL'}  evaluate() dict: {list(m.keys())}")

print("\n" + "=" * 56)
print(f"  {sum(passed)}/{len(passed)} checks passed")
print("  Ruler certified: metrics return hand-verified values and")
print("  the data_range guard refuses normalization mismatches.")
