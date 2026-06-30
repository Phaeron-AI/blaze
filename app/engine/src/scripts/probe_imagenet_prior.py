import sys
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/256x256_diffusion_uncond.pt"

print(f"probing: {CKPT}")
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}\n")

try:
  # weights_only=True is safer and avoids unpickling arbitrary code; the
  # guided-diffusion checkpoints are plain tensor state-dicts so this works.
  sd = torch.load(CKPT, map_location="cpu", weights_only=True)
  print("[1] torch.load -> OK (checkpoint readable, not policy-blocked)")
except FileNotFoundError:
  print(f"[1] FILE NOT FOUND: {CKPT}")
  print("    Download 256x256_diffusion_uncond.pt and pass its path as arg 1.")
  sys.exit(1)
except Exception as e:
  print(f"[1] torch.load FAILED: {type(e).__name__}: {e}")
  print("If this is an Application Control / DLL block, it's an environment")
  print("issue to solve before any wrapper — not a code bug.")
  sys.exit(1)

if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
  sd = sd["state_dict"]

if not isinstance(sd, dict):
  print(f"[2] unexpected checkpoint type: {type(sd)}")
  sys.exit(1)

keys = list(sd.keys())
n_tensors = sum(1 for v in sd.values() if torch.is_tensor(v))
total_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
print(f"[2] state-dict: {len(keys)} entries, {n_tensors} tensors, "
  f"{total_params/1e6:.1f}M params")
print(f"first keys: {keys[:3]}")
print(f"last keys: {keys[-3:]}")

out_candidates = [k for k in keys if k.endswith("weight") and "out" in k.lower()]
final = None
for k in out_candidates:
  w = sd[k]
  if torch.is_tensor(w) and w.dim() == 4 and w.shape[0] in (3, 6):
    final = (k, tuple(w.shape))
if final:
  name, shape = final
  out_ch = shape[0]
  print(f"[3] output conv '{name}' shape {shape} -> {out_ch} channels")
  if out_ch == 6:
    print("    => learn_sigma=True confirmed: 6 = 3 eps + 3 variance.")
    print("    => WRAPPER MUST slice channels [:3] as eps before the adapter.")
  elif out_ch == 3:
    print("    => 3 channels: eps-only, no variance slicing needed.")
else:
  print("[3] could not auto-identify the output conv; inspect keys manually.")
  print("    (look for the last conv weight with out_channels 3 or 6)")
 
print("\n" + "=" * 60)
print("If [1] OK and [3] reports 6 channels: environment is clear and we")
print("know exactly what the wrapper must handle. Ready to build it.")