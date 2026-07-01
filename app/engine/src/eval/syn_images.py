from __future__ import annotations
import torch


def make_test_image(kind: str = "shapes", size: int = 256, seed: int = 0, device: str = "cpu") -> torch.Tensor:
  H = W = size
  yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")

  if kind == "gradient":
    img = torch.stack([xx, yy, (xx + yy) / 2])

  elif kind == "checker":
    c = ((xx * 8).floor() + (yy * 8).floor()) % 2
    img = torch.stack([c, c, c])

  elif kind == "freq":
    img = torch.stack([
      0.5 + 0.4 * torch.sin(40 * xx),
      0.5 + 0.4 * torch.cos(40 * yy),
      0.5 + 0.4 * torch.sin(30 * (xx + yy)),
    ])

  elif kind == "shapes":
    g = torch.Generator().manual_seed(seed)
    img = torch.stack([xx, yy, 0.5 * (xx + yy)])          # gradient background
    for _ in range(6):
      cx, cy = torch.rand(2, generator=g).tolist()
      rad = 0.08 + 0.12 * torch.rand(1, generator=g).item()
      col = torch.rand(3, generator=g)
      mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < rad ** 2
      for ch in range(3):
        img[ch] = torch.where(mask, col[ch], img[ch])
  else:
    raise ValueError(f"unknown kind {kind!r}; "
                      "use gradient|checker|freq|shapes")

  return img.unsqueeze(0).clamp(0, 1).to(device)


def save_test_image(kind: str, path: str, size: int = 256, seed: int = 0) -> None:
  from PIL import Image
  import numpy as np
  img = make_test_image(kind, size=size, seed=seed)[0]        # (3,H,W)
  arr = (img.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
  Image.fromarray(arr).save(path)


if __name__ == "__main__":
  import argparse, os
  p = argparse.ArgumentParser()
  p.add_argument("--out", default="samples")
  p.add_argument("--size", type=int, default=256)
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  for k in ("gradient", "checker", "freq", "shapes"):
    path = os.path.join(args.out, f"synthetic_{k}.png")
    save_test_image(k, path, size=args.size)
    print(f"wrote {path}")