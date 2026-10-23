"""
check_unified.py  (run from Ark_Plus/Pretraining/ on a GPU node, before any training)

Swin-B from ImageNet (Ark recipe):
  python check_unified.py --arch base --ckpt https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth
Swin-L from Ark+:
  python check_unified.py --arch large_768 --ckpt /path/Ark6_swinLarge768_ep50.pth.tar
Add  --ct /path/one_volume.npy  to also write drr_check.png

1. prints checkpoint structure (+ --projector_features / --use_mlp if it is an Ark checkpoint)
2. verifies ArkUnifiedSwin == original ArkSwinTransformer on 2D input with the same weights
3. 3D forward/backward smoke test with peak-memory report
4. optional: DRR preview from a real volume
"""
import argparse
import time

import numpy as np
import torch

from unified_swin import ArkUnifiedSwin, load_ark_checkpoint, torch_load, pick_state_dict

ARCHS = {  # name: (unified kwargs, timm kwargs for the original class, 2D input size, default CT size, DRR size)
    "base":      (dict(embed_dim=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window2d=7),
                  dict(img_size=224, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
                  224, "128,224,224", 224),
    "base_384":  (dict(embed_dim=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window2d=12),
                  dict(img_size=384, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
                  384, "128,192,192", 384),
    "large_768": (dict(embed_dim=192, depths=(2, 2, 18, 2), heads=(6, 12, 24, 48), window2d=12),
                  dict(img_size=768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48)),
                  768, "128,192,192", 768),
}

p = argparse.ArgumentParser()
p.add_argument("--arch", choices=list(ARCHS), default="base")
p.add_argument("--ckpt", required=True)
p.add_argument("--key", default="auto")
p.add_argument("--vol", default=None, help="D,H,W for the 3D smoke test (default depends on --arch)")
p.add_argument("--batch3d", type=int, default=1)
p.add_argument("--patch_depth", type=int, default=4)
p.add_argument("--window_depth", type=int, default=4)
p.add_argument("--ct", default=None, help="optional .npy/.nii.gz volume for a DRR preview")
a = p.parse_args()
uni_kw, ref_kw, size2d, vol_default, drr_size = ARCHS[a.arch]

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ------------------------------------------------------------------ 1. structure
ck = torch_load(a.ckpt)
print("top-level keys:", list(ck.keys())[:12] if isinstance(ck, dict) else type(ck))
sd = pick_state_dict(ck, a.key)
backbone = ("patch_embed", "layers.", "norm.", "absolute_pos_embed", "head.", "head_dist.")
print("\nnon-backbone tensors:")
for k, v in sd.items():
    if not k.startswith(backbone):
        print(f"  {k:45s} {tuple(v.shape)}")
heads = sorted({int(k.split(".")[1]) for k in sd if k.startswith("omni_heads.") and k.endswith(".weight")})
ncls = [sd[f"omni_heads.{i}.weight"].shape[0] for i in heads]
if "projector.0.weight" in sd:
    proj, mlp = sd["projector.0.weight"].shape[0], True
elif "projector.weight" in sd:
    proj, mlp = sd["projector.weight"].shape[0], False
else:
    proj, mlp = None, False
if ncls:
    print(f"\n=> Ark checkpoint. heads: {ncls}.  use: --projector_features {proj} --use_mlp {mlp}")
else:
    print("\n=> no Ark heads/projector (ImageNet-style). Choose --projector_features yourself (Ark used 1376).")
n_rows = sd["layers.0.blocks.0.attn.relative_position_bias_table"].shape[0]
win = int(round((n_rows ** 0.5 + 1) / 2))
print(f"=> checkpoint window size: {win}  (--arch {a.arch} expects {uni_kw['window2d']})")
assert win == uni_kw["window2d"], "window mismatch: pick the --arch / --model that matches this checkpoint"

# ------------------------------------------------------------------ 2. 2D equivalence
from models import ArkSwinTransformer   # original Ark+ class (timm 0.5.4)

ref = ArkSwinTransformer(ncls, proj, mlp, **ref_kw)
print("\nref load:", ref.load_state_dict({k: v for k, v in sd.items() if "attn_mask" not in k}, strict=False))
uni = ArkUnifiedSwin(ncls, proj, mlp, window_depth=a.window_depth, patch3d=(a.patch_depth, 4, 4),
                     use_checkpoint=False, **uni_kw)
load_ark_checkpoint(uni, sd)

ref.eval().to(dev)
uni.eval().to(dev)
x = torch.randn(2, 3, size2d, size2d, device=dev)
with torch.no_grad():
    fr, fu = ref.forward_features(x), uni.forward_features(x)
    print(f"encoder features: max|diff| {(fr - fu).abs().max().item():.2e} (scale {fr.abs().max().item():.2e})")
    for h in range(len(ncls)):
        _, lr_ = ref(x, h)
        _, lu = uni(x, h)
        print(f"head {h}: logit max|diff| {(lr_ - lu).abs().max().item():.2e}")
print("=> expect ~1e-4 or smaller; ~1e-1 means a weight-mapping problem")
del ref
if dev == "cuda":
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ 3. 3D smoke test
D, H, W = map(int, (a.vol or vol_default).split(","))
for layer in uni.layers:
    layer.use_ckpt = True
uni.train()
v = torch.randn(a.batch3d, 1, D, H, W, device=dev)
if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
    f = uni.forward_features(v)
    loss = f.float().pow(2).mean()
loss.backward()
if dev == "cuda":
    torch.cuda.synchronize()
print(f"\n3D {a.batch3d}x{D}x{H}x{W}: stage-1 tokens {D // a.patch_depth}x{H // 4}x{W // 4}, "
      f"fwd+bwd {time.time() - t0:.1f}s, proj3d gets grad: {uni.patch_embed.proj3d.weight.grad is not None}")
if dev == "cuda":
    print(f"peak GPU memory (bf16, checkpointing, model+grads incl.): {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(f"=> in datasets_config.yaml use volume_size [{D}, {H}, {W}] and drr_size {drr_size} (= --input_size)")

# ------------------------------------------------------------------ 4. DRR preview
if a.ct:
    import cv2
    from ct_utils import load_volume, make_drr
    vol = load_volume(a.ct)
    print(f"\nCT {a.ct}: shape {vol.shape}, dtype {vol.dtype}, range [{vol.min():.0f}, {vol.max():.0f}]")
    if vol.dtype == np.uint8:
        print("pre-windowed volume (prep_ctrate.py output): check its DRR/previews from prep_ctrate.py --inspect instead")
    else:
        cv2.imwrite("drr_check.png", make_drr(vol.astype(np.float32), max(drr_size, 512)))
        print("wrote drr_check.png: head up, heart on the image right (patient's left), bones bright")
