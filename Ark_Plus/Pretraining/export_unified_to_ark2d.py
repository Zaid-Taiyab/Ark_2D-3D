"""
export_unified_to_ark2d.py  (run from Ark_Plus/Pretraining/, needs timm 0.5.4 like the rest of Ark)

Converts a checkpoint saved by the unified 2D/3D model (--model unified_swin_*) into the exact format of an
ORIGINAL Ark+ checkpoint, so Ark's own Finetuning / Linearprobing code can evaluate it on outside datasets
without any change, exactly as it evaluates an original Ark+ run.

Per state dict ('state_dict' = student, 'teacher'):
  patch_embed.proj2d.*            -> patch_embed.proj.*          (the 2D stem the X-ray path always used)
  *.relative_position_bias_table  -> its middle depth plane      (the plane X-rays read; Part 5.10 of the guide)
  patch_embed.proj3d.*, patch_embed.dim_embed_3d                 -> dropped (3D-only)
  relative_position_index / attn_mask buffers and the unused timm head -> rebuilt from a fresh timm model
  projector, omni_heads (including any CT / DRR heads)           -> kept as they are
The 'module.' prefix (DataParallel) is kept if present. Optimizer / scheduler state is dropped.
The result is verified by loading it with strict=True into the original ArkSwinTransformer.

  python export_unified_to_ark2d.py --arch base --ckpt Models/.../Ark_Plus_...pth.tar --out exported_2d.pth.tar
"""
import argparse

import torch

from models import ArkSwinTransformer
from unified_swin import torch_load

ARCHS = {
    "base": dict(img_size=224, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
    "base_384": dict(img_size=384, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
    "large_768": dict(img_size=768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48)),
}


def model_shape(sd):
    heads = sorted({int(k.split(".")[1]) for k in sd if k.startswith("omni_heads.") and k.endswith(".weight")})
    ncls = [sd[f"omni_heads.{i}.weight"].shape[0] for i in heads]
    if "projector.0.weight" in sd:
        return ncls, sd["projector.0.weight"].shape[0], True
    if "projector.weight" in sd:
        return ncls, sd["projector.weight"].shape[0], False
    return ncls, None, False


def convert(sd_in, arch):
    prefix = "module." if any(k.startswith("module.") for k in sd_in) else ""
    sd = {(k[len(prefix):] if prefix and k.startswith(prefix) else k): v for k, v in sd_in.items()}
    ncls, proj, mlp = model_shape(sd)
    ref = ArkSwinTransformer(ncls, proj, mlp, **ARCHS[arch])
    ref_sd = ref.state_dict()
    out, notes = {}, []
    for k, v in sd.items():
        if k.startswith("patch_embed.proj3d.") or k == "patch_embed.dim_embed_3d":
            continue
        if k.startswith("patch_embed.proj2d."):
            k = k.replace("patch_embed.proj2d.", "patch_embed.proj.")
        if k.endswith("relative_position_bias_table"):
            n2d = ref_sd[k].shape[0]
            planes = v.shape[0] // n2d
            assert planes * n2d == v.shape[0] and planes % 2 == 1, f"{k}: unexpected table shape {tuple(v.shape)}"
            mid = planes // 2                                    # = W_d - 1, the delta_d = 0 plane
            v = v[mid * n2d:(mid + 1) * n2d].clone()
        out[k] = v
    for k, v in ref_sd.items():                                  # buffers + unused timm head, from the fresh model
        if k not in out:
            out[k] = v.clone()
            notes.append(k)
    extra = [k for k in out if k not in ref_sd]
    assert not extra, f"keys not in the original model: {extra[:5]}"
    ref.load_state_dict(out, strict=True)                        # proves the format is exactly the original one
    kinds = sorted({n.split(".")[-1] for n in notes})
    print(f"   ok: {len(out)} tensors, heads {ncls}, projector {proj} (mlp={mlp}); rebuilt from timm: {kinds}")
    return {prefix + k: v for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint saved by a unified_swin_* run")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", choices=list(ARCHS), default="base")
    a = p.parse_args()
    ck = torch_load(a.ckpt)
    out = {k: ck[k] for k in ("epoch", "lossMIN") if k in ck}
    for key in ("state_dict", "teacher"):
        if key in ck:
            print(f"[{key}]")
            out[key] = convert(ck[key], a.arch)
    if not any(k in out for k in ("state_dict", "teacher")):
        print("[checkpoint is a bare state dict]")
        out = convert(ck, a.arch)
    torch.save(out, a.out)
    print(f"saved {a.out}  (original Ark+ format; use it wherever an original Ark+ checkpoint is accepted)")


if __name__ == "__main__":
    main()
