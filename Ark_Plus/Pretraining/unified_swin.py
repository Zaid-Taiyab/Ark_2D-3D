"""
unified_swin.py 
-------------------------------------------------
Swin-B that accepts 2D CXRs (B,3,H,W) and 3D CTs (B,1,D,H,W) with fully shared transformer
weights, plus ArkUnifiedSwin: a drop-in replacement for models.ArkSwinTransformer with the
same attribute names (projector, omni_heads) and the same forward(x, head_n) API, so engine.py,
trainer.py, EMA and checkpoint saving work unchanged.

Key design points
  * Tokens are always (B, D, H, W, C); a 2D image is D = 1.
  * Only the patch embedding is dimension-specific (Conv2d vs Conv3d, UniMiSS-style switchable).
  * ONE 3D relative-position-bias table per block. A 2D input only reads the delta_d = 0 plane,
    which is initialised from the Ark+ 2D table, so 2D outputs equal the original Ark+ model.
  * Patch merging is the timm 2x2 spatial merge (same weights); 3D depth is halved by averaging.
  * Shift masks / bias indices are built per input size (timm 0.5.4 bakes them for one size).

Parameter / buffer names mirror timm 0.5.4 SwinTransformer, so the Ark+ checkpoint loads with
only two remaps: patch_embed.proj -> proj2d (+ inflated proj3d), and bias tables -> 3D tables.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.hub import load_state_dict_from_url


# ----------------------------------------------------------------------------- helpers
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x, ws):
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws[0] * ws[1] * ws[2], C)


def window_reverse(win, ws, B, D, H, W):
    x = win.view(B, D // ws[0], H // ws[1], W // ws[2], ws[0], ws[1], ws[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D, H, W, -1)


def rel_pos_index(win, table_win, device):
    wd, wh, ww = win
    Td, Th, Tw = table_win
    coords = torch.stack(torch.meshgrid(
        torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij")).flatten(1)
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    rel[..., 0] += Td - 1
    rel[..., 1] += Th - 1
    rel[..., 2] += Tw - 1
    idx = rel[..., 0] * (2 * Th - 1) * (2 * Tw - 1) + rel[..., 1] * (2 * Tw - 1) + rel[..., 2]
    return idx.to(device)


_MASK_CACHE = {}


def shift_mask(Dp, Hp, Wp, win, shift, device):
    key = (Dp, Hp, Wp, win, shift, str(device))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    img = torch.zeros((1, Dp, Hp, Wp, 1), device=device)

    def slices(w, s):
        return [slice(None)] if s == 0 else [slice(0, -w), slice(-w, -s), slice(-s, None)]

    cnt = 0
    for d in slices(win[0], shift[0]):
        for h in slices(win[1], shift[1]):
            for w in slices(win[2], shift[2]):
                img[:, d, h, w, :] = cnt
                cnt += 1
    mw = window_partition(img, win).squeeze(-1)
    m = mw.unsqueeze(1) - mw.unsqueeze(2)
    m = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
    _MASK_CACHE[key] = m
    return m


# -------------------------------------------------------------- switchable patch embed
class SwitchablePatchEmbed(nn.Module):
    def __init__(self, embed_dim=192, patch2d=4, patch3d=(4, 4, 4), in_ch2d=3, in_ch3d=1):
        super().__init__()
        assert patch3d[1] == patch3d[2] == patch2d, "keep the in-plane 3D patch equal to the 2D patch"
        self.patch2d, self.patch3d = patch2d, tuple(patch3d)
        self.proj2d = nn.Conv2d(in_ch2d, embed_dim, patch2d, patch2d)
        self.proj3d = nn.Conv3d(in_ch3d, embed_dim, self.patch3d, self.patch3d)
        self.norm = nn.LayerNorm(embed_dim)                         # shared
        self.dim_embed_3d = nn.Parameter(torch.zeros(embed_dim))    # 3D only; 2D path stays identical to Ark+

    def forward(self, x):
        if x.ndim == 4:                                             # (B,3,H,W)
            p = self.patch2d
            x = F.pad(x, (0, (-x.shape[-1]) % p, 0, (-x.shape[-2]) % p))
            x = self.proj2d(x).unsqueeze(2).permute(0, 2, 3, 4, 1)  # (B,1,H',W',C)
            return self.norm(x)
        if x.ndim == 5:                                             # (B,1,D,H,W)
            pd, ph, pw = self.patch3d
            D, H, W = x.shape[-3:]
            x = F.pad(x, (0, (-W) % pw, 0, (-H) % ph, 0, (-D) % pd))
            x = self.proj3d(x).permute(0, 2, 3, 4, 1)               # (B,D',H',W',C)
            return self.norm(x) + self.dim_embed_3d
        raise ValueError(f"expected a 4D (2D image) or 5D (3D volume) tensor, got {tuple(x.shape)}")


# ------------------------------------------------------------------------ swin blocks
class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, table_win):
        super().__init__()
        self.heads, self.scale, self.table_win = heads, (dim // heads) ** -0.5, tuple(table_win)
        Td, Th, Tw = self.table_win
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Td - 1) * (2 * Th - 1) * (2 * Tw - 1), heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self._idx = {}

    def _bias(self, win, device):
        key = (win, str(device))
        if key not in self._idx:
            self._idx[key] = rel_pos_index(win, self.table_win, device)
        idx = self._idx[key]
        N = idx.shape[0]
        return self.relative_position_bias_table[idx.view(-1)].view(N, N, -1).permute(2, 0, 1)

    def forward(self, x, win, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._bias(win, x.device).unsqueeze(0).to(attn.dtype)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.heads, N, N) + mask[None, :, None].to(attn.dtype)
            attn = attn.view(-1, self.heads, N, N)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.window, self.shift = tuple(window), shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, heads, self.window)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        B, D, H, W, C = x.shape
        dims = (D, H, W)
        # timm 0.5.4 rule, applied per axis: a map no bigger than the window is one window, no shift
        win = tuple(min(w, s) for w, s in zip(self.window, dims))
        shift = tuple(0 if (not self.shift or s <= w) else w // 2 for w, s in zip(self.window, dims))

        shortcut = x
        x = self.norm1(x)
        pads = [(-s) % w for s, w in zip(dims, win)]
        x = F.pad(x, (0, 0, 0, pads[2], 0, pads[1], 0, pads[0]))
        Dp, Hp, Wp = x.shape[1:4]
        mask = None
        if any(shift):
            x = torch.roll(x, shifts=tuple(-s for s in shift), dims=(1, 2, 3))
            mask = shift_mask(Dp, Hp, Wp, win, shift, x.device)
        x = window_reverse(self.attn(window_partition(x, win), win, mask), win, B, Dp, Hp, Wp)
        if any(shift):
            x = torch.roll(x, shifts=shift, dims=(1, 2, 3))
        x = x[:, :D, :H, :W].contiguous()
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMergingSpatial(nn.Module):
    """timm-0.5.4-compatible 2x2 merge (same weights); 3D inputs first halve depth by averaging."""

    def __init__(self, dim, merge_depth_3d=True):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.merge_depth_3d = merge_depth_3d

    def forward(self, x):
        B, D, H, W, C = x.shape
        if self.merge_depth_3d and D > 1:                       # 2D inputs always have D == 1
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D % 2))
            x = 0.5 * (x[:, 0::2] + x[:, 1::2])
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
                       x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]], -1)   # timm order
        return self.reduction(self.norm(x))


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, window, dpr, downsample, merge_depth_3d, use_ckpt):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.blocks = nn.ModuleList([
            SwinBlock3D(dim, heads, window, shift=(i % 2 == 1), drop_path=dpr[i]) for i in range(depth)])
        self.downsample = PatchMergingSpatial(dim, merge_depth_3d) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_ckpt and self.training and torch.is_grad_enabled():
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.downsample(x) if self.downsample is not None else x


class UnifiedSwin(nn.Module):
    def __init__(self, embed_dim=192, depths=(2, 2, 18, 2), heads=(6, 12, 24, 48),
                 window2d=12, window_depth=4, patch3d=(4, 4, 4),
                 merge_depth_3d=(True, True, True), drop_path_rate=0.1, use_checkpoint=True):
        super().__init__()
        self.patch_embed = SwitchablePatchEmbed(embed_dim, 4, patch3d)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        window = (window_depth, window2d, window2d)
        self.layers = nn.ModuleList()
        for i, d in enumerate(depths):
            last = i == len(depths) - 1
            self.layers.append(Stage(
                embed_dim * 2 ** i, d, heads[i], window, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=not last, merge_depth_3d=(False if last else merge_depth_3d[i]),
                use_ckpt=use_checkpoint))
        self.num_features = embed_dim * 2 ** (len(depths) - 1)      # 1536 for Swin-L
        self.norm = nn.LayerNorm(self.num_features)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).flatten(1, 3).mean(1)                   # == timm 0.5.4 norm + avgpool


class ArkUnifiedSwin(UnifiedSwin):
    """Drop-in for models.ArkSwinTransformer (same projector / omni_heads names and API)."""

    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, **kwargs):
        super().__init__(**kwargs)
        assert num_classes_list is not None
        self.projector = None
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True),
                                               nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)
        self.omni_heads = nn.ModuleList([
            nn.Linear(self.num_features, n) if n > 0 else nn.Identity() for n in num_classes_list])

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector is not None:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj and self.projector is not None:
            x = self.projector(x)
        return x


# ----------------------------------------------------------------- checkpoint loading
def torch_load(path):
    if str(path).startswith("http"):
        return load_state_dict_from_url(url=path, map_location="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)   # torch >= 2.6 defaults to True
    except TypeError:                                                      # older torch: no weights_only arg
        return torch.load(path, map_location="cpu")


def pick_state_dict(ckpt, key="auto"):
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint is not a dict")
    order = [key] if key != "auto" else ["teacher", "state_dict", "model"]
    for k in order:
        if k in ckpt and isinstance(ckpt[k], dict):
            print(f"[load_ark_checkpoint] using checkpoint['{k}']")
            sd = ckpt[k]
            break
    else:
        if key != "auto":
            raise KeyError(f"'{key}' not in checkpoint (keys: {list(ckpt.keys())})")
        print("[load_ark_checkpoint] using the checkpoint dict itself")
        sd = ckpt
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_ark_checkpoint(model, path_or_sd, key="auto", head_map=None):
    """Load an Ark+ (or ImageNet timm-0.5.4 Swin) checkpoint into ArkUnifiedSwin.
    head_map: {checkpoint_head_index: model_head_index}; None = same index. Unmapped heads start random.
    Sets model.new_param_names = parameters that did NOT come from the checkpoint."""
    if isinstance(path_or_sd, dict):
        sd = path_or_sd if "patch_embed.proj.weight" in path_or_sd else pick_state_dict(path_or_sd, key)
    else:
        sd = pick_state_dict(torch_load(path_or_sd), key)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    tgt = model.state_dict()
    pd = model.patch_embed.proj3d.weight.shape[2]
    new, skipped = {}, []
    for k, v in sd.items():
        if "relative_position_index" in k or "attn_mask" in k or k.startswith("head.") or k.startswith("head_dist."):
            continue
        if k.startswith("patch_embed.proj."):
            new[k.replace("patch_embed.proj.", "patch_embed.proj2d.")] = v
            if k.endswith("weight"):   # I3D inflation: RGB summed -> 1 channel, spread over depth
                new["patch_embed.proj3d.weight"] = v.sum(1, keepdim=True).unsqueeze(2).repeat(1, 1, pd, 1, 1) / pd
            else:
                new["patch_embed.proj3d.bias"] = v.clone()
            continue
        if k.startswith("omni_heads."):
            idx, tail = k[len("omni_heads."):].split(".", 1)
            idx = int(idx)
            if head_map is not None:
                if idx not in head_map:
                    continue
                idx = head_map[idx]
            k = f"omni_heads.{idx}.{tail}"
        if k.endswith("relative_position_bias_table") and k in tgt:
            n2d, n3d = v.shape[0], tgt[k].shape[0]
            if n3d % n2d != 0:
                skipped.append(k)
                continue
            v = v.repeat(n3d // n2d, 1)   # every depth-offset plane starts as the 2D table
        if k in tgt and tgt[k].shape == v.shape:
            new[k] = v
        else:
            skipped.append(k)
    msg = model.load_state_dict(new, strict=False)
    loaded = set(new.keys())
    model.new_param_names = {n for n, _ in model.named_parameters() if n not in loaded} | \
                            {"patch_embed.proj3d.weight", "patch_embed.proj3d.bias"}
    print(f"[load_ark_checkpoint] loaded {len(loaded)} tensors")
    print(f"[load_ark_checkpoint] missing (new, randomly initialised): {msg.missing_keys}")
    print(f"[load_ark_checkpoint] skipped from checkpoint (name/shape mismatch): {skipped}")
    return model
