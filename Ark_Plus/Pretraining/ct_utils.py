"""
ct_utils.py  (place in Ark_Plus/Pretraining/)
---------------------------------------------
numpy/cv2-only CT helpers shared by dataloader_3d.py and prep_ctrate.py (no torch import, so the
preprocessing can run in any env with numpy, scipy, cv2, nibabel, pandas).

Internal volume convention everywhere: (z, y, x) = (inferior->superior, posterior->anterior,
patient-left->patient-right), i.e. nibabel RAS-canonical transposed.
"""
import numpy as np
import cv2
from scipy.ndimage import zoom, gaussian_filter

HU_WINDOW = (-1000.0, 400.0)   # window used for the stored uint8 volumes and the 3D network input


def load_volume(path):
    """.npy -> stored dtype (uint8 = pre-windowed, else HU). NIfTI -> float32 HU in (z,y,x) via the header."""
    if path.endswith(".npy"):
        return np.load(path)
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0)


def resample(vol, spacing, target):
    """Linear resampling of a (z,y,x) volume from `spacing` to `target` mm, Gaussian prefilter on
    downsampled axes to limit aliasing."""
    factors = [s / t for s, t in zip(spacing, target)]
    sigma = [max(0.0, (1.0 / f - 1.0) / 2.0) for f in factors]
    if any(s > 0 for s in sigma):
        vol = gaussian_filter(vol, sigma)
    return zoom(vol, factors, order=1)


def center_fit(vol, shape, fill=-1024.0):
    """Center crop / pad each axis to `shape`."""
    out = np.full(shape, fill, dtype=np.float32)
    src, dst = [], []
    for n, m in zip(vol.shape, shape):
        if n >= m:
            s = (n - m) // 2
            src.append(slice(s, s + m))
            dst.append(slice(0, m))
        else:
            s = (m - n) // 2
            src.append(slice(0, n))
            dst.append(slice(s, s + n))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def window_uint8(hu, window=HU_WINDOW):
    lo, hi = window
    return np.round(np.clip((hu - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def make_drr(hu, size=512, mu_water=0.02, spacing_zx=(1.0, 1.0)):
    """Parallel-beam AP DRR displayed like a CXR: head up, patient's right on image left, bone bright.
    hu: (z,y,x) HU. spacing_zx: (z, x) voxel size in mm, used to keep the physical aspect ratio.
    Returns uint8 (size, size)."""
    mu = np.clip(mu_water * (1.0 + np.maximum(hu, -1000.0) / 1000.0), 0, None)
    line = mu.sum(axis=1).astype(np.float32)               # integrate along y (AP) -> (z, x)
    lo, hi = np.percentile(line, (0.5, 99.5))
    img = np.clip((line - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    img = np.ascontiguousarray(img[::-1, ::-1])             # superior at top; patient right -> image left
    sz, sx = spacing_zx
    h, w = img.shape
    if abs(sz - sx) > 1e-6:
        img = cv2.resize(img, (w, max(1, int(round(h * sz / sx)))), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape
    s = max(h, w)
    canvas = np.zeros((s, s), np.float32)                   # pad to square without distorting
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return (canvas * 255).astype(np.uint8)


def preview_panel(hu, drr, spacing_zyx):
    """Axial (radiological: anterior up, patient right on image left), coronal (head up) and DRR,
    side by side, 256 px tall each, for checking orientation by eye."""
    sz, sy, sx = spacing_zyx
    lo, hi = HU_WINDOW
    to8 = lambda a: (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    ax = to8(hu[hu.shape[0] // 2])[::-1, ::-1]                       # (y,x)
    co = to8(hu[:, hu.shape[1] // 2, :])[::-1, ::-1]                  # (z,x)
    ax = cv2.resize(np.ascontiguousarray(ax), (256, int(round(256 * ax.shape[0] * sy / (ax.shape[1] * sx)))))
    co = cv2.resize(np.ascontiguousarray(co), (256, int(round(256 * co.shape[0] * sz / (co.shape[1] * sx)))))
    dr = cv2.resize(drr, (256, 256))
    tiles = []
    for im, name in ((ax, "axial"), (co, "coronal"), (dr, "DRR")):
        t = np.zeros((280, 256), np.uint8)
        t[24:24 + min(256, im.shape[0]), :] = im[:256]
        cv2.putText(t, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
        tiles.append(t)
    return np.hstack(tiles)
