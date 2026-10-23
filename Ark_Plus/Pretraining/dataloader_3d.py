"""
dataloader_3d.py  (place in Ark_Plus/Pretraining/)
---------------------------------------------------
CTVolume: 3D chest-CT dataset with the same constructor and return convention as Ark+'s 2D
datasets, registered into dataloader.dict_dataloarder. Select it in datasets_config.yaml with
`loader: CTVolume` under any dataset name.

List-file format (like ChestXray14):   <relative_path> <l1> <l2> ... <lK>

Volumes (relative to data_dir), in (z,y,x) RAS convention (see ct_utils.py):
  * uint8 .npy  = pre-windowed HU [-1000, 400] (what prep_ctrate.py writes)   <- recommended
  * int16/float .npy or .nii.gz = HU (windowed here with hu_window)
DRRs: with drr_dir set, loaded from <data_dir>/<drr_dir>/<volume stem>.png (precomputed from full HU);
otherwise computed on the fly (HU volumes only).

Returns:  train (augment is None):  student_vol, teacher_vol, label[, drr]     vol: (1, D, H, W)
          val / test:               vol, vol, label
Teacher view = resize only; student view = random crop + intensity jitter (Ark's 2D scheme).
"""
import copy
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataloader import dict_dataloarder, build_ts_transformations
from ct_utils import load_volume, make_drr

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GREY_MEAN, GREY_STD = 0.449, 0.226   # grey-level ImageNet stats; matches the RGB-summed 3D stem


class CTVolume(Dataset):
    def __init__(self, images_path, file_path, crop_size=None, resize=None, augment=None,
                 volume_size=(128, 224, 224), hu_window=(-1000, 400), return_drr=True,
                 drr_size=224, drr_dir=None, annotation_percent=100, **unused):
        self.train = augment is None          # main_ark.py passes augment=None only for the train split
        self.images_path = images_path
        self.volume_size = tuple(int(v) for v in volume_size)
        self.hu_window = tuple(float(v) for v in hu_window)
        self.return_drr = bool(return_drr) and self.train
        self.drr_size = int(drr_size)
        self.drr_dir = os.path.join(images_path, drr_dir) if drr_dir else None
        self.drr_augment = build_ts_transformations(self.drr_size)   # same student aug as the 2D datasets

        self.img_list, self.img_label = [], []
        with open(file_path, "r") as f:
            for line in f:
                items = line.strip().split()
                if not items:
                    continue
                self.img_list.append(os.path.join(images_path, items[0]))
                self.img_label.append([float(v) for v in items[1:]])

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            indexes = indexes[:int(indexes.shape[0] * annotation_percent / 100.0)]
            _l, _y = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = [_l[i] for i in indexes]
            self.img_label = [_y[i] for i in indexes]

    def __len__(self):
        return len(self.img_list)

    def _resize(self, v):
        return F.interpolate(v, size=self.volume_size, mode="trilinear", align_corners=False)

    @staticmethod
    def _random_crop(v, scale=(0.8, 1.0)):
        _, _, D, H, W = v.shape
        sl = []
        for n in (D, H, W):
            m = max(1, int(round(n * random.uniform(*scale))))
            s = random.randint(0, n - m)
            sl.append(slice(s, s + m))
        return v[:, :, sl[0], sl[1], sl[2]]

    @staticmethod
    def _intensity(x):   # x in [0,1]; mirrors Ark's OneOf(brightness/contrast, gamma), p=0.3
        if random.random() < 0.3:
            if random.random() < 0.5:
                x = x * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
            else:
                x = x.clamp(0, 1) ** random.uniform(0.8, 1.25)
        return x.clamp(0, 1)

    def _drr(self, path, vol, prewindowed):
        if self.drr_dir is not None:
            stem = os.path.basename(path)
            stem = stem[:-4] if stem.endswith(".npy") else stem.replace(".nii.gz", "")
            img = cv2.imread(os.path.join(self.drr_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"DRR not found for {path} in {self.drr_dir}")
            img = cv2.resize(img, (self.drr_size, self.drr_size), interpolation=cv2.INTER_AREA)
        elif not prewindowed:
            img = make_drr(vol.astype(np.float32), self.drr_size)
        else:
            raise ValueError("uint8 (pre-windowed) volumes need precomputed DRRs: set drr_dir in dataset_kwargs")
        img = np.repeat(img[..., None], 3, axis=2)
        img = self.drr_augment(image=img)["image"].astype(np.float32) / 255.0
        return torch.from_numpy(((img - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).astype(np.float32))

    def __getitem__(self, index):
        torch.set_num_threads(1)
        cv2.setNumThreads(0)
        path = self.img_list[index]
        vol = load_volume(path)
        prewindowed = vol.dtype == np.uint8
        v = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None]
        if prewindowed:
            to01 = lambda t: t / 255.0
        else:
            lo, hi = self.hu_window
            to01 = lambda t: (t.clamp(lo, hi) - lo) / (hi - lo)
        label = torch.FloatTensor(self.img_label[index])

        teacher = ((to01(self._resize(v)) - GREY_MEAN) / GREY_STD)[0]
        if not self.train:
            return teacher, teacher, label
        student = ((self._intensity(to01(self._resize(self._random_crop(v)))) - GREY_MEAN) / GREY_STD)[0]
        if not self.return_drr:
            return student, teacher, label
        return student, teacher, label, self._drr(path, vol, prewindowed)


dict_dataloarder["CTVolume"] = CTVolume
