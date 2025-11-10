import os
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

# VOC 有 20 类（不含 background）
NUM_CLS = 20

def _load_ids(fp: str) -> List[str]:
    with open(fp, "r") as f:
        return [x.strip() for x in f if x.strip()]

class VOCDatasetFGBG(Dataset):
    """
    使用 VOC2012 分割标注：
      - 前景/背景二值掩膜（>0 且 !=255 视作前景）
      - 多标签分类目标：图中出现过的所有非背景类置 1
    """
    def __init__(self, voc2012_root: str, split: str="train",
                 img_size: int=512, for_segmentation: bool=True, random_flip: bool=True):
        super().__init__()
        self.root = voc2012_root
        self.img_dir = os.path.join(self.root, "JPEGImages")
        self.seg_dir = os.path.join(self.root, "SegmentationClass")
        split_fp = os.path.join(self.root, "ImageSets", "Segmentation", f"{split}.txt")
        self.ids = _load_ids(split_fp)
        # 只保留既有图像也有掩膜的样本
        self.ids = [i for i in self.ids
                    if os.path.exists(os.path.join(self.img_dir, f"{i}.jpg"))
                    and os.path.exists(os.path.join(self.seg_dir, f"{i}.png"))]

        self.img_size = img_size
        self.for_segmentation = for_segmentation
        self.random_flip = random_flip
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self): return len(self.ids)

    def _resize(self, img, mask, size):
        return (F.resize(img, [size, size], Image.BILINEAR),
                F.resize(mask,[size, size], Image.NEAREST))

    def _hflip(self, img, mask):
        import numpy as np
        if np.random.rand() < 0.5:
            img = F.hflip(img); mask = F.hflip(mask)
        return img, mask

    def _build_multilabel(self, mask_np: np.ndarray) -> torch.Tensor:
        # mask: HxW, {0..20,255}；非 0/255 的值表示出现过对应类
        labels = np.unique(mask_np)
        labels = labels[(labels != 0) & (labels != 255)]
        y = np.zeros((NUM_CLS,), dtype=np.float32)
        for lb in labels:
            if 1 <= lb <= 20: y[lb-1] = 1.0
        return torch.from_numpy(y)  # (20,)

    def __getitem__(self, idx: int):
        _id = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.seg_dir, f"{_id}.png"))
        img, mask = self._resize(img, mask, self.img_size)
        if self.random_flip: img, mask = self._hflip(img, mask)

        img_t = self.norm(self.to_tensor(img))          # [3,H,W]
        mask_np = np.array(mask, dtype=np.uint8)        # [H,W]
        fg = ((mask_np != 0) & (mask_np != 255)).astype(np.uint8)
        fg_t = torch.from_numpy(fg).float().unsqueeze(0)  # [1,H,W]
        y = self._build_multilabel(mask_np)               # [20]

        sample = {"id": _id, "image": img_t, "fgmask": fg_t, "multilabel": y}
        if self.for_segmentation:
            sample["semseg"] = torch.from_numpy(mask_np.copy()).long()
        return sample
