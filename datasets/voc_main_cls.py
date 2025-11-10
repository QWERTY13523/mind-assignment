import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
NUM_CLS = 20

def _read_id_list(main_dir: str, split: str) -> List[str]:
    fp = os.path.join(main_dir, f"{split}.txt")
    with open(fp, "r") as f:
        return [x.strip() for x in f if x.strip()]

def _read_class_matrix(main_dir: str, split: str, ids: List[str]) -> np.ndarray:
    """
    读取每个 <cls>_<split>.txt，行格式：<img_id> <label>，label∈{-1,0,1}
    我们将 1->1，其他(0,-1)->0，得到 [N,20] 的多标签矩阵。
    """
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    Y = np.zeros((len(ids), NUM_CLS), dtype=np.float32)
    for c, cls in enumerate(CLASS_NAMES):
        fp = os.path.join(main_dir, f"{cls}_{split}.txt")
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                sid, val = line.split()
                if sid not in id_to_idx: continue
                lab = int(val)
                if lab == 1:
                    Y[id_to_idx[sid], c] = 1.0
                # lab==0(ambiguous) 或 -1(negative) -> 0
    return Y

class VOCDatasetMainCLS(Dataset):
    """
    仅用于分类（不要求分割标注）。
    root: /path/to/VOC2007 或 /path/to/VOC2012
    split: 'train', 'val', 'trainval', 'test'
    """
    def __init__(self, root: str, split: str = "trainval",
                 img_size: int = 512, random_flip: bool = True):
        super().__init__()
        self.root = root
        self.img_dir = os.path.join(root, "JPEGImages")
        self.main_dir = os.path.join(root, "ImageSets", "Main")
        self.ids = _read_id_list(self.main_dir, split)

        # 过滤缺失文件
        self.ids = [i for i in self.ids if os.path.exists(os.path.join(self.img_dir, f"{i}.jpg"))]

        self.labels = _read_class_matrix(self.main_dir, split, self.ids)  # [N,20]
        self.img_size = img_size
        self.random_flip = random_flip
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self): return len(self.ids)

    def _resize(self, img, size):
        return F.resize(img, [size, size], Image.BILINEAR)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{_id}.jpg")).convert("RGB")
        img = self._resize(img, self.img_size)
        if self.random_flip:
            import numpy as np
            if np.random.rand() < 0.5:
                img = F.hflip(img)
        x = self.norm(self.to_tensor(img))
        y = torch.from_numpy(self.labels[idx].copy())  # [20]
        return {"id": _id, "image": x, "multilabel": y}
