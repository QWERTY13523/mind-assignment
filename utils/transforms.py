import os, random
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

_gauss = T.GaussianBlur(kernel_size=11, sigma=(3.0, 3.0))

def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3: return _gauss(x)
    return torch.stack([_gauss(xx) for xx in x], dim=0)

def sample_random_bg_like(img: torch.Tensor, bg_dir: str=None) -> torch.Tensor:
    device = img.device
    if img.dim()==3: B, H, W = 1, img.shape[-2], img.shape[-1]
    else: B, H, W = img.size(0), img.shape[-2], img.shape[-1]
    if bg_dir and os.path.isdir(bg_dir):
        files = [f for f in os.listdir(bg_dir) if f.lower().endswith((".jpg",".png",".jpeg",".bmp"))]
        if files:
            fp = os.path.join(bg_dir, random.choice(files))
            im = Image.open(fp).convert("RGB").resize((W,H), Image.BILINEAR)
            arr = (np.asarray(im).astype(np.float32)/255.0 - [0.485,0.456,0.406])/[0.229,0.224,0.225]
            t = torch.from_numpy(arr).permute(2,0,1)
            return t.to(device) if B==1 else t.unsqueeze(0).repeat(B,1,1,1).to(device)
    # 退化：均值 + 噪声
    noise = torch.randn((B,3,H,W), device=device) * 0.05
    return noise if img.dim()==4 else noise.squeeze(0)

def ring_mask(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    """
    近似 DoG 的“外周环”先验：用形态学梯度 maxpool-minpool 得到边界带。
    mask: [B,1,H,W] in {0,1}
    """
    pad = k // 2
    maxed = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    mined = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    ring = (maxed - mined) - mask  # 去掉中心，保留环
    return ring.clamp(0, 1)

def apply_mask_only(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return img * mask

def apply_mask_blur(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return img * mask + gaussian_blur(img) * (1.0 - mask)

def apply_mask_randbg(img: torch.Tensor, mask: torch.Tensor, bg_dir: str=None) -> torch.Tensor:
    bg = sample_random_bg_like(img, bg_dir)
    return img * mask + bg * (1.0 - mask)
