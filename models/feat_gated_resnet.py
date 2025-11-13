import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

try:
    # torchvision>=0.12 支持对 Tensor 做高斯模糊
    from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

    _HAS_TV_GBLUR = True
except Exception:
    _HAS_TV_GBLUR = False


def _gaussian_blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """对 4D Tensor 进行高斯模糊（优先用 torchvision 的 gaussian_blur；否则用均值池化近似）"""
    k = int(2 * round(3 * sigma) + 1)  # ~ 6*sigma 的覆盖，取奇数
    k = max(3, k | 1)  # 至少 3，确保为奇数
    if _HAS_TV_GBLUR:
        return tv_gaussian_blur(x, [k, k], [sigma, sigma])
    # 兜底：用 avg_pool2d 近似（不是严格高斯，但够用来产生环）
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


@torch.no_grad()
def ring_mask(mask: torch.Tensor, beta: float = 0.5, sigma_base: float = 1.0) -> torch.Tensor:
    """
    生成“环形”边缘响应：DoG(mask) = G(sigma2) - G(sigma1)，并 ReLU+归一化到[0,1]
    mask: (B,1,H,W) 或 (B,C,H,W) 的浮点张量，值域建议在[0,1]
    beta: 环的相对宽度，越大环越宽（sigma2 = sigma1*(1+2*beta)）
    """
    if mask.dim() != 4:
        raise ValueError(f"ring_mask expects 4D tensor, got {mask.shape}")
    if mask.size(1) != 1:
        mask = mask.mean(1, keepdim=True)  # 多通道时取均值
    m = mask.detach()  # 不回传梯度到分割掩膜
    sigma1 = sigma_base
    sigma2 = sigma_base * (1.0 + 2.0 * float(beta))
    g1 = _gaussian_blur_tensor(m, sigma1)
    g2 = _gaussian_blur_tensor(m, sigma2)
    r = (g2 - g1).clamp_min_(0)  # 只保留外围正环
    # 归一化到 [0,1]，避免数值尺度影响 gating
    denom = r.amax(dim=(-2, -1), keepdim=True).clamp_min_(1e-6)
    r = r / denom
    return r

class FeatGatedResNet(nn.Module):
    """
    ResNet-50 分类器 + 掩膜 gating（在 layer3 后对特征进行 1+alpha*mask 的逐像素放大）
    """
    def __init__(self, num_classes=20, gate_alpha: float=0.8, ring_beta=0.5, pretrained=True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool, self.fc = base.avgpool, nn.Linear(2048, num_classes)
        self.gate_alpha = gate_alpha
        self.ring_beta = ring_beta

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        if mask is not None and (self.gate_alpha > 0.0 or self.ring_beta > 0.0):
            m = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
            r = ring_mask(m)  # 近似 DoG 外周
            # 中心促通 + 外周边界促通/抑制（按需要，正号为促通，负号为抑制）
            x = x * (1.0 + self.gate_alpha * m + self.ring_beta * r)
        x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)
