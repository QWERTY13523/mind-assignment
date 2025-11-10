import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
