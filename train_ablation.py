import os

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# 从 main.py 导入配置和数据集，确保"控制变量法"，除了模型结构外其他完全一致
try:
    from main import Config, CODDataset, DecoderBlock
except ImportError:
    print("Error: 无法导入 main.py，请确保该文件在当前目录下。")
    exit()


# ==========================================
# 1. 定义 Baseline 模型 (无 CS 模块)
# ==========================================
class BioCSTransNet_Baseline(nn.Module):
    """
    [Ablation Version]
    结构: Swin-Base -> (直连) -> Decoder
    作用: 证明 CS 模块的有效性
    """

    def __init__(self, num_classes=1, backbone_name=Config.BACKBONE):
        super(BioCSTransNet_Baseline, self).__init__()
        print(f"🔥 [Ablation] Loading Baseline Model (WITHOUT CS Module)...")

        # 1. 加载骨干 (保持一致)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            img_size=Config.IMG_SIZE
        )
        dims = self.backbone.feature_info.channels()

        # --- 关键修改点 1: 移除了 CS 模块的初始化 ---
        # self.cs_block4 = CenterSurroundModule(dims[3])
        # self.cs_block3 = CenterSurroundModule(dims[2])

        # 2. 解码器 (保持一致)
        self.decoder4 = DecoderBlock(dims[3], dims[2], 512)
        self.decoder3 = DecoderBlock(512, dims[1], 256)
        self.decoder2 = DecoderBlock(256, dims[0], 128)

        self.final_conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        features = self.backbone(x)
        c1, c2, c3, c4 = features[0], features[1], features[2], features[3]

        # 维度调整 (Swin 的 NHWC -> NCHW) 必须保留
        if c4.size(-1) == self.backbone.feature_info.channels()[-1]:
            c1 = c1.permute(0, 3, 1, 2).contiguous()
            c2 = c2.permute(0, 3, 1, 2).contiguous()
            c3 = c3.permute(0, 3, 1, 2).contiguous()
            c4 = c4.permute(0, 3, 1, 2).contiguous()

        # --- 关键修改点 2: 移除了 CS 模块的处理 ---
        # c4_enhanced = self.cs_block4(c4)
        # c3_enhanced = self.cs_block3(c3)

        # 直接将骨干特征传给解码器
        # 这里的特征是"原汁原味"的 Transformer 特征，没有经过仿生增强
        d4 = self.decoder4(c4, c3)
        d3 = self.decoder3(d4, c2)
        d2 = self.decoder2(d3, c1)

        logits = self.final_conv(d2)
        out = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        return out


# ==========================================
# 2. 训练辅助函数
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0
    print(f"\nEpoch [{epoch + 1}/{Config.EPOCHS}] Training (Baseline)...")
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if step % 50 == 0:
            print(f"  Step [{step}/{len(loader)}] Loss: {loss.item():.4f}")
    return epoch_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            inter = (pred * masks).sum()
            union = pred.sum() + masks.sum() - inter
            iou = (inter + 1e-6) / (union + 1e-6)
            total_iou += iou.item()
    return total_iou / len(loader)


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    print(f"Starting Ablation Study on {Config.device}...")

    # 1. 设置保存路径 (与主实验区分开)
    ABLATION_SAVE_DIR = './checkpoints_ablation'
    if not os.path.exists(ABLATION_SAVE_DIR):
        os.makedirs(ABLATION_SAVE_DIR)

    # 2. 数据加载 (复用 Config)
    train_dataset = CODDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, is_train=True)
    val_dataset = CODDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # 3. 初始化 Baseline 模型
    model = BioCSTransNet_Baseline(num_classes=1).to(Config.device)

    # 4. 优化器与 Loss (保持与主实验完全一致)
    pos_weight = torch.tensor([10.0]).to(Config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. 开始训练
    best_iou = 0.0
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.device, epoch)
        val_iou = validate(model, val_loader, Config.device)
        scheduler.step()

        print(f"Baseline Epoch [{epoch + 1}/{Config.EPOCHS}] Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(ABLATION_SAVE_DIR, 'best_model_baseline.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best Baseline Saved! (IoU: {best_iou:.4f})")

    print("\n✅ Ablation Study Finished.")