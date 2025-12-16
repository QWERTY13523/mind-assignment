import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

# 检查是否安装了 timm
try:
    import timm
except ImportError:
    print("Error: 请安装 timm (pip install timm)")
    exit()


# ==========================================
# 1. 配置参数 (Ablation: CNN Version)
# ==========================================
class Config:
    # --- 路径配置 ---
    DATASET_ROOT = './dataset'
    TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'TrainDataset', 'Imgs')
    TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, 'TrainDataset', 'GT')
    VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'TestDataset', 'COD10K', 'Imgs')
    VAL_MASK_DIR = os.path.join(DATASET_ROOT, 'TestDataset', 'COD10K', 'GT')

    # 结果保存路径 (建议区分文件夹，以免覆盖Swin的权重)
    SAVE_DIR = './checkpoints_ablation_cnn'

    # --- 消融实验核心修改 ---
    # 使用 ResNet50 作为对比基线 (CNN代表)
    BACKBONE = 'resnet50'

    IMG_SIZE = 384
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4  # CNN 通常比 Transformer 更稳定，LR 可以稍微大一点点，或者保持 5e-5 也可以
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists(Config.SAVE_DIR):
    os.makedirs(Config.SAVE_DIR)


# ==========================================
# 2. Dataset (保持不变)
# ==========================================
class CODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.img_names = []
        if os.path.exists(img_dir):
            self.img_names = [x for x in os.listdir(img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))

        file_basename = os.path.splitext(img_name)[0]
        mask_name = file_basename + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.mask_dir, img_name)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', img.size, 0)

        # 同步增强
        img = TF.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        mask = TF.resize(mask, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return img, mask


# ==========================================
# 3. 模块定义 (CSM 和 Decoder 保持不变)
# ==========================================
class CenterSurroundModule(nn.Module):
    def __init__(self, in_channels, reduce_factor=4):
        super(CenterSurroundModule, self).__init__()
        # 针对 ResNet50，in_channels 可能会很大 (例如 2048)，确保 reduce 后通道数合理
        mid_channels = max(in_channels // reduce_factor, 32)
        self.reduce_conv = nn.Conv2d(in_channels, mid_channels, 1)
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)
        )
        self.out_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x_reduced = self.reduce_conv(x)
        surround = F.avg_pool2d(x_reduced, kernel_size=3, stride=1, padding=1)
        diff = torch.abs(x_reduced - surround)
        att_map = torch.sigmoid(self.contrast_conv(diff))
        out = x * att_map + x
        return self.out_conv(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            if x_up.size() != skip.size():
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([x_up, skip], dim=1)
        else:
            out = x_up
        return self.conv(out)


# ==========================================
# 4. 模型定义 (Ablation: CNN Version)
# ==========================================
class BioCS_ResNet(nn.Module):
    def __init__(self, num_classes=1, backbone_name=Config.BACKBONE):
        super(BioCS_ResNet, self).__init__()
        print(f"Loading Ablation Backbone: {backbone_name} (CNN) ...")

        # 加载 ResNet50
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

        # [关键修正 1]: 获取通道数
        # ResNet 通常返回 5 层: [64(s2), 256(s4), 512(s8), 1024(s16), 2048(s32)]
        # 我们只需要后 4 层来对齐 Swin 的结构
        all_dims = self.backbone.feature_info.channels()
        dims = all_dims[1:]  # 取 [256, 512, 1024, 2048]

        print(f"Selected Feature Channels: {dims}")

        # 颈部 (CS Modules)
        self.cs_block4 = CenterSurroundModule(dims[3])  # 2048
        self.cs_block3 = CenterSurroundModule(dims[2])  # 1024

        # 解码器
        self.decoder4 = DecoderBlock(dims[3], dims[2], 512)
        self.decoder3 = DecoderBlock(512, dims[1], 256)
        self.decoder2 = DecoderBlock(256, dims[0], 128)  # dims[0] 对应 ResNet 的 layer1 (256通道)

        self.final_conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        features = self.backbone(x)

        # [关键修正 2]: 跳过 features[0] (Stride 2)
        # 取 features[1] (Stride 4) 到 features[4] (Stride 32)
        c1 = features[1]
        c2 = features[2]
        c3 = features[3]
        c4 = features[4]

        # 颈部处理
        c4_enhanced = self.cs_block4(c4)
        c3_enhanced = self.cs_block3(c3)

        # 解码
        d4 = self.decoder4(c4_enhanced, c3_enhanced)
        d3 = self.decoder3(d4, c2)
        d2 = self.decoder2(d3, c1)  # c1 现在是 Stride 4 (96x96)

        logits = self.final_conv(d2)

        # 96x96 * 4 = 384x384 (正确)
        out = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        return out


# ==========================================
# 5. 损失函数 (保持不变)
# ==========================================
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        wbce = F.binary_cross_entropy_with_logits(pred, mask)
        pred_sigmoid = torch.sigmoid(pred)
        inter = (pred_sigmoid * mask).sum(dim=(2, 3))
        union = (pred_sigmoid + mask).sum(dim=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return wbce + iou.mean()


# ==========================================
# 6. 训练与验证逻辑 (保持不变)
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0
    print(f"\nEpoch [{epoch + 1}/{Config.EPOCHS}] Training...")
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if step % 20 == 0:
            print(f"  Step [{step}/{len(loader)}] Loss: {loss.item():.4f}")
    return epoch_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_iou = 0
    print("Validating...")
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
# 7. 主程序
# ==========================================
if __name__ == '__main__':
    print(f"Device: {Config.device} (Ablation Study: CNN)")
    torch.backends.cudnn.benchmark = True

    # 1. 数据
    try:
        train_dataset = CODDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, is_train=True)
        val_dataset = CODDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    if len(train_dataset) == 0:
        print("无数据，请检查路径。")
        exit()

    # 2. 模型 (实例化 ResNet 版本)
    model = BioCS_ResNet(num_classes=1).to(Config.device)

    # 3. 优化器
    criterion = StructureLoss().to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 4. 训练
    best_iou = 0.0
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.device, epoch)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] Train Loss: {train_loss:.4f}")

        # 保存为不同的名字
        save_path = os.path.join(Config.SAVE_DIR, 'best_model_resnet_ablation.pth')
        torch.save(model.state_dict(), save_path)
        print(f"  >>> Best CNN Model Saved! (IoU: {best_iou:.4f})")

    print(f"\nAblation Finished. Best CNN IoU: {best_iou:.4f}")