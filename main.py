import os

import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF  # 用于同步增强
from PIL import Image


# ==========================================
# 1. 配置参数 (RTX 3090 定制版)
# ==========================================
class Config:
    # --- 路径配置 (保持你的目录结构) ---
    DATASET_ROOT = './dataset'

    # 训练集路径 (请确保该路径存在，如果只有TestDataset，临时改成TestDataset也能跑)
    TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'TrainDataset', 'Imgs')
    TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, 'TrainDataset', 'GT')

    # 验证集路径
    VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'TestDataset', 'COD10K', 'Imgs')
    VAL_MASK_DIR = os.path.join(DATASET_ROOT, 'TestDataset', 'COD10K', 'GT')

    SAVE_DIR = './checkpoints'

    # --- 3090 性能全开配置 ---
    # 升级 1: 使用更强的 Base 模型
    BACKBONE = 'swin_base_patch4_window12_384'

    # 升级 2: 分辨率提升到 352 (COD 黄金分辨率)
    IMG_SIZE = 384

    # 升级 3: 3090 显存大，可以开大 Batch
    BATCH_SIZE = 16

    EPOCHS = 50
    LEARNING_RATE = 5e-5  # Base 模型建议稍微调低一点 LR
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 8  # CPU 线程数也可以开大

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists(Config.SAVE_DIR):
    os.makedirs(Config.SAVE_DIR)


# ==========================================
# 2. 增强版 Dataset (含同步数据增强)
# ==========================================
class CODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train  # 标记是否为训练模式

        self.img_names = []
        if os.path.exists(img_dir):
            self.img_names = [x for x in os.listdir(img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.img_names.sort()

        if len(self.img_names) == 0:
            print(f"[Warning] 路径下无图片: {img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # --- A. 加载图片 ---
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))

        # --- B. 加载 Mask ---
        file_basename = os.path.splitext(img_name)[0]
        mask_name = file_basename + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)  # 尝试同名

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', img.size, 0)

        # --- C. 同步数据增强 (关键修改) ---
        # 1. 统一调整尺寸
        img = TF.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        mask = TF.resize(mask, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=Image.NEAREST)

        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # 随机垂直翻转
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            # 随机旋转 (-15 ~ 15度)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        # --- D. 转 Tensor & 归一化 ---
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  # 二值化

        return img, mask


# ==========================================
# 3. 模型定义 (修复维度报错版)
# ==========================================
try:
    import timm
except ImportError:
    print("Error: 请安装 timm (pip install timm)")
    exit()


class CenterSurroundModule(nn.Module):
    def __init__(self, in_channels, reduce_factor=4):
        super(CenterSurroundModule, self).__init__()
        mid_channels = in_channels // reduce_factor
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


class BioCSTransNet(nn.Module):
    def __init__(self, num_classes=1, backbone_name=Config.BACKBONE):
        super(BioCSTransNet, self).__init__()
        print(f"Loading Backbone: {backbone_name} ...")

        # 加载 Swin (pretrained=True 会自动下载权重)
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

        # 自动获取 Swin Base 的通道数 [128, 256, 512, 1024]
        dims = self.backbone.feature_info.channels()
        print(f"Feature Channels: {dims}")

        # 颈部 (CS Modules)
        self.cs_block4 = CenterSurroundModule(dims[3])
        self.cs_block3 = CenterSurroundModule(dims[2])

        # 解码器
        self.decoder4 = DecoderBlock(dims[3], dims[2], 512)  # Base模型通道多，解码器也加宽
        self.decoder3 = DecoderBlock(512, dims[1], 256)
        self.decoder2 = DecoderBlock(256, dims[0], 128)

        self.final_conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        features = self.backbone(x)
        c1, c2, c3, c4 = features[0], features[1], features[2], features[3]

        # [关键修复]：解决 Swin Transformer 通道在最后 (NHWC) 导致卷积报错的问题
        # 如果最后一个维度是通道数，则进行 Permute
        if c4.size(-1) == self.backbone.feature_info.channels()[-1]:
            c1 = c1.permute(0, 3, 1, 2).contiguous()
            c2 = c2.permute(0, 3, 1, 2).contiguous()
            c3 = c3.permute(0, 3, 1, 2).contiguous()
            c4 = c4.permute(0, 3, 1, 2).contiguous()

        c4_enhanced = self.cs_block4(c4)
        c3_enhanced = self.cs_block3(c3)

        d4 = self.decoder4(c4_enhanced, c3_enhanced)
        d3 = self.decoder3(d4, c2)
        d2 = self.decoder2(d3, c1)

        logits = self.final_conv(d2)
        out = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        return out


# ==========================================
# 4. 损失函数
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
# 5. 训练引擎
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
        if step % 10 == 0:
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
# 6. 主程序
# ==========================================
if __name__ == '__main__':
    print(f"Device: {Config.device} (RTX 3090 Mode)")
    torch.backends.cudnn.benchmark = True  # 开启加速

    # 1. 准备数据
    try:
        # 训练集: 开启增强 is_train=True
        train_dataset = CODDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, is_train=True)
        # 验证集: 关闭增强 is_train=False
        val_dataset = CODDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=True)

        print(f"Images Loaded: Train({len(train_dataset)}), Val({len(val_dataset)})")
    except Exception as e:
        print(f"Data Load Error: {e}")
        exit()

    if len(train_dataset) == 0:
        print("Error: No training images found. Please check paths in Config.")
        exit()

    # 2. 模型
    model = BioCSTransNet(num_classes=1).to(Config.device)

    # 3. 优化器
    criterion = StructureLoss().to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 4. 循环
    best_iou = 0.0
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.device, epoch)
        val_iou = validate(model, val_loader, Config.device)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(Config.SAVE_DIR, 'best_model_3090.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best Model Saved! (IoU: {best_iou:.4f})")

    print(f"\nFinished. Best IoU: {best_iou:.4f}")