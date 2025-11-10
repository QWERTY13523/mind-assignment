#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepLabV3-ResNet50 二分类（前景/背景）语义分割训练脚本（方案 B）
- 先加载分割预训练权重（21 类），再将分类头替换为 2 类
- 训练集 = VOC2007(trainval) ∪ VOC2012(trainval)
- 验证集 = VOC2012(val)
- 计算并打印 val_mIoU 与 per-class IoU
- 仅将最后 1x1 卷积层改为 2 通道；其余权重保持预训练
- 支持 ignore_index=255，二分类时把所有非 0 类并为前景=1（背景=0）

用法示例：
    python train_seg.py --data-root /path/to/VOCdevkit --epochs 50 --batch-size 8 --lr 0.01

目录要求：
    data-root/
        VOC2007/
            JPEGImages, SegmentationClass, ImageSets/Segmentation/{train.txt,val.txt}
        VOC2012/
            JPEGImages, SegmentationClass, ImageSets/Segmentation/{train.txt,val.txt}
"""

import os
import math
import time
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms.functional as F
from PIL import Image


# -----------------------------
#  公共工具
# -----------------------------
def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_fg_bg(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    将 VOC 多类标签映射为前景/背景二类：
      - 背景(0) 保留为 0
      - 所有 >0 的类别并为 1
      - ignore(255) 保持 255
    输入 mask(tensor, HxW) 的 dtype 为 long/int
    """
    out = mask.clone()
    ignore = (out == ignore_index)
    out = torch.where(out > 0, torch.tensor(1, dtype=out.dtype), out)
    out[ignore] = ignore_index
    return out


# -----------------------------
#  变换：保证图像与掩码同步
# -----------------------------
class SegTrainTransform:
    """训练增广：随机缩放、随机裁剪、随机翻转、归一化"""

    def __init__(self, crop_size: int = 512, min_scale: float = 0.5, max_scale: float = 2.0):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        # ImageNet 归一化
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # 随机缩放（等比例）
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

        # 随机水平翻转
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # 随机裁剪（必要时先填充）
        pad_h = max(self.crop_size - new_h, 0)
        pad_w = max(self.crop_size - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            # 图像用 0 填充，掩码用 255(ignore) 填充
            img = F.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = F.pad(mask, [0, 0, pad_w, pad_h], fill=255)
            new_w, new_h = img.size

        # 在有效范围内随机裁剪
        top = random.randint(0, new_h - self.crop_size)
        left = random.randint(0, new_w - self.crop_size)
        img = F.crop(img, top, left, self.crop_size, self.crop_size)
        mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        # 转张量与归一化 / 标签转 long
        img = F.to_tensor(img)
        img = F.normalize(img, mean=self.mean, std=self.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = to_fg_bg(mask, ignore_index=255)

        return img, mask


class SegValTransform:
    """验证变换：等比例缩放长边到指定最大尺寸，保持分辨率，归一化"""

    def __init__(self, long_side_max: int = 768):
        self.long_side_max = long_side_max
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        w, h = img.size
        long_side = max(h, w)
        if long_side > self.long_side_max:
            scale = self.long_side_max / long_side
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
            mask = F.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

        img = F.to_tensor(img)
        img = F.normalize(img, mean=self.mean, std=self.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = to_fg_bg(mask, ignore_index=255)

        return img, mask


# -----------------------------
#  包装数据集以应用成对变换
# -----------------------------
class VOCPairDataset(Dataset):
    def __init__(self, voc: VOCSegmentation, transform):
        self.voc = voc
        self.transform = transform

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]  # PIL.Image
        img, target = self.transform(img, target)
        return img, target


def build_datasets(data_root: str):
    """
    训练集：VOC2007(trainval) ∪ VOC2012(trainval)
    验证集：VOC2012(val)
    """
    # 训练
    tf_train = SegTrainTransform(crop_size=512, min_scale=0.5, max_scale=2.0)
    ds07_tr = VOCPairDataset(VOCSegmentation(root=data_root, year="2007", image_set="train"), tf_train)
    ds07_va = VOCPairDataset(VOCSegmentation(root=data_root, year="2007", image_set="val"), tf_train)
    ds12_tr = VOCPairDataset(VOCSegmentation(root=data_root, year="2012", image_set="train"), tf_train)
    ds12_va = VOCPairDataset(VOCSegmentation(root=data_root, year="2012", image_set="val"), tf_train)
    train_set = ConcatDataset([ds07_tr, ds07_va, ds12_tr, ds12_va])

    # 验证
    tf_val = SegValTransform(long_side_max=768)
    val_set = VOCPairDataset(VOCSegmentation(root=data_root, year="2012", image_set="val"), tf_val)

    return train_set, val_set


# -----------------------------
#  模型：方案 B
# -----------------------------
def build_deeplabv3_2c(aux_loss: bool = True) -> nn.Module:
    """
    先加载 COCO(VOC mapping) 的分割预训练权重（21 类），再替换头为 2 类。
    """
    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=aux_loss
    )

    # 替换主头
    in_ch = model.classifier[-1].in_channels  # 256
    model.classifier[-1] = nn.Conv2d(in_ch, 2, kernel_size=1)

    # 替换辅助头
    if aux_loss and hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 2, kernel_size=1)

    # 初始化新加层
    heads = [model.classifier[-1]]
    if aux_loss and model.aux_classifier is not None:
        heads.append(model.aux_classifier[-1])
    for m in heads:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    return model


# -----------------------------
#  优化器与调度
# -----------------------------
def build_optimizer(model: nn.Module, base_lr: float = 0.01, weight_decay: float = 1e-4):
    # 新头学习率大，骨干小（常见设置）
    params = [
        {"params": model.backbone.parameters(), "lr": base_lr * 0.1},
        {"params": model.classifier.parameters(), "lr": base_lr},
    ]
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        params.append({"params": model.aux_classifier.parameters(), "lr": base_lr})
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return optimizer


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly 学习率计划：lr = lr_init * (1 - iter/max_iter) ^ power
    通常用于语义分割
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


# -----------------------------
#  指标：mIoU
# -----------------------------
@torch.no_grad()
def miou_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes=2, ignore_index=255):
    """
    单批次计算混淆矩阵并返回 (miou, per_class_iou, confmat)
    logits: [B, C, H, W]
    target: [B, H, W]
    """
    pred = logits.argmax(dim=1)  # [B,H,W]

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    if target.numel() == 0:
        iou = torch.zeros(num_classes, device=logits.device)
        return torch.tensor(0.0, device=logits.device), iou, torch.zeros(num_classes, num_classes, device=logits.device)

    k = (target >= 0) & (target < num_classes)
    pred = pred[k]
    target = target[k]

    conf = torch.bincount(
        target * num_classes + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = torch.zeros(num_classes, device=logits.device)
    iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)
    return miou, iou, conf


# -----------------------------
#  训练 / 验证
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, scheduler=None, aux_weight=0.4):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    running_loss = 0.0
    num_iters = len(loader)

    start = time.time()
    for it, (images, targets) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(images)
            logits = out["out"]
            loss = criterion(logits, targets)
            if "aux" in out and out["aux"] is not None:
                loss = loss + aux_weight * criterion(out["aux"], targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        if it % 20 == 0 or it == num_iters:
            lr_head = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
            print(f"Epoch[{epoch}] Iter[{it}/{num_iters}]  loss={loss.item():.4f}  lr_head={lr_head:.6f}")

    cost = time.time() - start
    return running_loss / num_iters, cost


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_conf = torch.zeros(2, 2, device=device)
    total_iou = torch.zeros(2, device=device)
    count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        out = model(images)
        logits = out["out"]
        miou_b, iou_b, conf_b = miou_from_logits(logits, targets, num_classes=2, ignore_index=255)
        total_conf += conf_b
        total_iou += iou_b
        count += 1

    # 用总体混淆矩阵计算一次全局 IoU（更稳健）
    tp = total_conf.diag()
    fp = total_conf.sum(0) - tp
    fn = total_conf.sum(1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = torch.zeros(2, device=device)
    iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=device)

    return miou.item(), iou.detach().cpu().tolist()


# -----------------------------
#  主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepLabV3-ResNet50 二分类（前景/背景）训练（方案B）")
    parser.add_argument("--data-root", type=str, required=True, help="VOCdevkit 根目录（包含 VOC2007 与/或 VOC2012）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="使用混合精度")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    print(">> 构建数据集...")
    train_set, val_set = build_datasets(args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"训练样本数: {len(train_set)} | 验证样本数: {len(val_set)}")

    print(">> 构建模型（方案 B）...")
    model = build_deeplabv3_2c(aux_loss=True).to(device)
    optimizer = build_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)

    max_iters = args.epochs * len(train_loader)
    scheduler = PolyLR(optimizer, max_iters=max_iters, power=0.9)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    best_miou = 0.0
    best_path = os.path.join(args.output, "best_miou.pth")

    print(">> 开始训练")
    global_iter = 0
    for epoch in range(1, args.epochs + 1):
        avg_loss, t_cost = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, scheduler)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  time={t_cost:.1f}s")

        val_miou, per_class_iou = validate(model, val_loader, device)
        print(f"[Epoch {epoch}] val_mIoU={val_miou:.4f}  IoU_bg={per_class_iou[0]:.4f}  IoU_fg={per_class_iou[1]:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_miou,
                "args": vars(args)
            }, best_path)
            print(f">> 已保存最佳模型至: {best_path} (mIoU={best_miou:.4f})")

    print(f"训练结束。最佳验证 mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
