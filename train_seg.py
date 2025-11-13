#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# 兼容不同 torchvision 版本的权重枚举导入
try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except Exception:
    from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights


# ----------------------------
#  工具函数
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def fast_confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int = 255,
) -> torch.Tensor:
    """
    pred:   (B,H,W) int64
    target: (B,H,W) int64
    return: (C,C) confusion matrix
    """
    assert pred.shape == target.shape
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    k = (target >= 0) & (target < num_classes)
    pred = pred[k]
    target = target[k]
    conf = torch.bincount(
        target * num_classes + pred, minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return conf


def miou_from_confmat(conf: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    conf: (C,C)
    return: (mIoU, per_class_IoU)  都是 float 张量
    """
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean().item() if valid.any() else 0.0
    return miou, iou


# ----------------------------
#  联合图像/掩码变换
# ----------------------------
class JointResizeFlipToTensor:
    """
    - 统一 Resize 到给定 size
    - 训练时随机水平翻转
    - 图像：ToTensor + Normalize
    - 掩码：转 Long，不做归一化；保持 NEAREST 插值，保留 255 ignore
    """

    def __init__(self, size: int = 512, train: bool = True):
        self.size = size
        self.train = train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        # Resize
        img = TF.resize(img, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

        # Random horizontal flip (train only)
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # ToTensor / Normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)

        # mask -> tensor long（保留原始类别id/255）
        mask = torch.as_tensor(np.array(mask, dtype=np.int64), dtype=torch.long)

        return img, mask


def to_fg_bg(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    将 VOC 的 21 类合并为 {0:背景, 1:前景}，255 保持为 ignore。
    假设 mask ∈ {0..21} ∪ {255}，其中 0=背景，1..20=前景，21（若存在）也视作前景。
    """
    m = mask.clone()
    ign = (m == ignore_index)
    # >0 的都视为前景
    m[(m > 0) & (m != ignore_index)] = 1
    m[ign] = ignore_index
    return m


# ----------------------------
#  模型构建（方案 B）
# ----------------------------
def build_deeplabv3_2c(aux_loss: bool = True) -> nn.Module:
    # 1) 载入分割预训练（21 类 VOC mapping）
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights, aux_loss=aux_loss)

    # 2) 替换分类头 &（可选）辅助头为 2 类
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, 2, kernel_size=1)

    if aux_loss and model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 2, kernel_size=1)

    # 3) 初始化新头参数
    heads = [model.classifier[-1]]
    if aux_loss and model.aux_classifier is not None:
        heads.append(model.aux_classifier[-1])
    for m in heads:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    return model


# ----------------------------
#  数据集构建
# ----------------------------
def build_datasets(
        voc2007_root: str,
        voc2012_root: Optional[str],
        val_year: str = "2007",
        val_split: str = "val",
        size: int = 512,
):
    """
    vocXXXX_root: 指向包含 VOCdevkit 的目录（即该目录下应有 VOCdevkit/VOC2007 或 VOC2012）
    训练集：合并 2007 trainval (+ 可选 2012 trainval)
    验证集：2007 的 val（如你本地确有 test 分割标注，可把 val_split 改为 'test'）
    """
    train_tf = JointResizeFlipToTensor(size=size, train=True)
    eval_tf = JointResizeFlipToTensor(size=size, train=False)

    # 2007 trainval
    ds07_train = VOCSegmentation(
        root=voc2007_root,
        year="2007",
        image_set="trainval",
        download=False,
        transforms=train_tf,  # 注意：是 transforms（联合 img & target）
    )

    datasets = [ds07_train]

    # 2012 trainval（可选）
    if voc2012_root is not None and os.path.isdir(voc2012_root):
        ds12_train = VOCSegmentation(
            root=voc2012_root,
            year="2012",
            image_set="trainval",
            download=False,
            transforms=train_tf,
        )
        datasets.append(ds12_train)

    train_dataset = ConcatDataset(datasets)

    # 验证集（默认 2007 val）
    val_dataset = VOCSegmentation(
        root=voc2007_root,
        year=val_year,
        image_set=val_split,  # 'val'；若你有 test 的标注且路径正确，可改为 'test'
        download=False,
        transforms=eval_tf,
    )

    return train_dataset, val_dataset


# ----------------------------
#  训练 / 验证循环
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, aux_loss=True, ignore_index=255):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    running_loss = 0.0

    t0 = time.time()
    for it, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 二分类：将所有前景并到 1
        masks = to_fg_bg(masks, ignore_index=ignore_index)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)

        loss = criterion(outputs["out"], masks)
        if aux_loss and ("aux" in outputs):
            loss = loss + 0.4 * criterion(outputs["aux"], masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (it + 1) % 50 == 0:
            avg = running_loss / (it + 1)
            print(f"[Epoch {epoch}] iter {it + 1}/{len(loader)}  loss={avg:.4f}")

    elapsed = time.time() - t0
    return running_loss / max(1, len(loader)), elapsed


@torch.no_grad()
def evaluate(model, loader, device, ignore_index=255):
    model.eval()
    num_classes = 2
    conf_total = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        masks = to_fg_bg(masks, ignore_index=ignore_index)

        outputs = model(imgs)["out"]  # (B,2,H,W)
        pred = outputs.argmax(dim=1)  # (B,H,W)

        conf = fast_confusion_matrix(pred, masks, num_classes=num_classes, ignore_index=ignore_index)
        conf_total += conf

    miou, ious = miou_from_confmat(conf_total.cpu())
    return miou, ious.cpu().tolist()


# ----------------------------
#  主函数
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser("DeepLabV3-ResNet50 前景/背景二分类（方案B）训练脚本")
    parser.add_argument("--voc2007-root", type=str, required=True,
                        help="包含 VOCdevkit 的 2007 根目录，例如 /path/to/VOCtrainval_06-Nov-2007")
    parser.add_argument("--voc2012-root", type=str, default=None,
                        help="可选：包含 VOCdevkit 的 2012 根目录，例如 /path/to/VOCtrainval_11-May-2012")
    parser.add_argument("--val-year", type=str, default="2007", choices=["2007", "2012"])
    parser.add_argument("--val-split", type=str, default="val", choices=["val", "test"],
                        help="默认使用 2007 的 val 做验证；如你确实有 test 的分割标注可改为 test")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aux-loss", action="store_true", default=True)
    parser.add_argument("--no-aux-loss", dest="aux_loss", action="store_false")
    parser.add_argument("--save-path", type=str, default="checkpoints/deeplabv3_2c_best.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    # 构建数据集 & 数据加载器
    train_ds, val_ds = build_datasets(
        voc2007_root=args.voc2007_root,
        voc2012_root=args.voc2012_root,
        val_year=args.val_year,
        val_split=args.val_split,
        size=args.size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # 构建模型（方案B）
    model = build_deeplabv3_2c(aux_loss=args.aux_loss)
    model.to(device)

    # 分组学习率：骨干小，头部大
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ]
    if args.aux_loss and (getattr(model, "aux_classifier", None) is not None):
        param_groups.append({"params": model.aux_classifier.parameters(), "lr": args.lr})

    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # 可选：加载断点
    start_epoch = 1
    best_miou = 0.0
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_miou = ckpt.get("best_miou", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}: epoch={start_epoch} best_mIoU={best_miou:.4f}")

    # 训练
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, sec = train_one_epoch(model, train_loader, optimizer, device, epoch, aux_loss=args.aux_loss)
        scheduler.step()

        val_miou, per_class = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}  (bg={per_class[0]:.4f}, fg={per_class[1]:.4f})  "
              f"time={sec:.1f}s")

        # 保存最优
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"✅  Saved best to {args.save_path} (mIoU={best_miou:.4f})")

    print(f"Training done. Best val_mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
