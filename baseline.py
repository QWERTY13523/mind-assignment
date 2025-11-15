import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset

# 数据 & 模型 & 工具：沿用你仓库里的实现
from datasets.voc_main_cls import VOCDatasetMainCLS, NUM_CLS
from models.feat_gated_resnet import FeatGatedResNet
from utils.losses import multilabel_bce_with_logits
from utils.metrics import mAP_per_class, micro_f1


@torch.no_grad()
def evaluate_baseline(model, loader, device, args):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["multilabel"].to(device, non_blocking=True)

            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)

            with autocast(enabled=args.amp):
                # 这里不再输入 mask，纯分类
                logits = model(x, None) if args.use_mask_arg else model(x)
                p = torch.sigmoid(logits)

            ys.append(y.cpu().numpy())
            ps.append(p.cpu().numpy())

    ys = np.concatenate(ys, 0)
    ps = np.concatenate(ps, 0)
    aps, mAP = mAP_per_class(ys, ps)
    f1 = micro_f1(ys, (ps >= 0.5).astype(np.float32))
    return mAP, f1, aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc2007-root", type=str, required=True, help=".../VOCdevkit/VOC2007")
    ap.add_argument("--voc2012-root", type=str, required=True, help=".../VOCdevkit/VOC2012")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=45)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="checkpoints/cls_baseline/best.pth")
    ap.add_argument("--ring-beta", type=float, default=0.0, help="设成0相当于关掉 ring 正则（如果模型里用到了）")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    ap.add_argument("--resume", type=str, default="", help="从分类 checkpoint 继续训练")
    ap.add_argument(
        "--use-mask-arg",
        action="store_true",
        default=True,
        help="FeatGatedResNet 的 forward(x, mask=None) 形式；如果你后来改成只收 x，就关掉它",
    )
    args = ap.parse_args()

    # 设备
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # TF32（Ampere+ 上加速）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # ===== 模型：不再使用分割，gate_alpha=0 关掉特征门控 =====
    cls_model = FeatGatedResNet(
        num_classes=NUM_CLS,
        gate_alpha=0.0,          # 不用分割掩膜做 gating
        ring_beta=args.ring_beta,
        pretrained=True,
    ).to(device)

    if args.channels_last:
        cls_model = cls_model.to(memory_format=torch.channels_last)

    # ===== 继续训练（可选） =====
    start_epoch = 1
    best = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> resume from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = cls_model.load_state_dict(state, strict=False)
            print(
                f"   loaded epoch={ckpt.get('epoch', '?')}, "
                f"mAP={ckpt.get('mAP', 0.0):.4f}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            best = float(ckpt.get("mAP", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        else:
            print(f"[warn] resume checkpoint not found: {args.resume}")
    else:
        print("=> train from scratch (no resume)")

    # ===== 数据：训练 = VOC07 trainval ∪ VOC12 trainval; 测试 = VOC07 test =====
    ds07 = VOCDatasetMainCLS(args.voc2007_root, split="trainval", img_size=args.img_size, random_flip=True)
    ds12 = VOCDatasetMainCLS(args.voc2012_root, split="trainval", img_size=args.img_size, random_flip=True)
    train_set = ConcatDataset([ds07, ds12])

    test_set = VOCDatasetMainCLS(
        args.voc2007_root, split="test", img_size=args.img_size, random_flip=False
    )

    tl = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    vl = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ===== 优化器 & 学习率调度 =====
    opt = torch.optim.AdamW(cls_model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    # ===== 训练循环：纯 BCE，多标签分类 baseline =====
    for ep in range(start_epoch, args.epochs + 1):
        cls_model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)

        for bidx, b in enumerate(tl, 1):
            x = b["image"].to(device, non_blocking=True)
            y = b["multilabel"].to(device, non_blocking=True)

            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)

            if (bidx - 1) % max(1, args.accum) == 0:
                opt.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                logits = cls_model(x, None) if args.use_mask_arg else cls_model(x)
                loss = multilabel_bce_with_logits(logits, y)
                loss = loss / max(1, args.accum)

            scaler.scale(loss).backward()

            if (bidx % max(1, args.accum)) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            tot += loss.item() * b["image"].size(0)

        # 不足一个 accum 的尾巴
        if (len(tl) % max(1, args.accum)) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sch.step()

        # ===== 验证（VOC2007 test）=====
        mAP, f1, _ = evaluate_baseline(cls_model, vl, device, args)
        print(
            f"[Baseline][Epoch {ep:03d}] loss={tot / len(train_set):.6f}  "
            f"mAP(test2007)={mAP:.4f}  F1={f1:.4f}"
        )

        if mAP > best:
            best = mAP
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(
                {"model": cls_model.state_dict(), "epoch": ep, "mAP": mAP, "args": vars(args)},
                args.out,
            )
            print(f"  >> save {args.out} (mAP={mAP:.4f})")


if __name__ == "__main__":
    main()
