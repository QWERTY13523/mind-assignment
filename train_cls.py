import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50

# 分类数据：从 VOC/Main 读取（不需要分割标注）
from datasets.voc_main_cls import VOCDatasetMainCLS, NUM_CLS
# 分类模型
from models.feat_gated_resnet import FeatGatedResNet
from utils.losses import multilabel_bce_with_logits, sym_kl_bernoulli, js_divergence_probs
# 评测与损失
from utils.metrics import mAP_per_class, micro_f1
# 掩膜增强（由分割器推理得到掩膜后使用）
from utils.transforms import apply_mask_only, apply_mask_blur, apply_mask_randbg


def load_seg_infer(ckpt: str, device, aux_loss: bool = True):
    # 直接构建2类结构并从你的ckpt加载
    seg = deeplabv3_resnet50(weights=None, aux_loss=aux_loss)
    seg.classifier[-1] = nn.Conv2d(seg.classifier[-1].in_channels, 2, kernel_size=1)
    if aux_loss and getattr(seg, "aux_classifier", None) is not None:
        seg.aux_classifier[-1] = nn.Conv2d(seg.aux_classifier[-1].in_channels, 2, kernel_size=1)

    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        missing, unexpected = seg.load_state_dict(state, strict=False)
        print(f"[seg] loaded: {ckpt} | missing: {len(missing)} unexpected: {len(unexpected)}")
    else:
        print(f"[seg] warn: ckpt not found -> {ckpt}")

    seg.eval().to(device)
    for p in seg.parameters():
        p.requires_grad = False
    return seg


@torch.no_grad()
def make_mask(seg, img_normed, seg_infer_size=None, amp=True):
    # img_normed: 已归一化
    x = img_normed
    if seg_infer_size is not None:
        if isinstance(seg_infer_size, int):
            h, w = img_normed.shape[-2:]
            if h < w:
                new_h = seg_infer_size
                new_w = int(w * new_h / h)
            else:
                new_w = seg_infer_size
                new_h = int(h * new_w / w)
            x = F.interpolate(img_normed, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(img_normed, size=seg_infer_size, mode="bilinear", align_corners=False)

    with autocast(enabled=amp):
        out = seg(x)["out"]  # [B,2,h',w']
        prob = torch.softmax(out, 1)[:, 1:2]

    if x.shape[-2:] != img_normed.shape[-2:]:
        prob = F.interpolate(prob, size=img_normed.shape[-2:], mode="bilinear", align_corners=False)

    return (prob > 0.5).float()


def apply_strategy(img, mask, strategy: str, bg_dir=None):
    if strategy == "maskonly": return apply_mask_only(img, mask)
    if strategy == "blur":     return apply_mask_blur(img, mask)
    if strategy == "randbg":   return apply_mask_randbg(img, mask, bg_dir)
    if strategy == "mixed":
        import random
        return apply_strategy(img, mask, random.choice(["maskonly","blur","randbg"]), bg_dir)
    return img


def evaluate(model, seg_model, loader, device, args):
    model.eval(); ys, ps, js_all = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["multilabel"].to(device, non_blocking=True)

            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)

            mask = make_mask(seg_model, x, seg_infer_size=args.seg_infer_size, amp=args.amp).to(dtype=x.dtype)
            x_aug = apply_strategy(x, mask, args.mask_strategy, args.bg_dir)
            if args.channels_last:
                x_aug = x_aug.to(memory_format=torch.channels_last)

            with autocast(enabled=args.amp):
                p = torch.sigmoid(model(x, mask if args.gate_alpha > 0 else None))
                p2 = torch.sigmoid(model(x_aug, mask if args.gate_alpha > 0 else None))

            ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
            js = js_divergence_probs(p, p2).mean(1)
            js_all.append(js.cpu().numpy())

    ys = np.concatenate(ys, 0); ps = np.concatenate(ps, 0)
    aps, mAP = mAP_per_class(ys, ps)
    f1 = micro_f1(ys, (ps>=0.5).astype(np.float32))
    cir = float(np.mean(np.concatenate(js_all,0))) if js_all else 0.0
    return mAP, f1, cir, aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc2007-root", type=str, required=True, help=".../VOCdevkit/VOC2007")
    ap.add_argument("--voc2012-root", type=str, required=True, help=".../VOCdevkit/VOC2012")
    ap.add_argument("--seg-ckpt", type=str, default="checkpoints/seg/best.pth")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--gate-alpha", type=float, default=0.7)
    ap.add_argument("--mask-strategy", type=str, default="mixed",
                    choices=["maskonly","blur","randbg","mixed"])
    ap.add_argument("--bg-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default="checkpoints/cls/best.pth")
    ap.add_argument("--ring-beta", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    ap.add_argument("--seg-infer-size", type=int, default=256, help="分割掩膜的推理短边（None=不降采样）")
    ap.add_argument("--resume", type=str, default="", help="checkpoint 路径，用于继续训练")

    args = ap.parse_args()

    device = torch.device("cuda:3")
    # TF32（Amp下进一步加速/省显存，NVIDIA Ampere+）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    seg_model = load_seg_infer(args.seg_ckpt, device)
    cls_model = FeatGatedResNet(num_classes=NUM_CLS,
                                gate_alpha=args.gate_alpha,
                                ring_beta=args.ring_beta,
                                pretrained=True).to(device)
    if args.channels_last:
        cls_model = cls_model.to(memory_format=torch.channels_last)

    start_epoch = 1
    best = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> resume from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            # 为了兼容性，strict=False，防止你以后稍微改网络结构
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
    # 训练集 = VOC2007 trainval ∪ VOC2012 trainval
    ds07 = VOCDatasetMainCLS(args.voc2007_root, split="trainval", img_size=args.img_size, random_flip=True)
    ds12 = VOCDatasetMainCLS(args.voc2012_root, split="trainval", img_size=args.img_size, random_flip=True)
    train_set = ConcatDataset([ds07, ds12])

    # 测试集 = VOC2007 test
    test_set  = VOCDatasetMainCLS(args.voc2007_root, split="test", img_size=args.img_size, random_flip=False)

    tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(cls_model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    for ep in range(start_epoch, args.epochs + 1):
        cls_model.train();
        tot = 0.0
        opt.zero_grad(set_to_none=True)

        for bidx, b in enumerate(tl, 1):
            x = b["image"].to(device, non_blocking=True)
            y = b["multilabel"].to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)

            # 1) 用分割器生成 mask 和 x_aug（保持不变）
            with torch.no_grad():
                mask = make_mask(seg_model, x, seg_infer_size=args.seg_infer_size, amp=args.amp).to(dtype=x.dtype)
                x_aug = apply_strategy(x, mask, args.mask_strategy, args.bg_dir)
                if args.channels_last:
                    x_aug = x_aug.to(memory_format=torch.channels_last)

            if (bidx - 1) % max(1, args.accum) == 0:
                opt.zero_grad(set_to_none=True)

            # 2) 主分支：原图 BCE
            with autocast(enabled=args.amp):
                logits = cls_model(x, mask if args.gate_alpha > 0 else None)
                loss_main = multilabel_bce_with_logits(logits, y)
                loss_main = loss_main / max(1, args.accum)

            scaler.scale(loss_main).backward()
            del logits, x
            torch.cuda.empty_cache()

            # 3) 增强分支：掩膜增强图 BCE（带权重 lam）
            with autocast(enabled=args.amp):
                logits_aug = cls_model(x_aug, mask if args.gate_alpha > 0 else None)
                loss_aug = multilabel_bce_with_logits(logits_aug, y)
                loss_aug = args.lam * loss_aug  # lam 当权重
                loss_aug = loss_aug / max(1, args.accum)

            scaler.scale(loss_aug).backward()
            del logits_aug, x_aug
            torch.cuda.empty_cache()

            # 4) 梯度更新
            if (bidx % max(1, args.accum)) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            tot += (loss_main.item() + loss_aug.item()) * (b["image"].size(0))

        # 若最后不足一个accum，补一次step
        if (len(tl) % max(1, args.accum)) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sch.step()

        # —— 验证（保持你原来的 evaluate）——
        mAP, f1, cir, _ = evaluate(cls_model, seg_model, vl, device, args)
        print(
            f"[Cls][Epoch {ep:03d}] loss={tot / len(train_set):.6f}  mAP(test2007)={mAP:.4f}  F1={f1:.4f}  CIR={cir:.4f}")
        if mAP > best:
            best = mAP
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": cls_model.state_dict(), "epoch": ep, "mAP": mAP, "args": vars(args)}, args.out)
            print(f"  >> save {args.out} (mAP={mAP:.4f})")


if __name__ == "__main__":
    main()
