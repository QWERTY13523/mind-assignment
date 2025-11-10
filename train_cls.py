import os, argparse, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# 分类数据：从 VOC/Main 读取（不需要分割标注）
from datasets.voc_main_cls import VOCDatasetMainCLS, NUM_CLS
# 分类模型
from models.feat_gated_resnet import FeatGatedResNet
# 评测与损失
from utils.metrics import mAP_per_class, micro_f1
from utils.losses import multilabel_bce_with_logits, sym_kl_bernoulli, js_divergence_probs
# 掩膜增强（由分割器推理得到掩膜后使用）
from utils.transforms import apply_mask_only, apply_mask_blur, apply_mask_randbg

def load_seg_infer(ckpt: str, device):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    seg = deeplabv3_resnet50(weights=weights, num_classes=2)
    seg.classifier[-1] = nn.Conv2d(256, 2, 1)
    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu").get("model", None)
        if sd is not None:
            seg.load_state_dict(sd, strict=False)
            print(f"[seg] loaded: {ckpt}")
        else:
            print(f"[seg] warn: ckpt has no 'model' key, ignore")
    seg.eval().to(device)
    for p in seg.parameters(): p.requires_grad = False
    return seg

@torch.no_grad()
def make_mask(seg, img_normed):
    # img_normed: 已经做过 ImageNet 归一化
    out = seg(img_normed)["out"]               # [B,2,H,W]
    prob = torch.softmax(out, 1)[:, 1:2]       # 前景概率
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
            x = b["image"].to(device)
            y = b["multilabel"].to(device)
            mask = make_mask(seg_model, x)
            x_aug = apply_strategy(x, mask, args.mask_strategy, args.bg_dir)
            p  = torch.sigmoid(model(x,     mask if args.gate_alpha>0 else None))
            p2 = torch.sigmoid(model(x_aug, mask if args.gate_alpha>0 else None))
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
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--gate-alpha", type=float, default=0.8)
    ap.add_argument("--mask-strategy", type=str, default="mixed",
                    choices=["maskonly","blur","randbg","mixed"])
    ap.add_argument("--bg-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default="checkpoints/cls/best.pth")
    ap.add_argument("--gate-alpha", type=float, default=0.8)
    ap.add_argument("--ring-beta", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seg_model = load_seg_infer(args.seg_ckpt, device)
    cls_model = FeatGatedResNet(num_classes=NUM_CLS,
                                gate_alpha=args.gate_alpha,
                                ring_beta=args.ring_beta,
                                pretrained=True).to(device)

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
    best = 0.0

    for ep in range(1, args.epochs+1):
        cls_model.train(); tot = 0.0
        for b in tl:
            x = b["image"].to(device)
            y = b["multilabel"].to(device)
            with torch.no_grad():
                mask = make_mask(seg_model, x)
                x_aug = apply_strategy(x, mask, args.mask_strategy, args.bg_dir)

            opt.zero_grad(set_to_none=True)
            logits     = cls_model(x,     mask if args.gate_alpha>0 else None)
            logits_aug = cls_model(x_aug, mask if args.gate_alpha>0 else None)

            loss_cls  = multilabel_bce_with_logits(logits, y)
            prob      = torch.sigmoid(logits).detach()
            prob_aug  = torch.sigmoid(logits_aug)
            loss_cons = sym_kl_bernoulli(prob, torch.clamp(prob_aug,1e-6,1-1e-6))
            loss = loss_cls + args.lam * loss_cons
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)

        sch.step()
        mAP, f1, cir, _ = evaluate(cls_model, seg_model, vl, device, args)
        print(f"[Cls][Epoch {ep:03d}] loss={tot/len(train_set):.6f}  mAP(test2007)={mAP:.4f}  F1={f1:.4f}  CIR={cir:.4f}")
        if mAP > best:
            best = mAP
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": cls_model.state_dict(), "epoch": ep, "mAP": mAP, "args": vars(args)}, args.out)
            print(f"  >> save {args.out} (mAP={mAP:.4f})")

    mAP, f1, cir, _ = evaluate(cls_model, seg_model, vl, device, args)
    print(f"[Final] mAP(test2007)={mAP:.4f}  F1={f1:.4f}  CIR={cir:.4f}")

if __name__ == "__main__":
    main()
