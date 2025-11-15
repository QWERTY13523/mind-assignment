import torch
import torch.nn.functional as F


def multilabel_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _to_fp32(x: torch.Tensor) -> torch.Tensor:
    """在 AMP 下，把半精度张量先抬到 float32，避免 log 等操作数值炸掉。"""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def sym_kl_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对称 KL（多标签 Bernoulli 概率）
    这里强制在 float32 里做运算，避免 AMP 半精度导致 NaN。
    """
    p = _to_fp32(p)
    q = _to_fp32(q)

    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)

    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    # 返回标量 loss（float32 即可，给 GradScaler 用没问题）
    return (kl_pq + kl_qp).mean()


def js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Jensen–Shannon 散度（逐样本逐类），返回 [B,C]
    同样在 float32 里算，避免 AMP 下数值不稳定。
    """
    p = _to_fp32(p)
    q = _to_fp32(q)

    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)

    m = 0.5 * (p + q)
    kl_pm = p * torch.log(p / m) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - m))
    kl_qm = q * torch.log(q / m) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - m))
    return 0.5 * (kl_pm + kl_qm)
