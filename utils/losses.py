import torch
import torch.nn.functional as F

def multilabel_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

def sym_kl_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """对称 KL（多标签 Bernoulli 概率）"""
    p = torch.clamp(p, eps, 1.0-eps); q = torch.clamp(q, eps, 1.0-eps)
    kl_pq = p*torch.log(p/q) + (1-p)*torch.log((1-p)/(1-q))
    kl_qp = q*torch.log(q/p) + (1-q)*torch.log((1-q)/(1-p))
    return (kl_pq + kl_qp).mean()

def js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Jensen–Shannon 散度（逐样本逐类），返回 [B,C]"""
    p = torch.clamp(p, eps, 1.0-eps); q = torch.clamp(q, eps, 1.0-eps)
    m = 0.5*(p+q)
    kl_pm = p*torch.log(p/m)+(1-p)*torch.log((1-p)/(1-m))
    kl_qm = q*torch.log(q/m)+(1-q)*torch.log((1-q)/(1-m))
    return 0.5*(kl_pm+kl_qm)
