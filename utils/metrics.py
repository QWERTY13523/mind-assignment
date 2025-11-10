import numpy as np

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = (y_true == 1).astype(np.float32)
    fp = (y_true == 0).astype(np.float32)
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / np.maximum(tp.sum(), 1e-12)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

def mAP_per_class(y_true: np.ndarray, y_score: np.ndarray):
    N, C = y_true.shape
    aps = [average_precision(y_true[:,c], y_score[:,c]) for c in range(C)]
    return np.array(aps), float(np.mean(aps))

def micro_f1(y_true: np.ndarray, y_pred_bin: np.ndarray):
    tp = (y_true * y_pred_bin).sum()
    fp = ((1 - y_true) * y_pred_bin).sum()
    fn = (y_true * (1 - y_pred_bin)).sum()
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    return float(2 * precision * recall / max(precision + recall, 1e-12))
