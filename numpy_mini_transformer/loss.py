import numpy as np


def softmax_last(x: np.ndarray):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """
    :param logits:  (B, L, V)
    :param targets: (B, L) int
    :return: loss(scalar), dlogits_flat (B, L, vocab_size)
    """
    B, L, vocab_size = logits.shape
    N = B * vocab_size
    N = B * L

    logits_flat = logits.reshape(N, vocab_size)
    targets_flat = targets.reshape(N)

    logits_flat = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_flat)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    loss = -np.log(probs[np.arange(N), targets_flat] + 1e-12).mean()  # 目标位置处取值的对数

    dlogits_flat = probs.copy()
    dlogits_flat[np.arange(N), targets_flat] -= 1
    dlogits_flat /= N

    return loss, dlogits_flat.reshape(B, L, vocab_size)  # floor  (B, L, vocab_size)


def clip(x: np.ndarray, max_norm: float = 1.0):
    n = np.linalg.norm(x)
    if n > max_norm:
        x = x * (max_norm / (n + 1e-6))
    return x
