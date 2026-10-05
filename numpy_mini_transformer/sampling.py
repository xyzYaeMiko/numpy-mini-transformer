import numpy as np


def top_k_filter(logits, k):
    if k is None or k <= 0:
        return logits
    # 每行第 k 大的值
    kth = np.sort(logits, axis=-1)[:, -k][:, None]
    return np.where(logits < kth, -1e9, logits)


def top_p_filter(logits, p):
    if p is None or p >= 1.0:
        return logits
    probs = softmax(logits)
    order = np.argsort(-probs, axis=-1)  # 从大到小
    sorted_probs = np.take_along_axis(probs, order, axis=-1)
    cum = np.cumsum(sorted_probs, axis=-1)
    # 超过 p 的位置置 0
    cutoff = cum > p
    cutoff[:, 1:] = cutoff[:, :-1].copy()
    cutoff[:, 0] = False
    sorted_probs = np.where(cutoff, 0.0, sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdims=True)
    # 还原顺序
    out = np.zeros_like(probs)
    np.put_along_axis(out, order, sorted_probs, axis=-1)
    return np.log(out + 1e-12)  # 转回 logits


def sample(logits, temperature=1.0, top_k=None, top_p=None) -> np.ndarray:
    """
    :param top_p:
    :param top_k:
    :param temperature:
    :param logits: (B, vocab_size)
    :return: (B,) int
    """
    if temperature != 1.0:
        logits = logits / temperature

    logits = top_k_filter(logits, top_k)  # 抓取top_k
    logits = top_p_filter(logits, top_p)  # 抓取top_p

    probs = softmax(logits)  # (B, V)
    B, V = probs.shape

    rng = np.random.default_rng(0)   # 固定种子, 也可选择不固定
    next_ids = np.array([
        rng.choice(V, p=probs[b]) for b in range(B)
    ])
    return next_ids


def softmax(logits_last):
    """
    x = x - max(x) # 将其限制在负
    x = exp(x)  # 指数放大差距（更容易采样）
    y = x / Σx  # 归一化
    :param logits_last:(B, vocab_size)
    :return probs:(B, vocab_size)
    """
    logits = logits_last - logits_last.max(axis=-1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return probs
