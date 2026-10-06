import numpy as np


def softmax_last(x: np.ndarray) -> np.ndarray:
    """
    转归一分布
    """
    x = x - x.max(axis=-1, keepdims=True)  # 每个都去掉一个最大值 防止数值溢出（不改变结果），softmax的平移不变性
    e = np.exp(x)  # 使用e指数放大差距
    return e / e.sum(axis=-1, keepdims=True)  # 归一化


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float and np.ndarray:
    """
    交叉熵损失，返回平均损失和梯度

    交叉熵（单标签）： L = -1 * log(p)  p：分布， -1： 目标分布
    梯度:  (p - p_target) / N
    :param logits:  (B, L, V)
    :param targets: (B, L) int
    :return: loss(scalar), dlogits_flat (B, L, vocab_size)
    """
    B, L, vocab_size = logits.shape

    # 将数据处理成 分布列表  [p, p, ... ]  对于目标来说分布只有一个且为一
    N = B * L  # 参与计算的token总数
    logits_flat = logits.reshape(N, vocab_size)  #  (B*L, vocab_size)  [B1L1, B1L2, ..., B2L1, ...]  将每个token的输出分布横向展开
    targets_flat = targets.reshape(N) # (B*L)   每个token的目标分布（由于只有单个标签，所以分布为1）

    # 使用 softmax
    probs = softmax_last(logits_flat)

    # 交叉熵（单标签）  L = -1* log(p)
    loss = -np.log(probs[np.arange(N), targets_flat] + 1e-12).mean()  # 提取预测里头的目标的概率。将目标的目标概率定为1,进行交叉熵

    # 计算梯度
    dlogits_flat = probs.copy()
    dlogits_flat[np.arange(N), targets_flat] -= 1  # 在目标处减去目标概率（1） 获得 dp 概率的差分
    dlogits_flat /= N  # 计算梯度

    return loss, dlogits_flat.reshape(B, L, vocab_size)  # floor  (B, L, vocab_size)


def clip(x: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    梯度裁切

    :return 输入输出维度一致
    """
    n = np.linalg.norm(x)  # 范数  所有元素的平方和取根号,  一维时为向量长度
    if n > max_norm:
        x = x * (max_norm / (n + 1e-6))
    return x
