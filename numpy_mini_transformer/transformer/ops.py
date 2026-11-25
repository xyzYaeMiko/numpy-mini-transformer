import numpy as np


def clip(x: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    梯度裁切

    :return 输入输出维度一致
    """
    n = np.linalg.norm(x)  # 范数  所有元素的平方和取根号,  一维时为向量长度
    if n > max_norm:
        x = x * (max_norm / (n + 1e-6))
    return x


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """
    减去最大值再取 exp，避免 exp(大数) 溢出。
    """
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)