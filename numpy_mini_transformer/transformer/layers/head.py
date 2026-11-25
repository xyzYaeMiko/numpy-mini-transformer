import numpy as np


class LMHead:
    """语言模型输出头"""

    def __init__(self, d_model: int, vocab_size: int, seed=0):
        """
        是一个单层全连接神经网络

        :param d_model: 词向量维度
        :param vocab_size: 词库数量（定义的token数量）
        :param seed:
        """
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.W = rng.normal(0, s, (d_model, vocab_size))
        self.b = np.zeros(vocab_size)

        self.w_pre = None

        self.dw = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def get_w(self) -> np.ndarray:
        """
        获取权重（一维）
        """
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_w(self, ws: np.ndarray):
        """
        设置权重

        :param ws: (d_model * vocab_size + vocab_size , )
        """
        n = self.W.size
        self.W = ws[:n].reshape(self.W.shape)
        self.b = ws[n:n + self.b.size].reshape(self.b.shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        :param x: (B, L, d_model)
        :return: (B, L, vocab_size)
        """

        self.w_pre = x  # 记录输入
        return x @ self.W + self.b  # (B, L, vocab_size)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        反向传播梯度并叠加保存

        dL/dW = x^T @ dL/dy  # 用于跟新权重
        dL/db = sum(dL/dy)  # 用于更新偏置
        dL/dx = dL/dy @ W^T  # 沿权重继续吧梯度传给上一个
         (B, L, vocab_size)
        :param gradient: (B, L, vocab_size)
        :return: (B, L, d_model)
        """
        # self.x.T @ gradient  # 权重的变化  (B, d_model, L) @ (B, L, vocab_size) -> (d_model, vocab_size)
        dw = np.einsum('bld,blv->dv', self.w_pre, gradient)
        db = gradient.sum(axis=tuple(range(gradient.ndim - 1)))
        # 将梯度沿权重传递到上一个
        dx = gradient @ self.W.T  # (B, L, vocab_size) @ (vocab_size, d_model) -> (B, L, d_model)
        self.dw += dw
        self.db += db
        return dx

    def update_w(self, lr: float):
        """
       跟新权重

       :param lr: 学习率
        """
        self.W -= lr * self.dw
        self.b -= lr * self.db

    def zero_grad(self):
        """
        梯度缓存清零

        """
        self.dw[...] = 0
        self.db[...] = 0

