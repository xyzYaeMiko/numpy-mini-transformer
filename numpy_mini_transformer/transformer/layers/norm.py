import numpy as np


class LayerNorm:
    """
    归一化层
    对每个 token 的 d_model维向量，单独减均值、除标准差，再用可学习的 γ,β 缩放平移。
    稳定训练的组件
    """

    def __init__(self, d_model: int, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

        self.x_hat = None
        self.std = None

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def get_w(self) -> np.ndarray:
        """
        获取γ,β参数（一维）

        :return (d_model * 2, .)
        """
        return np.concatenate([self.gamma.ravel(), self.beta.ravel()])

    def set_w(self, ws: np.ndarray):
        """
        设置γ,β参数

         :param ws: (d_model * 2, )
        """
        n = self.gamma.size
        self.gamma = ws[:n].reshape(self.gamma.shape)
        self.beta = ws[n:2 * n].reshape(self.beta.shape)

        self.x_hat = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        μ  = (1/d) * Σ_{i=1}^d x_i
        σ = (1/d) * Σ_{i=1}^d (x_i - μ)^2
        y_i = γ_i * (x_i - μ) / sqrt(σ) + β
        :param x: (B, L, d_model)
        :return:-> (B, L, d_model)
        """

        # x:
        mu = x.mean(axis=-1, keepdims=True)  # (B, L, 1) 向量的平均值
        var = x.var(axis=-1, keepdims=True)  # (B, L, 1) 向量的方差

        self.std = np.sqrt(var + self.eps)
        self.x_hat = (x - mu) / np.sqrt(
            var + self.eps)  # ((B, L, d_model) - (B, L, 1（广播）)) -> (B, L, d_modle)) / (B, L, 1（广播）) -> (B, L, d_model)
        return self.gamma * self.x_hat + self.beta  # -> (B, L, d_model)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        反向传播梯度并叠加保存

        :param gradient: (B, L, d_model)
        :return: (B, L, d_model)
        """
        axes = tuple(range(gradient.ndim - 1))  #
        dgamma = (gradient * self.x_hat).sum(axis=axes)  # 每个元素的梯度成乘以输入，然后再L轴上求和， 再沿B轴求和 ——> (1, d_model)
        dbeta = gradient.sum(axis=axes)  # 梯度沿L轴求和， 再沿B轴求和 -> (1, dmodel)
        self.dgamma += dgamma
        self.dbeta += dbeta

        dx_hat = gradient * self.gamma
        m = dx_hat.mean(axis=-1, keepdims=True)
        v = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        dx = (dx_hat - m - self.x_hat * v) / np.maximum(self.std, 1e-5)

        return dx  # (B, L, d_model)

    def update_w(self, lr: float):
        """
        跟新权重

        :param lr: 学习率
        """
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta
        ...

    def zero_grad(self):
        """
        梯度缓存清零

        """
        self.dgamma[...] = 0
        self.dbeta[...] = 0

