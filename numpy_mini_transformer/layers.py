import numpy as np


class LayerNorm:
    """
    对每个 token 的 d_model维向量，单独减均值、除标准差，再用可学习的 γ,β 缩放平移。
    稳定训练的组件
    """

    def __init__(self, d_model: int, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

        self.x_hat = None
        self.std = None

    def get_w(self):
        return np.concatenate([self.gamma.ravel(), self.beta.ravel()])

    def set_w(self, ws: np.ndarray):
        n = self.gamma.size
        self.gamma = ws[:n].reshape(self.gamma.shape)
        self.beta = ws[n:2 * n].reshape(self.beta.shape)

        self.x_hat = None

    def __call__(self, x: np.ndarray):
        """
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

    def backward(self, gradient: np.ndarray, lr: float):
        """

        :param gradient: (B, L, d_model)
        :param lr:
        :return: (B, L, d_model)
        """
        axes = tuple(range(gradient.ndim - 1))  #
        dgamma = (gradient * self.x_hat).sum(axis=axes)  # 每个元素的梯度成乘以输入，然后再L轴上求和， 再沿B轴求和 ——> (1, d_model)
        dbeta = gradient.sum(axis=axes)  # 梯度沿L轴求和， 再沿B轴求和 -> (1, dmodel)

        dx_hat = gradient * self.gamma
        m = dx_hat.mean(axis=-1, keepdims=True)
        v = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        dx = (dx_hat - m - self.x_hat * v) / np.maximum(self.std, 1e-5)

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta
        return dx  # (B, L, d_model)


class FFN:
    """
    Transformer 中的前馈神剑网络, 一个双层全连接神经网络
    将注意力输出的d_v（d_model）维向量 变换成 d_model 维的向量
    用于学习 token 语义
    """

    def __init__(self, input_size: int, hidden_size: int = None, ouput_size: int = None, seed=0):
        """

        :param input_size: d_v, 与注意力输出向量维度一致
        :param hidden_size: 隐藏层
        :param ouput_size: d_model， 需为词向量的维度
        :param seed:
        """
        h_size = hidden_size or 4 * input_size
        o_size = ouput_size or input_size

        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(input_size)

        self.W_ih = rng.normal(0, s, (input_size, h_size))
        self.b_h = np.zeros(h_size)
        self.W_ho = rng.normal(0, s, (h_size, o_size))
        self.b_o = np.zeros(o_size)

        self.w_ih_per = None
        self.relu_pre = None
        self.w_ho_pre = None

    @staticmethod
    def ReLU(x: np.ndarray):
        return np.maximum(x, 0)

    def get_w(self):
        return np.concatenate([
            self.W_ih.ravel(), self.b_h.ravel(),
            self.W_ho.ravel(), self.b_o.ravel()
        ])

    def set_w(self, ws: np.ndarray):
        idx = 0
        for p in [self.W_ih, self.b_h, self.W_ho, self.b_o]:
            n = p.size
            p[...] = ws[idx:idx + n].reshape(p.shape)
            idx += n

    def forward(self, x: np.ndarray):
        """

        :param x: (B, L, d_v)
        :return:(B, L, d_model)
        """
        self.w_ih_per = x
        x = x @ self.W_ih + self.b_h  # (B, L, d_v) @ (d_v, h_size) + (h_size) -> (B, L, h_size)
        self.relu_pre = x
        x = self.ReLU(x)
        self.w_ho_pre = x
        x = x @ self.W_ho + self.b_o  # (B, L, h_size) @ (h_size, d_model) + (d_model) -> (B, L, d_model)
        return x  # (B, L, d_model)

    def backward(self, gradient: np.ndarray, lr: float):
        """

        :param gradient: (B, L, d_model)
        :param lr:
        :return:
        """

        dw_ho = np.einsum('blh,bld->hd', self.w_ho_pre, gradient)  # (B, d_model, L) @ (B, L, d_model) -> (L, d_model)
        db_o = gradient.sum(axis=tuple(range(gradient.ndim - 1)))
        g = gradient @ self.W_ho.T  # (B, L, H)

        # ReLU
        gh_pre = g * (self.relu_pre > 0)

        dw_ih = np.einsum('bld,blh->dh', self.w_ih_per, gh_pre)
        db_h = gh_pre.sum(axis=tuple(range(gh_pre.ndim - 1)))
        dx = gh_pre @ self.W_ih.T  # (B, L, d_model)

        # update
        self.W_ih -= lr * dw_ih
        self.b_h -= lr * db_h
        self.W_ho -= lr * dw_ho
        self.b_o -= lr * db_o
        return dx
