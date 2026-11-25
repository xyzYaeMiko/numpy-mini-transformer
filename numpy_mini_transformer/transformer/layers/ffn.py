import numpy as np

class FFN:
    """
    Transformer 中的前馈神剑网络, 一个双层全连接神经网络
    用于学习 token 语义
    """

    def __init__(self, input_size: int, hidden_size: int = None, output_size: int = None, seed=0):
        """

        :param input_size: d_v, 与注意力输出向量维度一致
        :param hidden_size: 隐藏层
        :param ouput_size: d_model， 需为词向量的维度
        :param seed:
        """
        h_size = hidden_size or 4 * input_size
        o_size = output_size or input_size

        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(input_size)

        self.W_ih = rng.normal(0, s, (input_size, h_size))
        self.b_h = np.zeros(h_size)
        self.W_ho = rng.normal(0, s, (h_size, o_size))
        self.b_o = np.zeros(o_size)

        self.w_ih_per = None
        self.relu_pre = None
        self.w_ho_pre = None

        self.dw_ih = np.zeros_like(self.W_ih)
        self.db_h = np.zeros_like(self.b_h)
        self.dw_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)

    @staticmethod
    def ReLU(x: np.ndarray):
        return np.maximum(x, 0)

    def get_w(self) -> np.ndarray:
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

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

    def backward(self, gradient: np.ndarray, ) -> np.ndarray:
        """
        反向传播梯度并叠加保存

        :param gradient: (B, L, d_model)
        :return: (B, L, d_model)
        """

        dw_ho = np.einsum('blh,bld->hd', self.w_ho_pre, gradient)  # (B, d_model, L) @ (B, L, d_model) -> (L, d_model)
        db_o = gradient.sum(axis=tuple(range(gradient.ndim - 1)))
        g = gradient @ self.W_ho.T  # (B, L, H)

        # ReLU
        gh_pre = g * (self.relu_pre > 0)

        dw_ih = np.einsum('bld,blh->dh', self.w_ih_per, gh_pre)
        db_h = gh_pre.sum(axis=tuple(range(gh_pre.ndim - 1)))
        dx = gh_pre @ self.W_ih.T  # (B, L, d_model)

        self.dw_ih += dw_ih
        self.db_h += db_h
        self.dw_ho += dw_ho
        self.db_o += db_o
        return dx

    def update_w(self, lr: float):
        """
       跟新权重

       :param lr: 学习率
       """
        self.W_ih -= lr * self.dw_ih
        self.b_h -= lr * self.db_h
        self.W_ho -= lr * self.dw_ho
        self.b_o -= lr * self.db_o

    def zero_grad(self):
        """
        梯度缓存清零

        """
        self.dw_ih[...] = 0
        self.db_h[...] = 0
        self.dw_ho[...] = 0
        self.db_o[...] = 0
