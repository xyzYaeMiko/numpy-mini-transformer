import numpy as np


class SingleHeadAttention:
    """
    单头注意力模块
    """

    def __init__(self, d_model: int, d_k: int = None, causal: bool = True, seed=0):
        """

        :param d_model: 词向量维度
        :param d_k: 查询向量 和 键向量 的维度
        :param causal: 因果
        :param seed:
        """
        d_k = d_k or d_model
        d_v = d_model

        self.d_model, self.d_k, self.d_v = d_model, d_k, d_v
        self.causal = causal

        # 一般用 Xavier/小方差初始化，这里简单缩放
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(0, s, (d_model, d_k))
        self.W_k = rng.normal(0, s, (d_model, d_k))
        self.W_v = rng.normal(0, s, (d_model, d_v))
        self.W_o = rng.normal(0, s, (d_v, d_model))

        self.K_cache = None  # (B, L, d_k)
        self.V_cache = None  # (B, L, d_v)

        self.x = None
        self.Q = None
        self.K = None
        self.V = None
        self.mask = None

        self.w_o_pre = None
        self.v_pre = None

    @staticmethod
    def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
        """
        减去最大值再取 exp，避免 exp(大数) 溢出。
        """
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None) -> np.ndarray:
        """
                缩放点积注意力（核心公式）
        Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

        L :单个batch里含有的查询向量的数量，一般为token数或1
        S :单个batch里含有的键向量的数量，与token数一致
        :param Q: (B, L, d_k)  查询
        :param K: (B, S, d_k)  健
        :param V: (B, S, d_v)  值
        :param mask: (B, L, S) 的 bool 数组，True = 保留，False = 屏蔽
        :return: out # (B, L, d_v)
        """
        d_k = Q.shape[-1]

        scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)  # ( (B, L, d_k) @ (B, d_k, S) )/sqrt(d_k) -> (B, L, S)
        # 其中单batch中，该矩阵为 token相关向量（横向量）堆叠而成
        if mask is not None:
            # 被屏蔽的位置给 -inf，softmax 后权重为 0
            scores = np.where(mask, scores, -1e9)  # 防止产生NaN
        attn = SingleHeadAttention.softmax(scores, axis=-1)  # 每一行和为 1
        self.v_pre = attn
        out = attn @ V  # (B, L, S) @ (B, S, d_v) -> (B, L, d_v)
        return out  # (B, L, d_v)

    def get_w(self) -> np.ndarray:
        """
        将所有参数展平后拼接成一个一维向量。
        """
        return np.concatenate([
            self.W_q.ravel(), self.W_k.ravel(),
            self.W_v.ravel(), self.W_o.ravel()
        ])

    def set_w(self, ws: np.ndarray):
        """
        从一维向量中按原始形状恢复参数。
        """
        idx = 0
        for p in [self.W_q, self.W_k, self.W_v, self.W_o]:
            n = p.size
            p[...] = ws[idx:idx + n].reshape(p.shape)
            idx += n

    def update_cache(self, K_new: np.ndarray, V_new: np.ndarray):
        """
        每个对话（batch）的单个token的键&值向量

        :param K_new: (B, 1, d_model)
        :param V_new: (B, 1, d_model)
        :return:
        """
        if self.K_cache is None:
            self.K_cache = K_new
            self.V_cache = V_new
        else:
            self.K_cache = np.concatenate([self.K_cache, K_new], axis=1)  # (B, L+1, d_k)
            self.V_cache = np.concatenate([self.V_cache, V_new], axis=1)  # (B, L+1, d_v)

    def clear_KV_cache(self):
        self.K_cache = None
        self.V_cache = None

    def get__kv_cache(self) -> np.ndarray:
        """
        :return (2, B, L, d_k)  if d_v == d_k
        """
        return np.array([self.K_cache, self.V_cache])

    def set__kv_cache(self, kv_cache: np.ndarray):
        """
        :param kv_cache: [k_cache, v_cache]  (2, B, L, d_k)
        """
        assert kv_cache.shape[0] == 2

        self.K_cache = kv_cache[0]
        self.V_cache = kv_cache[1]

    def forward(self, x: np.ndarray, mask=None, kv_cache: bool = False) -> np.ndarray:
        """
         前向传播

        :param x: (B, L, d_model)
        :param mask:
        :param kv_cache: bool 是否启用KV缓存
        :return: out: (B, L, d_model)
        """

        if kv_cache and self.K_cache is not None:  # 在KV缓存模式且存在缓存时
            return self._KV_cache(x)  # (B ,1 ,d_model) -> (B, 1, d_model)
        # 在kv缓存模式下， 当初次输入时，走全注意力，
        # 在kv缓存模式下， 当再次输入时，走分支（将新token并入缓存，然后全部过一遍（没有因果（输出只与最后一个token相关）））

        self.x = x  # 新增
        self.Q = x @ self.W_q  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        self.K = x @ self.W_k  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        self.V = x @ self.W_v  # (B, L, d_model) @ (d_model, d_v) -> (B, L, d_v)

        if kv_cache:
            self.update_cache(K_new=self.K, V_new=self.V)

        if self.causal:
            L, S = self.Q.shape[-2], self.K.shape[-2]
            causal_mask = np.tril(np.ones((L, S), dtype=bool))  # 下三角
            mask = causal_mask if mask is None else (mask & causal_mask)
        self.mask = mask
        out = self.scaled_dot_product_attention(self.Q, self.K, self.V, mask)  # -> (B, L, d_v)
        self.w_o_pre = out
        out = out @ self.W_o  # (B, L, d_v) @ (d_v, d_model) -> (B, L, d_model)
        return out

    def _KV_cache(self, x_new: np.ndarray) -> np.ndarray:
        """
            :param x_new:(B, 1, d_model)  , 初次输入时为(B, L, d_model), 再次输入多token时为(B, input_token_num, d_model)
            :return:out: (B, 1, d_model)
        """
        Q_new = x_new @ self.W_q  # (B, 1, d_model) @ (d_model, d_k) -> (B, 1, d_k) # 对于单一对话，获得了一个d_k维的查询向量
        K_new = x_new @ self.W_k  # (B, 1, d_model) @ (d_model, d_k) -> (B, 1, d_k) #  对于单一对话，获得了一个d_k维的键向量
        V_new = x_new @ self.W_v  # (B, 1, d_model) @ (d_model, d_v) -> (B, 1, d_v) #  对于单一对话，获得了一个d_v维的值向量

        self.update_cache(K_new=K_new, V_new=V_new)

        out = self.scaled_dot_product_attention(Q_new, self.K_cache, self.V_cache)  # (B, 1, d_v)
        out = out @ self.W_o  # (B, 1, d_v) @ (d_v, d_model) -> (B, 1, d_model) # 将相关性向量变为词向量，每个对话的最后一个词向量（注意力后的）
        return out

    def backward(self, gradient: np.ndarray, lr: float) -> np.ndarray:
        """
        反向传播

        :param gradient: (B, L, d_model)
        :param lr:
        :return:(B, L, d_model)
        """
        # out = out_pre @ W_o
        dW_o = np.einsum('blv,bld->vd', self.w_o_pre, gradient)
        g_out_pre = gradient @ self.W_o.T  # (B, L, d_model) @ (d_model, d_v) -> (B, L, d_v（d_model）)

        # out_pre = attn @ V
        g_v_pre = self.v_pre.swapaxes(-1, -2) @ g_out_pre  # (B, S（L）, L) @ (B, L, d_v) -> (B, S, d_v)
        dattn = g_out_pre @ self.V.swapaxes(-1, -2)  # (B, L, d_v) @ (B, d_v, S（L）) -> (B, L, S（L）)

        # softmax 反向
        dscores = self.v_pre * (dattn - (dattn * self.v_pre).sum(-1, keepdims=True))
        if self.mask is not None:
            dscores = dscores * self.mask
        dscores /= np.sqrt(self.d_k)  # ->(B, L, S)

        # scores = Q @ K^T
        g_q_pre = dscores @ self.K  # (B, L, d_k)
        g_k_pre = dscores.swapaxes(-1, -2) @ self.Q  # (B, S, L) @ (B, L, d_k) -> (B, S, d_k)

        # Q = x W_q, K = x W_k, V = x W_v
        dW_q = np.einsum('bld,blk->dk', self.x, g_q_pre)
        dW_k = np.einsum('bld,blk->dk', self.x, g_k_pre)
        dW_v = np.einsum('bld,blv->dv', self.x, g_v_pre)

        dx = g_q_pre @ self.W_q.T + g_k_pre @ self.W_k.T + g_v_pre @ self.W_v.T  # -> (B, L, d_model)

        self.W_q -= lr * dW_q
        self.W_k -= lr * dW_k
        self.W_v -= lr * dW_v
        self.W_o -= lr * dW_o
        return dx
