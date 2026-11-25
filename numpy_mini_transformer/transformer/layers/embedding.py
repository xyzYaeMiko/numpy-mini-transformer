import numpy as np


class InputEmbedding:
    """
    token id -> 带位置编码的词向量。
    同时负责反向时把梯度累加回词表。
    """

    def __init__(self, vocab_size: int, d_model: int, seed=0):
        """
        生成词向量

        :param vocab_size: 词表尺寸
        :param d_model: 词向量维度
        :param seed:
        """
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.vocab = rng.normal(0, s, (vocab_size, d_model))  # (V, d_model)
        self.d_model = d_model

        self.dvocab = np.zeros_like(self.vocab)

    def positional_encoding(self, L: int, offset: int = 0):
        """
        生成位置编码

        pe_2i = sin(pos/(10000^(2i/d_model)))   i ∈ [0, d_model/2], pos ∈ [0, L]
            # 对于单一位置的位置编码为 该函数在[0, d_modle]区间内的2的倍数处的值构成的向量
        pe_(2i-1) = cos(pos/(10000^(2i/d_model)))   i ∈ [0, d_model/2], pos ∈ [0, L]

        :param L: 序列长度(token数)
        :param offset: 起始位置偏移，等于已缓存 token 数；
                   预填充时 =0，增量解码时 = 已缓存长度
        :return: (L, d_model)
        """
        pos = np.arange(offset, offset + L)[:, None]  # (L, 1)  从 offset 开始
        i = np.arange(self.d_model)[None, :]  # (1, d_model)
        angle = pos / np.power(10000, (2 * (i // 2)) / self.d_model)  # (L, d_model)
        # (L, 1) / (1, d_model) -> (L, d_model) 在二维，将pos（1个元素）广播到i的每一个二维元素中，
        # 在一为，将i的元素（一个元素（一个列表））广播到pos的每一个一维元素（多个元素（列表））中
        pe = np.zeros((L, self.d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        return pe

    def get_w(self):
        return self.vocab.ravel()

    def set_w(self, ws: np.ndarray):
        self.vocab = ws.reshape(self.vocab.shape)

    def __call__(self, token_ids: np.ndarray, offset=0) -> np.ndarray:
        """
        添加位置（前向传播）

        :param token_ids: (B, L) int
        :param offset:    位置编码起始下标，KV cache 增量解码时用
        :return:          (B, L, d_model)
        """
        x = self.vocab[token_ids]  # (B, L) -> (B, L, d_model)
        pe = self.positional_encoding(token_ids.shape[1], offset)  # 生成位置编码
        return x + pe

    def backward(self, token_ids: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        词表反向：把每个 token 位置的梯度累加回词表对应行

        :param token_ids: (B, L) int
        :param gradient:  (B, L, d_model)
        """
        np.add.at(self.dvocab, token_ids.ravel(), gradient.reshape(-1, self.d_model))
    def update_w(self, lr: float):
        """
       跟新词向量

       :param lr: 学习率
        """
        self.vocab -= lr * self.dvocab

    def zero_grad(self):
        """
        梯度缓存清零

        """
        self.dvocab[...] = 0
