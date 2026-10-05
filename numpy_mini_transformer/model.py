# model.py
import numpy as np
from .block import TransformerBlock
from .layers import LayerNorm
from .loss import clip

"""
token 序列
  ↓
词嵌入 Embedding
  ↓
位置编码 Positional Encoding
  ↓
重复 N 次：
    TransformerBlock(
        LayerNorm → Masked Self-Attention → 残差
        LayerNorm → FFN → 残差
    )
  ↓
最终 LayerNorm
  ↓
输出线性层 W_vocab
  ↓
Softmax 
  ↓
下一个 token 概率

"""


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

    def get_w(self) -> np.ndarray:
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

    def backward(self, token_ids: np.ndarray, gradient: np.ndarray, lr: float) -> np.ndarray:
        """
        词表反向：把每个 token 位置的梯度累加回词表对应行

        :param token_ids: (B, L) int
        :param gradient:  (B, L, d_model)
        """
        dvocab = np.zeros_like(self.vocab)
        np.add.at(dvocab, token_ids.ravel(), gradient.reshape(-1, self.d_model))
        self.vocab -= lr * dvocab


class LMHead:
    """预言模型输出头"""

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

    def backward(self, gradient: np.ndarray, lr: float):
        """
        反向传播

        dL/dW = x^T @ dL/dy  # 用于跟新权重
        dL/db = sum(dL/dy)  # 用于更新偏置
        dL/dx = dL/dy @ W^T  # 沿权重继续吧梯度传给上一个
         (B, L, vocab_size)
        :param gradient: (B, L, vocab_size)
        :param lr:
        :return: (B, L, d_model)
        """
        # self.x.T @ gradient  # 权重的变化  (B, d_model, L) @ (B, L, vocab_size) -> (d_model, vocab_size)
        dw = np.einsum('bld,blv->dv', self.w_pre, gradient)
        db = gradient.sum(axis=tuple(range(gradient.ndim - 1)))
        # 将梯度沿权重传递到上一个
        dx = gradient @ self.W.T  # (B, L, vocab_size) @ (vocab_size, d_model) -> (B, L, d_model)

        # 跟新
        self.W -= lr * dw
        self.b -= lr * db
        return dx


class Transformer:
    """
    一个仅使用解码器（Decoder-only）架构的 Transformer 语言模型。

    由四部分组成：
        1. InputEmbedding  : 词嵌入层，把 token id 映射为 d_model 维向量，并注入位置编码
        2. TransformerBlock: n_layers 个堆叠的 Transformer 块（含多头自注意力 + FFN）
        3. LayerNorm       : 输出前的层归一化
        4. LMHead          : 语言模型头，把 d_model 投影到 vocab_size，输出下一个 token 的 logits

    参数经 get_w / set_w 以“一维 np.ndarray”形式整体读写，便于参数序列化、
    进化算法（如 CMA-ES）等需要把所有权重压平成单个向量的场景。

    KV Cache：
        推理时若开启 kv_cache=True，每个 TransformerBlock 会缓存历史的 K/V，
        使增量解码（每次只输入 1 个新 token）无需重复计算前缀，
        显著降低自回归生成的计算量。缓存可通过 get_kv_cache / set_kv_cache
        在 block 之间整体存取，通过 clear_kv_cache 清空。

    典型用法：
        model = Transformer(vocab_size=..., d_model=..., n_layers=6)
        logits = model.forward(input_ids)                    # 训练 / 全序列前向
        logits = model.forward(next_id, kv_cache=True)       # 增量解码（L=1）
        loss.backward(); model.backward(input_ids, grad, lr) # 反向更新
    """
    def __init__(self, vocab_size: int, d_model: int, seed=0, n_layers: int = 6,
                 ):

        self.d_model = d_model  # 词向量维度

        self.embedding = InputEmbedding(vocab_size, d_model, seed=seed)
        self.blocks = [TransformerBlock(d_model, seed=seed + i) for i in range(n_layers)]
        self.ln = LayerNorm(d_model=d_model)
        self.lmh = LMHead(d_model, vocab_size)

    def get_w(self) -> np.ndarray:
        """
        设置权重

        :return:np 一维np向量
        """
        return np.concatenate(
            [self.embedding.get_w()]
            + [b.get_w() for b in self.blocks]
            + [self.ln.get_w()]
            + [self.lmh.get_w()]
        )

    def set_w(self, ws: np.ndarray):
        """
        获取权重

        :param ws 一维
        :return:
        """
        idx = 0
        # 1. vocab
        n = self.embedding.vocab.size
        self.embedding.vocab = ws[idx:idx + n].reshape(self.embedding.vocab.shape)
        idx += n

        # 2. transformer blocks + lm head
        for p in self.blocks + [self.ln] + [self.lmh]:
            n = p.get_w().size
            p.set_w(ws[idx:idx + n])
            idx += n
        ...

    def clear_kv_cache(self):
        for block in self.blocks:
            block.att.clear_KV_cache()

    def get_kv_cache(self) -> list:
        """
        获取KV缓存

        :return [kv_cache1, kv_cache2, ...]  (block_number, 2,  B, L, d_v（d_k）)
        """
        kv_caches = []
        for block in self.blocks:
            kv_caches.append(block.att.get__kv_cache())

        shapes = {kv.shape for kv in kv_caches}
        if len(shapes) > 1:
            raise ValueError(f"各 block KV cache 形状不一致: {shapes}")

        return kv_caches


    def set_kv_cache(self, kv_cache: list):
        """
        ::param kv_cache: [kv_cache1, kv_cache2, ...]  (block_number, 2,  B, L, d_v（d_k）)
        """
        i = 0
        for block in self.blocks:
            block.att.set__kv_cache(kv_cache[i])
            i += 1


    def _cache_offset(self):
        """当前 KV cache 里的 token 数；无缓存返回 0。只在这一处判断。"""
        first = self.blocks[0].att.K_cache
        return 0 if first is None else first.shape[1]

    def forward(self,
                input_ids: np.ndarray,
                kv_cache: bool = False
                ) -> np.ndarray:
        """
        前向传播

        :param input_ids: (B, L) int
        :param  KV_cache: bool 增量时 L=1
        :return:logits:(B, L, vocab_size) token概率分布向量矩阵
        """

        # 词嵌入 和 注入位置编码
        offset = self._cache_offset() if kv_cache else 0  # 获取当前缓存数量
        x = self.embedding(input_ids, offset=offset)  # (B, L) -> (B, L, d_model)

        # block层
        for block in self.blocks:
            x = block.forward(x, kv_cache=kv_cache)  # (B, L, d_model) -> (B, L, d_model)
            # KV: (B, 1, d_model) -> (B, 1, d_model)

        # ln层
        x = self.ln.forward(x)  # (B, L, d_model) -> (B, L, d_model)
        # KV: (B, 1, d_model) -> (B, 1, d_model)

        # LMHead层
        logits = self.lmh.forward(x)  # (B, L, vocab_size) 每个token的下一个token的概率分布
        # KV: (B, 1, vocab_size) 最后一个token的下一个token的概率分布
        return logits  # (B, L, vocab_size)

    def backward(self, input_ids: np.ndarray, gradient: np.ndarray, lr: float):
        """
        反向传播

        input_ids 只用于跟新词向量矩阵。反向传播的核心仅 gradient
        :param input_ids: 前向传播的token id, (B, L)
        :param gradient: 梯度 (B, L, vocab_size)
        :param lr: 学习率
        """
        grad = clip(gradient, 1.0)
        grad = self.lmh.backward(grad, lr)  # -> (B, L, d_model)
        grad = clip(grad, 1.0)
        grad = self.ln.backward(grad, lr)
        # 重复n个TransformerBlock中回传（逆序：反向传播需与前向顺序相反）
        for block in reversed(self.blocks):
            grad = block.backward(grad, lr)  # (B, L, d_model) -> (B, L, d_model)
        # 跟新词向量矩阵
        # 词嵌入表梯度：把 gradient 累加回对应 id 的行
        grad = clip(grad, 1.0)
        self.embedding.backward(input_ids, grad, lr)  # 更新词表
