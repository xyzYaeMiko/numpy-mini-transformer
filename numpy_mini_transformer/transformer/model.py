import numpy as np
from .layers import TransformerBlock, LayerNorm, InputEmbedding, LMHead
from .ops import clip


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
        获取权重

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
        设置权重

        :param ws 一维
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
        设置KV缓存

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

    def backward(self, input_ids: np.ndarray, gradient: np.ndarray):
        """
        反向传播梯度并叠加保存

        input_ids 只用于跟新词向量矩阵。反向传播的核心仅 gradient
        :param input_ids: 前向传播的token id, (B, L)
        :param gradient: 梯度 (B, L, vocab_size)
        """
        grad = clip(gradient, 1.0)
        grad = self.lmh.backward(grad)  # -> (B, L, d_model)
        grad = clip(grad, 1.0)
        grad = self.ln.backward(grad)
        # 重复n个TransformerBlock中回传（逆序：反向传播需与前向顺序相反）
        for block in reversed(self.blocks):
            grad = block.backward(grad)  # (B, L, d_model) -> (B, L, d_model)
        # 跟新词向量矩阵
        # 词嵌入表梯度：把 gradient 累加回对应 id 的行
        grad = clip(grad, 1.0)
        self.embedding.backward(input_ids, grad)  # 更新词表

    def update_w(self, lr:float):
        """
        跟新权重

        :param lr: 学习率
        """
        self.lmh.update_w(lr)
        for block in reversed(self.blocks):
            block.update_w(lr)
        self.ln.update_w(lr)
        self.embedding.update_w(lr)

    def zero_grad(self):
        """
        梯度缓存清零

        """
        self.lmh.zero_grad()
        for block in reversed(self.blocks):
            block.zero_grad()
        self.ln.zero_grad()
        self.embedding.zero_grad()
