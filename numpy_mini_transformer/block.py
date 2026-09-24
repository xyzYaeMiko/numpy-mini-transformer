import numpy as np
from .attention import SingleHeadAttention
from .layers import LayerNorm, FFN
from .loss import clip


class TransformerBlock:
    """LayerNorm → 多头自注意力 → 残差 → LayerNorm → FFN → 残差"""

    def __init__(self, d_model: int, seed=0):
        self.att = SingleHeadAttention(d_model, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, seed=seed)

    def get_w(self):
        return np.concatenate([
            self.att.get_w(),
            self.ln1.get_w(),
            self.ln2.get_w(),
            self.ffn.get_w(),
        ])

    def set_w(self, ws: np.ndarray):
        idx = 0
        for m in [self.att, self.ln1, self.ln2, self.ffn]:
            n = m.get_w().size
            m.set_w(ws[idx:idx + n])
            idx += n

    def forward(self, x: np.ndarray, kv_cache: bool = False):
        """
        x -> ln ->att+x -> ln ->ffn+x ->out
        └------------↗   └---------↗
        :param kv_cache: 将参数传给注意力
        :param x: 词嵌入矩阵（包含位置编码的词向量矩阵） (B, L, d_model)
        :return:(B, L, d_model)
        """

        x = x + self.att.forward(self.ln1(x), kv_cache=kv_cache)  # (B, L, d_model) + (B, L, d_model) -> (B, L, d_model)
        # 其他头计算
        # 合并多头结果

        x = x + self.ffn.forward(self.ln2(x))  # (B, L, d_model) + (B, L, d_model) -> (B, L, d_model)
        return x  # (B, L, d_model)

    def backward(self, gradient: np.ndarray, lr: float):
        """

        :param gradient: (B, L, d_model)
        :param lr:
        :return:
        """
        g_ffn_in = clip(gradient, 1.0)  # 或 clip(gradient, 5.0)
        g_ffn_out = self.ffn.backward(g_ffn_in, lr)  # ->(B, L, d_model)
        g_ln2_in = clip(g_ffn_out, 1.0)
        g_ln2_out = self.ln2.backward(g_ln2_in, lr)  # -> (B, L, d_model)
        g_att_in = clip(g_ln2_out, 1.0)

        g_att_in = gradient + g_att_in  # 残差链接

        g_att_out = self.att.backward(g_att_in, lr)  # -> (B, L, d_model)
        g_ln1_in = clip(g_att_out, 1.0)  # 或 clip(gradient, 5.0)
        g_ln1_out = self.ln1.backward(g_ln1_in, lr)  # -> (B, L, d_model)

        gradient = g_ln1_out + g_att_in  # 残差链接
        return gradient  # -> (B, L, d_model)
