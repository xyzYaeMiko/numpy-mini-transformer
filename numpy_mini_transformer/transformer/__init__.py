from .model import Transformer
from .layers import (
    InputEmbedding, TransformerBlock, LMHead,
    SingleHeadAttention, LayerNorm, FFN,
)
from .ops import clip, softmax


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
