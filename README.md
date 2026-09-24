# numpy-mini-transformer

用纯 NumPy 从零实现 Transformer 与自回归语言模型。

## 特性
- 纯 NumPy，无 PyTorch/TensorFlow 依赖
- 手写反向传播（Attention / LayerNorm / Softmax 梯度）
- 支持 KV Cache 增量解码
- 支持 top-k / top-p / temperature 采样
- 兼容 tiktoken 的 cl100k_base 编码器

## 安装
pip install -r requirements.txt

## 运行
python mini_transformer.py

## 说明
纯玩具项目
不能训练真正的 LLM。