import tiktoken
import numpy as np
from typing import Protocol
import codecs
from .model import Transformer
from .loss import cross_entropy_loss
from .sampling import sample

IM_END_TEMPLATE = "<|im_end|>\n"
IM_END_LEN = len(IM_END_TEMPLATE)  # 12


class Tokenizer(Protocol):
    """
    任何 tokenizer 都要满足这三个方法 + 一个属性。
    LLM 只依赖这个协议，不依赖具体实现。
    """

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def decode_single_token_bytes(self, token_id: int) -> bytes: ...
    # 返回 bytes 而非 str：因为单个 token 可能是 UTF-8 的中途片段，
    # 需要配合 codecs 增量解码器使用（你现在的流式输出就靠这个）


class TiktokenTokenizer:
    def __init__(self, name: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(name)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self._enc.decode_single_token_bytes(int(token_id))


class LLM:
    """
       简易自回归语言模型封装类。

       支持多轮聊天上下文管理、批量对话、采样生成、增量 UTF-8 解码、
       KV Cache、权重获取/设置以及单步训练。

       Parameters
       ----------
       system_content : str, optional
           系统提示词，默认为 "你是一个ai助手"。
       B : int, optional
           batch 数量，即同时维护几个对话，默认为 1。
           注意：当前实现中部分逻辑只有 B=1 时完全有效。
       d_model : int, optional
           模型隐藏维度，默认为 16。
       temperature : float, optional
           采样温度，默认为 0.8。
       seed : int, optional
           随机种子，默认为 0。
       n_layers : int, optional
           Transformer 层数，默认为 6。
       vocab_size : int or None, optional
           词表大小。若为 None，则使用 tiktoken 的 cl100k_base 词表大小。

       Attributes
       ----------
       messages : list[list[dict]]
           每个 batch 一个消息列表，每个消息为 {"role": str, "content": str}。
       enc : tiktoken.Encoding
           token 编码器。
       _decoders : list[codecs.IncrementalDecoder]
           每个 batch 一个增量 UTF-8 解码器。
       vocab_size : int
           实际使用的词表大小。
       transformer : Transformer
           Transformer 模型实例。
       """

    def __init__(self, system_content: str = "你是一个ai助手", B: int = 1, d_model: int = 16, temperature: int = 0.8,
                 seed=0, n_layers: int = 6,
                 vocab_size: int = None,
                 tokenizer: Tokenizer | None = None,
                 ):
        """
        初始化 LLM 实例。

        :param system_content: str, optional
            系统提示词，默认为 "你是一个ai助手"。
        :param B:int, optional
            batch 数量，默认为 1。该参数与学习无关。
        :param d_model: int, optional
            模型隐藏维度，默认为 16。
        :param temperature:float, optional
            采样温度，默认为 0.8。
        :param seed:int, optional
            随机种子，默认为 0。
        :param n_layers: int, optional
            TransformerBlock 层数，默认为 6。
        :param vocab_size:int or None, optional 词表大小。若为 None，则使用 tiktoken 的 cl100k_base 词表大小。
        """

        self.temperature = temperature
        self.B = B  # 该参数与学习无关
        self.system_content = system_content
        self.messages = []  # 格式化存储聊天  #当前模式下，只有B=1层是有效的
        self._messages_init_(B)

        self.tokenizer = tokenizer or TiktokenTokenizer("cl100k_base")  # 加载编码器, 默认 tiktoken（GPT-4 使用 cl100k_base）
        tok_vocab = self.tokenizer.vocab_size
        if vocab_size is None:
            self.vocab_size = tok_vocab
        else:
            if vocab_size != tok_vocab:
                raise ValueError(
                    f"vocab_size({vocab_size}) 与 tokenizer.vocab_size({tok_vocab}) 不一致。"
                    f"如果 tokenizer 是自定义的，请让二者匹配，否则 embedding 会越界。"
                )

        self._decoders = [
            codecs.getincrementaldecoder("utf-8")(errors="replace")
            for _ in range(B)
        ]  # 加载增量解码器
        self.transformer = Transformer(
            vocab_size=self.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            seed=seed,
        )

    def _messages_init_(self, B: int):
        """
          初始化消息列表。

        为每个 batch 创建一个新的对话，并在其中加入系统消息。
        :param B: batch 层数
        :return:
        """
        self.messages = []
        for _ in range(B):
            self.messages.append([{"role": "system", "content": self.system_content}])

    @staticmethod
    def _messages_to_ai_input_text(
            messages: [[{str: str}]]
    ) -> list[str]:
        """

        :param messages: [[{str:str}]]
        :return: (B, 1)  # str
        """
        ai_input_text_batch = []
        for content in messages:  # 提取单个对话的上下文 ->单个对话列表，元素为聊天气泡
            ai_input_text_content = ''  # 将对话列表转为格式化字符串
            for text in content:  # 从对话列表中提取聊天气泡 -> 字典
                role = text["role"]
                content = text['content']
                ai_input_text_content += f'<|im_start|>{role}\n{content}<|im_end|>\n'
            ai_input_text_batch.append(ai_input_text_content)
        return ai_input_text_batch  # (B, 1) str

    def _texts_to_token_id(self,
                           texts: list[str]
                           ):
        """

        :param texts:
        :return:
        """
        batch = []
        for text in texts:
            batch.append(self.tokenizer.encode(text))
        return batch

    def get_w(self) -> np.ndarray:
        """
        获取 transformer 权重

        :return: 权重（一维）
        """
        return self.transformer.get_w()

    def set_w(self, ws: np.ndarray):
        '''
          设置 Transformer 的权重。

        :param ws: (n, 1)一维数据集 list or np.array
        :return:
        '''
        if type(ws) == list:
            ws = np.asarray(ws)
        self.transformer.set_w(ws)

    def clear_content(self):
        """
          清空对话内容、增量解码器以及 KV Cache。

        调用后会重置 messages 为初始系统消息，并清空 Transformer 的 KV 缓存。
        :return:
        """
        self._messages_init_(self.B)  # 清空上下文
        self._decoders = [
            codecs.getincrementaldecoder("utf-8")(errors="replace")
            for _ in range(self.B)
        ]
        self.transformer.clear_kv_cache()  # 清空缓存

    def read_content(self, messages: [[dict]]):
        """
         读取外部消息作为上下文。

        会先调用 clear_content() 清空当前内容，然后设置新的 messages。
        :param messages:
        :return:
        """
        # 合法性检测
        if type(messages[0]) is dict:
            m = []  # 将上下文广播到每个batch层
            for _ in range(self.B):
                m.append(messages)
            messages = m
        self.clear_content()
        self.messages = messages

    def __call__(self,
                 text: str or [str] = '你好',
                 print_m: bool = False,
                 kv_cache: bool = True,
                 max_new_tokens: int = 1000
                 ) -> list[str]:
        """
        执行聊天生成。

        :param text:用户输入 str or [str]
        :param print_m: 是否流式打印生成内容，默认为 False。
        :param kv_cache:    是否使用 KV Cache，默认为 True。
                             为 True 时每步只输入新 token 的向量；
                            为 False 时拼接完整序列。
        :param max_new_tokens: 单次回复最大输出token
        :return: 模型输出（仅第一层block）
        """
        "====上下文管理===="
        if type(text) == str:
            self.messages[0].append({"role": "user", "content": text})
            self.messages[0].append({"role": "assistant", "content": ''})
            "====输入处理===="
            ai_in_texts = self._messages_to_ai_input_text(self.messages)  # (B， 1)
            ai_in_texts[0] = ai_in_texts[0][:-IM_END_LEN]  # 去掉自动添加的结束符  (B, 1)  # 后续不用加回来，因为ai应当输出回来
            ai_in_token_id = self._texts_to_token_id(ai_in_texts)  # (B, L)
            ai_in_token_id = np.array([ai_in_token_id[0]])  # 转为B=1的模式  (1, L)
        elif type(text) == list:
            if self.B == len(text):
                for i in range(self.B):
                    self.messages[i].append({"role": "user", "content": text[i]})
                    self.messages[i].append({"role": "assistant", "content": ''})
                ai_in_texts = self._messages_to_ai_input_text(self.messages)  # (B， 1)
                for i in range(self.B):
                    ai_in_texts[i] = ai_in_texts[i][:-IM_END_LEN]  # 去掉自动添加的结束符  (B, 1)  # 后续不用加回来，因为ai应当输出回来
                ai_in_token_id = np.array(self._texts_to_token_id(ai_in_texts))  # (B, L)
            else:
                raise ValueError("B != len(text)")
        else:
            raise ValueError(f"text type is not list or str.{type(text)}")

        out_tokens_num = 0
        input_ids = ai_in_token_id  # （B, L)
        while True:
            "input_ids 将保证transformer上下文的跟新。 messages作为记录，用户接口和当前聊天气泡外的上下文存储"
            logits = self.transformer.forward(input_ids=input_ids,
                                              kv_cache=kv_cache)  # (B, L, d_model) -> (B, L, vocab_size)

            logits_last = logits[:, -1, :]  # (B, vocab_size) 只取最后一个token
            # 采样 概率分布 -> token_id
            next_ids = sample(logits_last, temperature=self.temperature, top_k=10, top_p=0.9)  # (B, 1)

            # 记录与输出 token_id -> text   跟新 input_id 与 self.messages
            if kv_cache:
                input_ids = next_ids[:, None]  # (B, 1)
            else:
                input_ids = np.concatenate([input_ids, next_ids[:, None]], axis=1)

            # 转文本打印并保存
            for i, tid in enumerate(next_ids):
                # 1) token id -> 原始字节（不做 UTF-8 合法性检查）
                raw_bytes = self.tokenizer.decode_single_token_bytes(int(tid))
                # 2) 喂给该 batch 的增量解码器，拿回“能完整解码的部分”
                next_token = self._decoders[i].decode(raw_bytes)  # 可能是 ''
                # 3) 更新 messages（只追加已完整的文本）
                if next_token:
                    self.messages[i][-1]["content"] += next_token
                if print_m and i == 0 and next_token:
                    print(next_token, end="", flush=True)

            out_tokens_num += 1
            if out_tokens_num >= 300:
                break
        if print_m:
            print('')
        return [mg[-1]['content'] for mg in self.messages]

    def learn(self, token_ids: np.ndarray, lr: float = 1e-3) -> float:
        """
        执行单步训练。
        :param token_ids: (B, L+1)  np数组  该数据的B层数量决定训练时的B层数
        :param lr: float 学习率
        :return: loss 损失
        """
        input_ids = token_ids[:, :-1]  # (B, L)
        target_ids = token_ids[:, 1:]  # (B, L)
        # 前向传播(记录必要的部分)
        logits = self.transformer.forward(input_ids=input_ids)  # -> (B, L, vocab_size)
        # 计算损失和梯度
        loss, gradient = cross_entropy_loss(logits=logits, targets=target_ids)  # -> float  (B, L, vocab_size)
        # 反向传播
        self.transformer.backward(input_ids=input_ids, gradient=gradient, lr=lr)
        return loss
