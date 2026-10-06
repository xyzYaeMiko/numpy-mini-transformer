from numpy_mini_transformer import *
import time
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    text = '自然语言处理很有趣。深度学习改变了世界。'

    print("=======tset========")
    model = LLM(d_model=32, n_layers=2)  # 使用初始词库
    print("\n输出测试（KV_cache=F）")
    print(">>> out1 = model(text=TEXT, kv_cache=False)")
    t1 = time.time()
    out1 = model(text=text, kv_cache=False)
    t2 = time.time()
    print(model.messages)
    model.clear_content()
    print("\n输出测试（KV_cache=T）")
    print(">>> out2 = model(text=TEXT, kv_cache=True)")
    t3 = time.time()
    out2 = model(text=text, kv_cache=True)
    t4 = time.time()
    print(model.messages)


    print('\n', '使用全注意力节点用时=', t2 - t1, ' | ', '使用KV缓存用时=', t4 - t3, "输出是否相同:", out1 == out2)
    print('\n流式输出测试')
    print(">>> out = model(text='你好', print_m=True, kv_cache=False)")
    model.clear_content()
    out = model(text='你好', print_m=True, kv_cache=False)

    print("\n批处理测试")
    t = ['小狗在吃饭', '猫在睡觉喵']
    print(">>> model = LLM(d_model=32, n_layers=2, B=2)\n>>>model(text=['小狗在吃饭', '猫在睡觉喵'])")
    model = LLM(d_model=32, n_layers=2, B=2)
    model(text=t)
    print(model.messages)
    print("\n清空上下文测试")
    print("清空前：", model.messages)
    print(">>> model.clear_content()")
    model.clear_content()
    print('清空后', model.messages)
    m1 = LLM(d_model=32, n_layers=2)
    w = m1.get_w()
    print('\n', 'get_w()测试', w.shape)
    m1.set_w(w)
    print('set_w()测试', m1.get_w().shape)

    print("\n读取上下文测试")
    model(text="'读取上下文测试'用户输入")
    content = [{'role': 'system', 'content': '系统提示词'}, {'role': 'user', 'content': '用户输入'},
               {'role': 'assistant', 'content': '人工智能输出'}]
    print('要读取的上下文:', content)
    print('当前上下文:', model.messages)
    model.read_content(content)
    print('>>>model.read_content(content)')
    print('读取后的上下文:', model.messages)
    print("读取后测试运行：\n>>>model(text=‘你好’)")
    out = model(text="你好")
    print("输出:", out)
    print('\n学习测试(单文本拟合)')
    print(""">>>enc = tiktoken.get_encoding("cl100k_base")\n>>>text = '自然语言处理很有趣。深度学习改变了世界。'\n>>>raw_ids = enc.encode(text)\n>>>B = np.array([raw_ids, raw_ids, raw_ids, raw_ids])\n>>>loss_ls = train(model=model, batch_id=B, steps=1000, ax=ax, lr=1e-3)""")
    model.clear_content()
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    line_train, = ax.plot([], [], 'r-', label='train')
    ax.legend()

    enc = tiktoken.get_encoding("cl100k_base")
    raw_ids = enc.encode(text)
    B = np.array([raw_ids, raw_ids, raw_ids, raw_ids])
    loss_ls = train(model=model, micro_batches=[B, B, B], steps=10000, ax=ax, lr=1e-2)
    print(model.test_run(text='自然'))


