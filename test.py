"""numpy-mini-transformer 功能自检脚本（也可以当 demo 跑）。

三种运行方式都支持:

    python test.py            # 跑全部 7 区域，训练 100 步（约 1 分钟）
    python test.py 20         # 只训练 20 步
    pytest test.py -q         # PyCharm 绿色运行：只跑 test_smoke，训练 5 步（约 3 秒）

说明:
    第 7 节「训练演示」是最慢的部分，约 0.6 秒/步（vocab=100277, d_model=32）。
"""

import os
import sys
import time

import numpy as np

from numpy_mini_transformer import LLM, train

# ---------------- 可调参数 ----------------
D_MODEL = 32
N_LAYERS = 2
TRAIN_LR = 1e-2
DEFAULT_STEPS = 100  # 命令行不传时的训练步数
SMOKE_STEPS = 1000      # pytest / PyCharm 里跑 test_smoke 时的步数

_results = []  # [(名称, 是否通过)]


# ---------------- 初始化小工具 ----------------
def _setup_stdout():
    """Windows 控制台默认 GBK，模型输出是任意 UTF-8 片段（可能含 U+FFFD），
    直接 print 会抛 UnicodeEncodeError。让不可编码字符用 '?' 代替。"""
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def _steps_from_argv(default=DEFAULT_STEPS):
    """只接受纯数字参数。

    这样 `python test.py 20` 有效；而 pytest 传进来的文件路径（sys.argv[1]）
    不会被误当成步数。"""
    for arg in sys.argv[1:]:
        if arg.isdigit():
            return int(arg)
    return default


# ---------------- 断言与输出小工具 ----------------
def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def check(name, ok):
    _results.append((name, ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")


def new_model(**kwargs):
    kwargs.setdefault("d_model", D_MODEL)
    kwargs.setdefault("n_layers", N_LAYERS)
    return LLM(**kwargs)


def timed(fn):
    """返回 (结果, 耗时秒)"""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def clip_text(text, limit=70):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def show_messages(model):
    for i, dialog in enumerate(model.messages):
        print(f"    batch[{i}]:")
        for msg in dialog:
            print(f"      {msg.get('role'):<9} {msg.get('content')!r}")


# ---------------- 各项检查 ----------------
def demo_kv_cache():
    section("1. 全注意力 vs KV Cache")
    text = "时间流逝吧，你是多么残酷。时间停止吧，你是多么美丽。"
    model = new_model()

    model.clear_content()
    out_full, t_full = timed(lambda: model(text=text, kv_cache=False))

    model.clear_content()
    out_cached, t_cached = timed(lambda: model(text=text, kv_cache=True))

    print(f"    全注意力 {t_full:5.2f}s   KV Cache {t_cached:5.2f}s")
    print(f"    输出     {clip_text(out_full[0])!r}")
    check("kv_cache=True / False 输出一致", out_full == out_cached)


def demo_stream():
    section("2. 流式输出 (print_m=True)")
    model = new_model()
    print("    >>> model(text='你好', print_m=True, kv_cache=False)")
    print("    ", end="")
    model(text="你好", print_m=True, kv_cache=False)
    print()
    content = model.messages[0][-1]["content"]
    check(f"流式输出非空 (len={len(content)})", len(content) > 0)


def demo_batch():
    section("3. 批量对话 (B=2)")
    model = new_model(B=2)
    outputs = model(text=["小狗在吃饭", "猫在睡觉喵"])
    for i, out in enumerate(outputs):
        print(f"    batch[{i}] {clip_text(out)!r}")
    check(f"返回 {len(outputs)} 条结果", len(outputs) == 2)


def demo_clear_content():
    section("4. 清空上下文 (clear_content)")
    model = new_model()
    model(text="你好")
    before = len(model.messages[0])
    model.clear_content()
    after = len(model.messages[0])
    show_messages(model)
    check(f"回到只有 system 消息 ({before} -> {after})", after == 1 and before > 1)


def demo_weights():
    section("5. 权重读写 (get_w / set_w)")
    source = new_model()
    weights = source.get_w()

    target = new_model()
    target.set_w(weights)

    print(f"    get_w() 形状 {weights.shape}")
    check("set_w 后权重与来源完全一致", np.array_equal(target.get_w(), source.get_w()))


def demo_read_content():
    section("6. 读取外部上下文 (read_content)")
    model = new_model()
    content = [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "人工智能输出"},
    ]
    model.read_content(content)
    show_messages(model)
    check("messages 被替换为传入内容", model.messages == [content])

    out, _ = timed(lambda: model(text="你好"))
    check("替换后仍可正常生成", len(out) == 1 and isinstance(out[0], str))



def demo_train(steps=DEFAULT_STEPS):
    section(f"7. 训练演示（单文本拟合） ({steps} 步, lr={TRAIN_LR})")
    text = "自然语言处理很有趣。深度学习改变了世界。"
    model = new_model()

    token_ids = model.tokenizer.encode(text)
    micro = np.array([token_ids, token_ids, token_ids, token_ids])  # (B=4, L)
    print(f"    样本 {len(token_ids)} 个 token，4 行，{steps} 步")

    # matplotlib 是可选的：没装、或用着非交互后端（Agg/Svg 等）都跳过画图。
    # 否则 train() 里每步一次 plt.pause 会刷满 "FigureCanvasAgg is non-interactive" 警告。
    ax = None
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        backend = matplotlib.get_backend().lower()
        non_interactive = {"agg", "pdf", "svg", "ps", "cairo", "template"}
        if backend in non_interactive:
            print(f"    (matplotlib 后端 {backend!r} 不可交互，跳过绘图)")
        else:
            plt.ion()
            _, ax = plt.subplots(1, 1)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
    except Exception as exc:  # pragma: no cover
        print(f"    (跳过绘图: {exc})")

    t0 = time.perf_counter()
    loss_list = train(
        model=model, micro_batches=[micro], lr=TRAIN_LR, steps=steps, ax=ax
    )
    elapsed = time.perf_counter() - t0

    if loss_list:
        print(f"    loss {loss_list[0]:.3f} -> {loss_list[-1]:.3f}，耗时 {elapsed:.1f}s")
        check("训练后 loss 下降", loss_list[-1] < loss_list[0])
    else:
        check("训练产生了 loss 记录", False)

    # 训练后再对比一次两种推理路径
    model.clear_content()
    out_cached = model(text="自然", kv_cache=True)
    model.clear_content()
    out_full = model(text="自然", kv_cache=False)

    after = model.test_run(text="自然语言")
    check("训练后 kv_cache 两条路径仍一致", out_cached == out_full)
    print("训练后生成测试，输入：'自然'", after)
    check("训练后能复现训练文本", "自然语言处理很有趣。深度学习改变了世界。" in after)


# ---------------- 入口 ----------------
def run_all(steps=DEFAULT_STEPS):
    """按顺序跑完 7 节，返回退出码（0=全部通过）"""
    _results.clear()
    print(f"numpy-mini-transformer 自检 | d_model={D_MODEL} n_layers={N_LAYERS} "
          f"训练步数={steps}")

    demo_kv_cache()
    demo_stream()
    demo_batch()
    demo_clear_content()
    demo_weights()
    demo_read_content()
    demo_train(steps)

    section("汇总")
    failed = [name for name, ok in _results if not ok]
    for name, ok in _results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"\n  {len(_results) - len(failed)}/{len(_results)} 项通过")
    if failed:
        print(f"  失败项: {failed}")
    return 0 if not failed else 1


def main(steps=None):
    _setup_stdout()
    return run_all(_steps_from_argv() if steps is None else steps)


def test_smoke():
    """pytest / PyCharm 绿色运行入口：快速冒烟，训练 SMOKE_STEPS 步。

    环境变量 MINI_TRAIN_STEPS 可以覆盖步数。"""
    steps = int(os.environ.get("MINI_TRAIN_STEPS", SMOKE_STEPS))
    assert main(steps=steps) == 0, "自检存在失败项"


if __name__ == "__main__":
    sys.exit(main())
