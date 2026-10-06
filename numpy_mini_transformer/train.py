# train.py
import numpy as np
import matplotlib.pyplot as plt
from .llm import LLM

def train(
        model: LLM,
        micro_batches: list[np.ndarray],
        lr=1e-3,
        steps: int = 1,
        ax: plt.axes = None,
        print_step: bool = True

) -> list:
    """
    一个简易的文本训练循环， 会对固定几个（B）文本拟合

    :param model: LLM 类
    :param micro_batches: [(B, L)]  np数组对象
    :param lr:  学习率
    :param steps: 训练次数
    :param ax: plt子图对象
    :return: 训练过程的损失列表
    """
    loss_ls = []
    for step in range(steps):
        loss = model.learn(micro_batches, lr=lr)
        loss_ls.append(loss)
        if ax is not None:
            ax.plot(loss_ls, 'r-', linewidth=2)
            plt.pause(0.001)
        if step % 20 == 0:
            if print_step:
                print(f"step {step:3d}/{steps}  loss {loss:.4f}")
            if loss < 0.5:
                break
            if step > 10000:
                lr = 1e-3
                if all(loss_ls[-5:]) < loss_ls[-1]:
                    break
    if ax is not None:
        ax.clear()
    return loss_ls
