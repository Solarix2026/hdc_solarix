"""
Solarix — Qwen 语义嵌入模块

使用 Qwen2.5-0.5B-Instruct 将文本编码为浮点语义向量，
后续可量化为二进制超向量接入 HDCCore 记忆系统。
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 默认使用轻量级 Qwen2.5-0.5B-Instruct（CPU 友好，~1 GB）
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def _load_model() -> tuple:
    """懒加载模型和分词器（首次调用时加载，后续复用）。"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.float32,
        ).to("cpu")
        _model.eval()
    return _tokenizer, _model


def get_embedding(text: str) -> np.ndarray:
    """
    将文本编码为语义向量（最后一层隐藏状态的均值池化）。

    Parameters
    ----------
    text : str
        输入文本。

    Returns
    -------
    np.ndarray
        一维浮点向量，维度等于模型隐藏层大小（Qwen2.5-0.5B 为 896）。
    """
    tokenizer, model = _load_model()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 取最后一层隐藏状态，均值池化（沿 token 维度）
    last_hidden: torch.Tensor = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    embedding = last_hidden.mean(dim=1).squeeze(0)          # (hidden_dim,)

    return embedding.numpy()


if __name__ == "__main__":
    text = "Solarix is a low-power hyperdimensional computing memory system"
    print(f"输入文本: {text}")
    print("正在加载模型...")

    vec = get_embedding(text)

    print(f"输出向量形状: {vec.shape}")
    print(f"前 5 个值: {vec[:5]}")
