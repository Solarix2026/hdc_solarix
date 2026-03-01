"""
Solarix — Stage 2 端到端验证

完整流程验证：
自然语言 (文本) -> Qwen Embedder (896维浮点) -> LSHMapper (10000维二进制) -> HDCCore (记忆匹配与检索)
"""

from __future__ import annotations

import numpy as np

from hdc_core import HDCCore
from lsh_mapper import LSHMapper
from qwen_embedder import get_embedding


def main():
    print("=" * 60)
    print("Solarix Stage 2: 自然语言 -> 记忆 -> 检索验证")
    print("=" * 60)

    # 1. 初始化三个组件
    # 注意: Qwen2.5-0.5B-Instruct 模型最后一层的维度为 896
    print("\n[1] 初始化核心组件...")
    mapper = LSHMapper(input_dim=896, output_dim=10000, seed=42)
    hdc = HDCCore(dimension=10000)
    # Embedder 组件直接作为函数调用 (get_embedding)

    # 2. 构建记忆库（Encoding）
    texts = [
        "Solarix uses hyperdimensional computing for low-power memory",
        "Qwen 0.5B is a small but powerful language model",
        "Python is great for rapid prototyping of AI systems"
    ]

    memory_items = []

    print("\n[2] 正在存入记忆...")
    for i, text in enumerate(texts, 1):
        # Embed: 提取文本语义向量 (896维浮点)
        float_vec = get_embedding(text)
        
        # Map: 映射入高维汉明空间 (10000维二进制)
        hypervector = mapper.map(float_vec)
        
        # 保存至记忆库
        memory_items.append((text, hypervector))
        print(f"  - 记忆 {i} 已存入 (超向量形状: {hypervector.shape}): '{text}'")

    # 3. 测试检索（Retrieval）
    query_text = "What is Solarix's memory technology?"
    print(f"\n[3] 接收查询: '{query_text}'...")

    # 用同样的管线处理查询文本
    query_float_vec = get_embedding(query_text)
    query_hv = mapper.map(query_float_vec)

    # 计算与每条记忆的相似度
    best_match_text = None
    best_similarity = -1.0

    print("\n[4] 正在比对相似度...")
    for text, hv in memory_items:
        sim = hdc.similarity(query_hv, hv)
        print(f"  与 '{text[:15]}...' 的相似度: {sim:.4f}")
        
        if sim > best_similarity:
            best_similarity = sim
            best_match_text = text

    # Output 最终结果
    print("-" * 60)
    print(f"最匹配的记忆: {best_match_text}, 相似度: {best_similarity:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
