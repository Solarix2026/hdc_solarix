"""
Solarix — Stage 3.1 压力测试与相似度拥挤诊断

验证 100 条记忆规模下检索性能的退化情况，
引入 "Margin" 指标诊断是否出现 "相似度拥挤" 问题。
"""

from __future__ import annotations

from hdc_core import HDCCore
from lsh_mapper import LSHMapper
from hdc_coder import HDCCoder


def main():
    print("=" * 60)
    print("Solarix Stage 3.1: 100条记忆压力测试 & 诊断")
    print("=" * 60)

    # 1. 初始化核心组件
    print("\n[1] 初始化核心组件...")
    coder = HDCCoder()
    hdc = HDCCore(dimension=10000)

    # 2. 构建记忆库（Encoding）
    # 核心记忆
    texts = [
        "Solarix uses hyperdimensional computing for low-power memory",
        "Qwen 0.5B is a small but powerful language model",
        "Python is great for rapid prototyping of AI systems"
    ]
    
    # 补充干扰记忆，凑满 100 条
    for i in range(4, 101):
        texts.append(f"Random fact number {i}: The quick brown fox jumps over the lazy dog to test memory crowdedness.")

    memory_items = []

    print(f"\n[2] 正在批量存入 {len(texts)} 条记忆（这可能需要一些时间）...")
    for i, text in enumerate(texts, 1):
        hypervector = coder.encode(text)
        memory_items.append((text, hypervector))
        
        # 进度提示
        if i == 3 or i % 20 == 0:
            print(f"  已成功编码 {i}/100 条记忆...")

    # 3. 检索逻辑增强
    query_text = "What is Solarix's memory technology?"
    print(f"\n[3] 接收查询: '{query_text}'...")

    query_hv = coder.encode(query_text)

    # 计算相似度并加入列表
    results = []
    print("  计算 100 条记忆的相似度...")
    for text, hv in memory_items:
        sim = hdc.similarity(query_hv, hv)
        results.append((sim, text))

    # 按相似度降序排列
    results.sort(key=lambda x: x[0], reverse=True)

    # 4. 诊断输出
    print("\n[Top 5 检索匹配]")
    for rank in range(5):
        score, text = results[rank]
        # 对超长文本做下截断展示
        display_text = text if len(text) <= 60 else text[:57] + "..."
        print(f"  Top {rank+1} (匹配度: {score:.4f}) | {display_text}")

    # 计算 Margin (差值)
    top1_score = results[0][0]
    top2_score = results[1][0]
    margin = top1_score - top2_score

    print("\n" + "-" * 60)
    print(f"[诊断] 最高分: {top1_score:.4f}, 第二高分: {top2_score:.4f}, 差值 (Margin): {margin:.4f}")
    if margin > 0.02:
        print("✅ 系统状态健康，区分度良好。")
    else:
        print("⚠️ 警告：区分度过低，建议触发维度升级预案。")
    print("-" * 60)


if __name__ == "__main__":
    main()
