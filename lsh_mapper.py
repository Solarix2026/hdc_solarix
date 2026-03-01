"""
Solarix — LSH 随机投影映射模块

通过局部敏感哈希（Locality-Sensitive Hashing）风格的随机投影，
将低维浮点语义向量映射为高维二进制超向量，接入 HDCCore 记忆系统。

注意：当前实现输出 {0, 1} 二进制格式，与 HDCCore 的 XOR 绑定兼容。
HDC 领域另一种常见格式为 {-1, 1}（与乘法绑定兼容），后续可根据
下游运算需求通过 `2 * vector - 1` 轻松转换。
"""

from __future__ import annotations

import numpy as np


class LSHMapper:
    """局部敏感哈希映射器：浮点向量 → 二进制超向量。

    利用 Johnson-Lindenstrauss 引理的思想，通过随机高斯投影将
    低维语义空间中的距离关系近似保留到高维汉明空间中。

    Attributes:
        input_dim: 输入浮点向量的维度。
        output_dim: 输出二进制超向量的维度。
        projection_matrix: 形状 (input_dim, output_dim) 的随机投影矩阵。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10000,
        seed: int | None = None,
    ) -> None:
        """初始化随机投影矩阵。

        Args:
            input_dim: 输入浮点向量的维度（如 Qwen2.5-0.5B 的 896）。
            output_dim: 输出超向量的维度，默认 10000。
            seed: 随机种子，用于复现投影矩阵。None 表示不固定种子。
        """
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        rng = np.random.default_rng(seed)
        self.projection_matrix: np.ndarray = rng.standard_normal(
            (input_dim, output_dim)
        ).astype(np.float32)

    def map(self, float_vector: np.ndarray) -> np.ndarray:
        """将浮点向量映射为二进制超向量。

        步骤：矩阵乘法 → 符号二值化（>0 → 1, ≤0 → 0）。

        Args:
            float_vector: 形状 (input_dim,) 的一维浮点向量。

        Returns:
            位压缩后的数组，形状为 (output_dim // 8,)，dtype 为 np.uint8。

        Raises:
            ValueError: 输入向量维度与 input_dim 不匹配时抛出。
        """
        if float_vector.shape[0] != self.input_dim:
            raise ValueError(
                f"输入向量维度 {float_vector.shape[0]} 与期望 {self.input_dim} 不一致"
            )

        projected = float_vector @ self.projection_matrix  # (output_dim,)
        binary = (projected > 0).astype(np.uint8)
        packed_binary = np.packbits(binary)
        return packed_binary


if __name__ == "__main__":
    print("=" * 60)
    print("Solarix LSHMapper — 随机投影映射验证")
    print("=" * 60)

    # 模拟输入：512 维随机浮点向量
    np.random.seed(42)
    fake_embedding = np.random.randn(512).astype(np.float32)

    mapper = LSHMapper(input_dim=512, output_dim=10000, seed=999)
    print(f"\n[初始化] 投影矩阵形状: {mapper.projection_matrix.shape}")
    print(f"[初始化] 投影矩阵 dtype: {mapper.projection_matrix.dtype}")

    hv = mapper.map(fake_embedding)

    # 为了计算 1 的占比，需要先将其解包
    unpacked_hv = np.unpackbits(hv)[:mapper.output_dim]
    ones_ratio = unpacked_hv.sum() / unpacked_hv.size * 100

    print(f"\n[映射结果] 超向量(压缩包)形状: {hv.shape}")
    print(f"[映射结果] 前 10 个字节值: {hv[:10]}")
    print(f"[映射结果] 1 的占比: {ones_ratio:.2f}%")
    print(f"[映射结果] dtype: {hv.dtype}")

    print("\n" + "=" * 60)
    print("验证完成 ✓")
    print("=" * 60)
