"""
Solarix — 低功耗神经符号记忆系统 PoC
HDCCore: 超维计算（Hyperdimensional Computing）核心引擎

本模块用纯 NumPy 实现 HDC 的基本操作：
  - 生成二进制超向量（Hypervector）作为记忆痕迹
  - 绑定（Binding）：逐元素 XOR，编码"关系"
  - 打包（Bundling）：多数表决，叠加"集合记忆"
  - 汉明距离相似度（Hamming Similarity）：衡量记忆检索质量
  - 压缩 / 解压（Pack / Unpack）：节省存储开销
"""

from __future__ import annotations

import numpy as np


class HDCCore:
    """超维计算核心类，管理超向量空间的生成、绑定、打包与相似度计算。"""

    def __init__(self, dimension: int = 10000) -> None:
        """
        初始化超向量空间。

        Parameters
        ----------
        dimension : int
            超向量维度，默认 10000。维度越高，记忆容量和抗干扰能力越强。
        """
        self.dimension: int = dimension

    # ------------------------------------------------------------------
    # 超向量生成
    # ------------------------------------------------------------------
    def generate_random_vector(self) -> np.ndarray:
        """
        生成一个随机二进制超向量（记忆痕迹）。

        Returns
        -------
        np.ndarray
            形状 (dimension,)，元素为 0 或 1，各占 50% 概率。
        """
        return np.random.randint(0, 2, size=self.dimension, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 绑定（Binding）—— 编码"关系"
    # ------------------------------------------------------------------
    def bind(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        绑定两个超向量：逐元素异或（XOR）。

        绑定后的向量与任一输入向量近似正交，可用于编码
        "A 与 B 的关联"这类关系型记忆。

        Parameters
        ----------
        vector1, vector2 : np.ndarray
            待绑定的二进制超向量。

        Returns
        -------
        np.ndarray
            绑定结果，形状 (dimension,)。

        Raises
        ------
        ValueError
            若任一向量维度与初始化维度不一致。
        """
        if vector1.shape != (self.dimension,):
            raise ValueError(
                f"vector1 维度 {vector1.shape} 与期望 ({self.dimension},) 不一致"
            )
        if vector2.shape != (self.dimension,):
            raise ValueError(
                f"vector2 维度 {vector2.shape} 与期望 ({self.dimension},) 不一致"
            )
        return np.bitwise_xor(vector1, vector2)

    # ------------------------------------------------------------------
    # 打包（Bundling）—— 叠加"集合记忆"
    # ------------------------------------------------------------------
    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        打包多个超向量：带平局随机处理的多数表决。

        打包后的向量与每个输入向量都保持较高相似度，可用于
        表示"集合记忆"（如一组相关概念的叠加）。

        规则（逐位）：
          - 1 的个数 > 0 的个数 → 取 1
          - 0 的个数 > 1 的个数 → 取 0
          - 平局 → 随机取 0 或 1（避免信息丢失）

        Parameters
        ----------
        vectors : list[np.ndarray]
            至少包含 2 个二进制超向量。

        Returns
        -------
        np.ndarray
            打包结果，形状 (dimension,)。

        Raises
        ------
        ValueError
            若向量数量 < 2 或任一向量维度不一致。
        """
        if len(vectors) < 2:
            raise ValueError("打包操作至少需要 2 个超向量")
        for i, v in enumerate(vectors):
            if v.shape != (self.dimension,):
                raise ValueError(
                    f"vectors[{i}] 维度 {v.shape} 与期望 ({self.dimension},) 不一致"
                )

        # 按列求和：sum_arr[j] = 所有向量第 j 位的 1 的个数
        stacked = np.stack(vectors)                       # (n, dimension)
        ones_count = stacked.sum(axis=0)                  # (dimension,)
        n = len(vectors)
        half = n / 2.0

        result = np.empty(self.dimension, dtype=np.uint8)
        result[ones_count > half] = 1
        result[ones_count < half] = 0

        # 平局位随机打破
        tie_mask = ones_count == half
        tie_count = int(tie_mask.sum())
        if tie_count > 0:
            result[tie_mask] = np.random.randint(0, 2, size=tie_count, dtype=np.uint8)

        return result

    # ------------------------------------------------------------------
    # 压缩 / 解压 —— 节省存储
    # ------------------------------------------------------------------
    def pack(self, vector: np.ndarray) -> np.ndarray:
        """
        将二进制超向量压缩为字节数组（8 bit → 1 byte）。

        Parameters
        ----------
        vector : np.ndarray
            形状 (dimension,) 的二进制超向量。

        Returns
        -------
        np.ndarray
            压缩后的 uint8 字节数组。
        """
        return np.packbits(vector)

    def unpack(self, packed_vector: np.ndarray) -> np.ndarray:
        """
        将压缩字节数组还原为原始维度的二进制超向量。

        会自动裁剪 packbits 尾部填充，保证维度与 self.dimension 一致。

        Parameters
        ----------
        packed_vector : np.ndarray
            由 pack() 返回的压缩数组。

        Returns
        -------
        np.ndarray
            形状 (dimension,) 的二进制超向量。
        """
        unpacked = np.unpackbits(packed_vector)
        return unpacked[: self.dimension]

    # ------------------------------------------------------------------
    # 相似度计算（汉明距离）
    # ------------------------------------------------------------------
    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个超向量的相似度（基于汉明距离）。

        公式: similarity = 1 - hamming_distance / dimension
          - 1.0 表示完全相同
          - 0.0 表示完全不同
          - ≈0.5 表示近似正交（不相关）

        Parameters
        ----------
        vector1, vector2 : np.ndarray
            待比较的二进制超向量。

        Returns
        -------
        float
            相似度分数，范围 [0, 1]。
        """
        hamming_distance = int(np.sum(vector1 != vector2))
        return 1.0 - hamming_distance / self.dimension


# ======================================================================
# 测试 / 演示
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Solarix HDCCore — 超维计算核心逻辑验证")
    print("=" * 60)

    hdc = HDCCore(dimension=10000)
    print(f"\n[初始化] 超向量维度: {hdc.dimension}")

    # 1. 生成 3 个随机超向量
    A = hdc.generate_random_vector()
    B = hdc.generate_random_vector()
    C = hdc.generate_random_vector()
    print(f"[生成] A, B, C 三个随机超向量 (shape={A.shape})")

    # 2. 绑定 A 和 B → AB（编码 A-B 关系）
    AB = hdc.bind(A, B)
    print(f"[绑定] AB = A ⊕ B  (shape={AB.shape})")

    # 3. 打包 AB 和 C → ABC_bundle（集合记忆）
    ABC_bundle = hdc.bundle([AB, C])
    print(f"[打包] ABC_bundle = bundle(AB, C)  (shape={ABC_bundle.shape})")

    # 4. 相似度验证
    sim_AB_bundle = hdc.similarity(AB, ABC_bundle)
    sim_A_bundle = hdc.similarity(A, ABC_bundle)
    sim_A_B = hdc.similarity(A, B)

    print(f"\n[相似度] AB vs ABC_bundle : {sim_AB_bundle:.4f}  (期望 ≈0.75，打包成员)")
    print(f"[相似度] A  vs ABC_bundle : {sim_A_bundle:.4f}  (期望 ≈0.50，非直接成员)")
    print(f"[相似度] A  vs B          : {sim_A_B:.4f}  (期望 ≈0.50，随机正交)")

    # 5. 压缩 / 解压验证
    packed_AB = hdc.pack(AB)
    unpacked_AB = hdc.unpack(packed_AB)
    is_identical = np.array_equal(AB, unpacked_AB)

    print(f"\n[压缩] AB 原始大小: {AB.nbytes} bytes → 压缩后: {packed_AB.nbytes} bytes")
    print(f"[解压] 还原后与原向量一致: {is_identical}")

    print("\n" + "=" * 60)
    print("验证完成 ✓" if is_identical else "验证失败 ✗")
    print("=" * 60)
