"""
Solarix — 低功耗神经符号记忆系统 PoC
HDCCore: 超维计算（Hyperdimensional Computing）核心引擎

**重点重构**：位压缩原生架构
所有超向量在类内外均以位压缩包（uint8, shape=byte_size_）流转，
彻底消除未压缩的 (10000,) 数组。1 个字节表示 8 个物理维度。
"""

from __future__ import annotations

import numpy as np


class HDCCore:
    """超维计算原生位压缩核心类，为 C++ 移植铺路。"""

    def __init__(self, dimension: int = 10000) -> None:
        """
        初始化超向量空间。

        Parameters
        ----------
        dimension : int
            逻辑超向量维度，默认 10000。
        """
        self.dimension_: int = dimension
        # 物理字节数（向上取整，以处理非 8 倍数的维度）
        self.byte_size_: int = (dimension + 7) // 8

    # ------------------------------------------------------------------
    # 生成随机向量（直接压缩）
    # ------------------------------------------------------------------
    def generate_random_vector(self) -> np.ndarray:
        """
        生成一个随机二进制超向量，并立刻打进压缩包。

        Returns
        -------
        np.ndarray
            形状 (byte_size_,)，dtype 为 np.uint8。
        """
        # 第一步：生成逻辑随机向量
        raw_vec = np.random.randint(0, 2, size=self.dimension_, dtype=np.uint8)
        # 第二步（强制）：立即调用 np.packbits 压缩为 (byte_size_,)
        packed_vec = np.packbits(raw_vec)
        return packed_vec

    # ------------------------------------------------------------------
    # 绑定操作（极速 XOR）
    # ------------------------------------------------------------------
    def bind(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        绑定两个压缩包：逐字节异或。

        Parameters
        ----------
        vector1, vector2 : np.ndarray
            压缩后的二进制超向量。

        Returns
        -------
        np.ndarray
            绑定结果，形状 (byte_size_,)。
        """
        if vector1.shape != (self.byte_size_,) or vector2.shape != (self.byte_size_,):
            raise ValueError(f"输入形状需要为 ({self.byte_size_,})")

        # C++: 直接对 std::vector<uint8_t> 进行逐字节 XOR，1条指令处理8个维度。
        return np.bitwise_xor(vector1, vector2)

    # ------------------------------------------------------------------
    # 打包操作
    # ------------------------------------------------------------------
    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        打包多个超向量压缩包：带平局判断的多数表决。
        需先解包所有向量实现整数加法，再重新打包。

        Parameters
        ----------
        vectors : list[np.ndarray]
            包含压缩包的列表。

        Returns
        -------
        np.ndarray
            多数表决后的新超向量压缩包。
        """
        if len(vectors) < 2:
            raise ValueError("打包操作至少需要 2 个超向量")

        # 必须解包以进行多数求和计算
        unpacked_list = []
        for v in vectors:
            unpacked = np.unpackbits(v)[:self.dimension_]
            unpacked_list.append(unpacked)

        stacked = np.stack(unpacked_list)
        ones_count = stacked.sum(axis=0)
        
        n = len(vectors)
        half = n / 2.0

        result = np.empty(self.dimension_, dtype=np.uint8)
        result[ones_count > half] = 1
        result[ones_count < half] = 0

        # 平局随机打破
        tie_mask = ones_count == half
        tie_count = int(tie_mask.sum())
        if tie_count > 0:
            result[tie_mask] = np.random.randint(0, 2, size=tie_count, dtype=np.uint8)

        # 重新打包后返回
        return np.packbits(result)

    # ------------------------------------------------------------------
    # 相似度计算（引入 popcount 思想）
    # ------------------------------------------------------------------
    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个压缩包的相似度（0.0 ~ 1.0）。

        Parameters
        ----------
        vector1, vector2 : np.ndarray
            压缩后的二进制超向量。

        Returns
        -------
        float
            相似度分数。
        """
        # 第一步：逐字节 XOR 找出对应位不同的集合（1 代表位不同）
        xor_result = np.bitwise_xor(vector1, vector2)
        
        # 第二步：计算汉明距离（统计 xor_result 中 1 的个数）
        # 方式一（为 C++ 扫雷铺路）：先 unpack 展开，然后 sum
        # （注：在 C++ 中，这一步会用 __builtin_popcount，不需要展开）
        diff_bits = np.unpackbits(xor_result)[:self.dimension_]
        hamming_distance = int(np.sum(diff_bits))

        # 第三步：将距离转换为相似度分数
        return 1.0 - (hamming_distance / self.dimension_)


# ======================================================================
# 测试 / 演示（第三部分回归测试）
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Solarix HDCCore [位压缩原生版本] — 核心逻辑回归测试")
    print("=" * 60)

    hdc = HDCCore(dimension=10000)
    print(f"\n[初始化] 逻辑维度: {hdc.dimension_}, 物理字节: {hdc.byte_size_}")

    # 1. 生成向量形状验证
    A = hdc.generate_random_vector()
    B = hdc.generate_random_vector()
    print(f"\n[生成检查] 向量 A 的形状: {A.shape} | 预期: ({hdc.byte_size_},)")
    assert A.shape == (hdc.byte_size_,), "形状不符预期的压缩包设计"

    # 2. 绑定计算
    C = hdc.bind(A, B)
    print(f"[绑定操作] 向量 C(A_xor_B) 的形状: {C.shape}")

    # 3. 相似度数学一致性验证
    sim_A_A = hdc.similarity(A, A)
    sim_A_B = hdc.similarity(A, B)
    
    print(f"\n[相似度] sim(A, A) = {sim_A_A:.4f}  (期望 = 1.0000)")
    print(f"[相似度] sim(A, B) = {sim_A_B:.4f}  (期望 ≈ 0.5000)")
    
    # Bundle 测试
    bundled = hdc.bundle([A, C])
    sim_A_bundle = hdc.similarity(A, bundled)
    print(f"[合并操作] sim(A, bundle(A, C)) = {sim_A_bundle:.4f}  (期望 ≈ 0.7500)")

    print("\n" + "=" * 60)
    print("回归测试完成 ✓")
    print("=" * 60)
