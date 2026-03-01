# Solarix HDCCore — C++ 移植方案设计文档 (Plan)

为了实现 Solarix 系统在低功耗硬件上的终极性能，HDCCore 将从 Python/NumPy 移植到纯 C++（C++17 或以上标准）。本设计文档描述了移植的关键映射与架构思路。

## 1. 数据结构映射表

在“位压缩原生架构”中，所有操作都围绕紧凑的字节级容器展开。

| 概念 | Python / NumPy 实现 | C++ 类型映射 |
| :--- | :--- | :--- |
| **超向量维度 (逻辑)** | `self.dimension_` (int) | `size_t dimension_` |
| **超向量尺寸 (物理)** | `self.byte_size_` (int) | `size_t byte_size_` |
| **超向量容器** | `np.ndarray` (`dtype=np.uint8`, shape=`(byte_size_,)`) | `std::vector<uint8_t>` (或基于栈的 `std::array`，如果维度固定) |
| **打包 / 位压缩** | `np.packbits()` | 自定义位运算（位掩码结合移位），或一次性构造 |
| **解包 / 位展开** | `np.unpackbits()` | 自定义位展开运算（仅在 Bundle 等必要时使用） |

## 2. C++ 类接口草稿 (Header File Pseudo-code)

```cpp
#ifndef SOLARIX_HDC_CORE_HPP
#define SOLARIX_HDC_CORE_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>

namespace solarix {

class HDCCore {
public:
    // 构造函数
    explicit HDCCore(size_t dimension = 10000);

    // 1. 生成随机初始超向量（直接返回位压缩格式）
    std::vector<uint8_t> generateRandomVector() const;

    // 2. 绑定操作：逐字节异或运算
    std::vector<uint8_t> bind(const std::vector<uint8_t>& v1, 
                              const std::vector<uint8_t>& v2) const;

    // 3. 打包操作：带平局干预的多数表决
    std::vector<uint8_t> bundle(const std::vector<std::vector<uint8_t>>& vectors) const;

    // 4. 相似度：汉明距离换算
    float similarity(const std::vector<uint8_t>& v1, 
                     const std::vector<uint8_t>& v2) const;

private:
    size_t dimension_; // 逻辑维度，例如 10000
    size_t byte_size_; // 物理字节数，(dimension_ + 7) / 8
};

} // namespace solarix
#endif // SOLARIX_HDC_CORE_HPP
```

## 3. 性能关键点备注

**性能核心：避免解包带来的内存分配与访问开销。**

### 相似度计算（汉明距离）
- **Python (原型阶段)**：由于缺乏原生的基于数组字节的 popcount 映射原语，为了代码清晰可靠，我们在 `similarity()` 中使用了 `np.unpackbits(xor_result)` 展开并求和。
- **C++ (生产阶段)**：*为了极致性能，我们不需要也不允许在求相似度时展开数组。* 在对 `v1` 和 `v2` 进行运算后，我们可以直接使用现代编译器提供的内建函数（如 `__builtin_popcount`、`__builtin_popcountll` 或 C++20 的 `<bit>` 库 `std::popcount`）以底层寄存器指令甚至 SIMD 指令（如 AVX2 的 `_mm256_popcnt_epi8`）来迅速统计异或结果中 `1` 的个数。如果 `1` 指令能处理 8 字节或 32 字节，这部分性能将被极其放大。

### 绑定计算 (Binding)
- 无论是在 Python 的 `bitwise_xor` 底层，还是未来的 C++ 中的 `std::transform` 或手动 `for` 循环，针对位压缩数据的 XOR 都自带高度并行化的效果（1 个循环处理 8 个特征维度或更是 64 个），在资源受限的环境中这是核心优势。
