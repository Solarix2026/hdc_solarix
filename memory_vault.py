# C++ Future: This layer stays in Python. C++ will read the same SQLite file via sqlite3.h.
"""
Solarix — 海马体持久化层 (Memory Vault)
使用 SQLite 实现轻量级的超维记忆存取，支持通过 HDCCore 在库中进行相似度检索。
"""

from __future__ import annotations

import sqlite3
import numpy as np


class MemoryVault:
    """超维记忆穹顶：管理基于 SQLite 的数字海马体。"""

    def __init__(self, db_path: str = "solarix_memory.db") -> None:
        """
        初始化并连接 SQLite 数据库，创建必要的表结构。
        
        Parameters
        ----------
        db_path : str
            SQLite 数据库文件的路径，默认为 'solarix_memory.db'。
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        """初始化 memories 表结构。"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_text TEXT NOT NULL,
                hypervector BLOB NOT NULL,
                source TEXT DEFAULT 'manual'
            )
        """)
        self.conn.commit()

    def add_memory(self, text: str, hypervector: np.ndarray, source: str = "manual", timestamp_unix: float = None) -> int:
        """
        将文本及其对应的位压缩超向量存入数据库。

        Parameters
        ----------
        text : str
            原始记忆文本。
        hypervector : np.ndarray
            对应的超向量（位压缩格式，dtype=uint8）。
        source : str
            记忆来源标签。
        timestamp_unix: float
            外部触发该记忆的精确Unix时间戳。如果为空，则使用当前时间。

        Returns
        -------
        int
            新插入记录的数据库自增 ID。
        """
        import datetime
        import time
        if timestamp_unix is None:
            timestamp_unix = time.time()
            
        dt = datetime.datetime.fromtimestamp(timestamp_unix)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
        # 将 numpy 数组转为原生 bytes 存入 SQLite BLOB
        hv_bytes = hypervector.tobytes()
        
        print(f"[Vault] 正在存入记忆，时间戳: {timestamp_str}")
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (timestamp, original_text, hypervector, source) VALUES (?, ?, ?, ?)",
            (timestamp_str, text, hv_bytes, source)
        )
        self.conn.commit()
        
        return cursor.lastrowid

    def retrieve_all(self) -> list[tuple[int, str, np.ndarray, str]]:
        """
        从数据库全量读取并反序列化所有记忆。

        Returns
        -------
        list[tuple[int, str, np.ndarray, str]]
            返回结构：[(id, text, hypervector, source), ...]
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, original_text, hypervector, source FROM memories")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            mem_id, text, hv_bytes, source = row
            # 将 BLOB 数据逆向转换回 numpy uint8 数组格式（保证与输入一致，即压缩包形式）
            hv_array = np.frombuffer(hv_bytes, dtype=np.uint8)
            results.append((mem_id, text, hv_array, source))
            
        return results

    def retrieve_by_similarity(
        self, query_hv: np.ndarray, hdc_core, top_k: int = 5
    ) -> list[tuple[float, int, str, str]]:
        """
        使用 HDCCore 在记忆库中进行相似度检索（KNN 匹配）。

        Parameters
        ----------
        query_hv : np.ndarray
            输入查询的超向量（位压缩格式）。
        hdc_core : HDCCore
            用于调用 .similarity() 方法的核心引擎实例。
        top_k : int
            最多返回的类似记忆数量。

        Returns
        -------
        list[tuple[float, int, str, str]]
            按相似度分数降序排列：[(相似度, id, text, source), ...]
        """
        all_memories = self.retrieve_all()
        scored_results = []

        for mem_id, text, hv_array, source in all_memories:
            # 使用 HDCCore 原生自带的相似度算法计算得分
            score = hdc_core.similarity(query_hv, hv_array)
            scored_results.append((score, mem_id, text, source))

        # 根据相似度分数降序排列选出 Top K
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:top_k]

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


# ======================================================================
# 测试验证
# ======================================================================
if __name__ == "__main__":
    from hdc_core import HDCCore
    from lsh_mapper import LSHMapper
    from qwen_embedder import get_embedding
    import os

    print("=" * 60)
    print("Solarix MemoryVault — 数据库存取与检索验证")
    print("=" * 60)

    # 为了避免之前的测试污染，确保一个崭新的测试库
    test_db = "solarix_memory.db"
    
    # 因为这是每次都跑的测试
    if os.path.exists(test_db):
        os.remove(test_db)

    print("[1] 初始化核心组件...")
    vault = MemoryVault(db_path=test_db)
    hdc = HDCCore(dimension=10000)
    mapper = LSHMapper(input_dim=896, output_dim=10000, seed=42)

    print("\n[2] 写入测试数据 (Solarix, Qwen, Python)...")
    texts_to_store = [
        "Solarix uses hyperdimensional computing for low-power memory.",
        "Qwen 0.5B is a small but powerful language model.",
        "Python is great for rapid prototyping of AI systems."
    ]

    for t in texts_to_store:
        hv = mapper.map(get_embedding(t))
        inserted_id = vault.add_memory(t, hv, source="test_script")
        print(f"  [写入成功] ID={inserted_id}, Text '{t[:30]}...'")

    print(f"\n验证: 库中现有一共 {len(vault.retrieve_all())} 条记忆。")

    print("\n[3] 模拟检索查询...")
    query = "What is Solarix? Tell me about its hyperdimensional memory."
    print(f"  Query: '{query}'")
    query_hv = mapper.map(get_embedding(query))

    print("\n[4] 执行相似度检索...")
    results = vault.retrieve_by_similarity(query_hv, hdc, top_k=3)

    for rank, (score, mem_id, text, source) in enumerate(results, 1):
        print(f"  Top {rank} (分数: {score:.4f}): [ID:{mem_id}] {text}")

    print("\n" + "=" * 60)
    # Check if 'Solarix' is in the top match
    if "Solarix" in results[0][2]:
        print("验证完成 ✓：最相似结果准确！")
    else:
        print("验证失败 ✗：检索结果不如预期！")
    print("=" * 60)

    # Optionally close connection.
    if vault.conn:
        vault.conn.close()
