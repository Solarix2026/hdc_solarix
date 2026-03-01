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
                source TEXT DEFAULT 'manual',
                solution TEXT DEFAULT '',
                is_consolidated INTEGER DEFAULT 0
            )
        """)
        # 尝试升级老表结构
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN solution TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN is_consolidated INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
        self.conn.commit()

    def add_memory(self, *, hv: np.ndarray, context: str, window_title: str = "manual", timestamp: float = None, solution: str = "") -> int:
        """
        将文本及其对应的位压缩超向量存入数据库。
        """
        import datetime
        import time
        if timestamp is None:
            timestamp = time.time()
            
        dt = datetime.datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
        # 将 numpy 数组转为原生 bytes 存入 SQLite BLOB
        hv_bytes = hv.tobytes()
        
        print(f"[Vault] 正在存入记忆，时间戳: {timestamp_str}")
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (timestamp, original_text, hypervector, source, solution) VALUES (?, ?, ?, ?, ?)",
            (timestamp_str, context, hv_bytes, window_title, solution)
        )
        self.conn.commit()
        
        return cursor.lastrowid

    def retrieve_all(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, timestamp, original_text, hypervector, source, solution, is_consolidated FROM memories")
        rows = cursor.fetchall()
        import datetime

        results = []
        for row in rows:
            mem_id, ts_str, text, hv_bytes, source, solution, is_cons = row
            try:
                dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                timestamp = dt.timestamp()
            except:
                timestamp = 0.0
            
            # 将 BLOB 数据逆向转换回 numpy uint8 数组格式（保证与输入一致，即压缩包形式）
            hv_array = np.frombuffer(hv_bytes, dtype=np.uint8)
            results.append((mem_id, timestamp, text, hv_array, source, solution, is_cons))
            
        return results

    def retrieve_by_similarity(
        self, hv: np.ndarray, top_k: int = 3, threshold: float = 0.7, hdc_core=None
    ):
        """
        使用 HDCCore 在记忆库中进行相似度检索（KNN 匹配）。
        """
        # 为了兼容，如果没传 hdc_core 参数，自动实例化一个HDCCore来进行计算
        if hdc_core is None:
            from hdc_core import HDCCore
            hdc_core = HDCCore(dimension=10000)
            
        all_memories = self.retrieve_all()
        scored_results = []

        for mem_id, ts, text, hv_array, source, solution, is_cons in all_memories:
            if is_cons == 1:
                # 已折叠的可以被检索，但是暂时不强制过滤
                pass
                
            # 使用 HDCCore 原生自带的相似度算法计算得分
            score = hdc_core.similarity(hv, hv_array)
            if score >= threshold:
                scored_results.append({
                    "similarity_score": score,
                    "timestamp": ts,
                    "context": text,
                    "solution": solution if solution else "",
                    "solution_path": "", # 预留字段
                    "id": mem_id
                })

        # 根据相似度分数降序排列选出 Top K
        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:top_k]

    def get_unconsolidated_memories(self):
        """获取所有未折叠的记忆"""
        all_memories = self.retrieve_all()
        results = []
        for mem_id, ts, text, hv_array, source, solution, is_cons in all_memories:
            if is_cons == 0:
                results.append({
                    "id": mem_id,
                    "hv": hv_array,
                    "context": text,
                    "timestamp": ts
                })
        return results

    def add_consolidated_memory(self, merged_hv: np.ndarray, context_summary: str, original_ids: list, timestamp: float):
        import json
        sol_json = json.dumps({"consolidated_from": original_ids})
        self.add_memory(
            hv=merged_hv,
            context=context_summary,
            window_title="DBSCAN Consolidation",
            timestamp=timestamp,
            solution=sol_json
        )
        
    def mark_as_consolidated(self, ids: list):
        if not ids: return
        cursor = self.conn.cursor()
        placeholders = ','.join(['?']*len(ids))
        cursor.execute(f"UPDATE memories SET is_consolidated = 1 WHERE id IN ({placeholders})", ids)
        self.conn.commit()

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
    test_db = "temp_solarix_test_memory.db"
    
    # 因为这是每次都跑的测试
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except:
            pass

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
        inserted_id = vault.add_memory(hv=hv, context=t, window_title="test_script")
        print(f"  [写入成功] ID={inserted_id}, Text '{t[:30]}...'")

    print(f"\n验证: 库中现有一共 {len(vault.retrieve_all())} 条记忆。")

    print("\n[3] 模拟检索查询...")
    query = "What is Solarix? Tell me about its hyperdimensional memory."
    print(f"  Query: '{query}'")
    query_hv = mapper.map(get_embedding(query))

    print("\n[4] 执行相似度检索...")
    results = vault.retrieve_by_similarity(query_hv, top_k=3, threshold=0.0)

    for rank, res in enumerate(results, 1):
        print(f"  Top {rank} (分数: {res['similarity_score']:.4f}): [ID:{res['id']}] {res['context']}")

    print("\n" + "=" * 60)
    # Check if 'Solarix' is in the top match
    if "Solarix" in results[0]['context']:
        print("验证完成 ✓：最相似结果准确！")
    else:
        print("验证失败 ✗：检索结果不如预期！")
    print("=" * 60)

    # Optionally close connection.
    if vault.conn:
        vault.conn.close()
