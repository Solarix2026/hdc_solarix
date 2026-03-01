import numpy as np
from sentence_transformers import SentenceTransformer

class HDCCoder:
    def __init__(self):
        # 替换为轻量级模型，避免高CPU占用
        self.embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.embed_model.max_seq_length = 512  # 适配窗口标题/输入缓冲区长度
        self.embed_cache = {}  # Embedding缓存（避免重复计算，但窗口切换必重新计算）
        self.hdc_dim = 10000  # HDC超向量维度（896维特征扩展）

        # Precompute projection matrix mapped from embedding dim to HDC dim
        # bge-small-zh-v1.5 output dim is 512
        np.random.seed(42)
        self.proj_matrix = np.random.randn(512, self.hdc_dim).astype(np.float32)

    def _embedding_to_hdc(self, embedding: np.ndarray) -> np.ndarray:
        # LSH mapping from embedding to binary vector
        projected = np.dot(embedding, self.proj_matrix)
        return (projected > 0).astype(np.uint8)

    def encode(self, text: str) -> np.ndarray:
        """轻量级特征提取→HDC超向量转换，降低CPU占用"""
        # 缓存已编码的文本（避免重复计算，但窗口切换必重新计算）
        if text in self.embed_cache:
            return self.embed_cache[text]
        # 特征提取（CPU友好）
        embedding = self.embed_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        # 转换为HDC超向量（扩展到10000维）
        hv = self._embedding_to_hdc(embedding)
        # 打包为uint8型（节省存储空间）
        hv_packed = np.packbits(hv.astype(np.uint8))
        self.embed_cache[text] = hv_packed
        return hv_packed
