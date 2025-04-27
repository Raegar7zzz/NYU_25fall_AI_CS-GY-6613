"""
retrieve.py
-----------
• 输入问题 (query)
• 检索 Qdrant 中最相似的字幕片段
• 返回字幕内容、视频名、时间戳
"""

import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ------------------ 配置 ------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6334
COLLECTION_NAME = "video_chunks"

# ------------------ 连接 Qdrant ------------------

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)

# ------------------ 加载文本嵌入模型 ------------------

text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ 检索函数 ------------------

def retrieve(query_text, top_k=5):
    # 1. 查询文本向量化
    query_vec = text_model.encode(query_text, normalize_embeddings=True)

    # 2. 因为之前向量是 文本384 + 图像512 拼接的，所以这里只用文本部分（384维）
    # 后面补512维0向量对齐成896维
    text_padding = np.zeros(512, dtype=np.float32)
    query_vec_full = np.concatenate([query_vec, text_padding])

    # 3. 检索
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec_full.tolist(),
        limit=top_k,
    )

    # 4. 打印结果
    for idx, hit in enumerate(hits):
        payload = hit.payload
        print(f"\n🔎 Top-{idx+1}:")
        print(f"Score: {hit.score:.4f}")
        print(f"Subtitle: {payload.get('subtitle', 'N/A')}")
        print(f"Video: {payload.get('video_name', 'N/A')}")
        print(f"Timestamp (s): {payload.get('timestamp', 'N/A')}")

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    while True:
        query = input("\n请输入你的英文问题（q退出）：\n> ")
        if query.lower() in ["q", "quit", "exit"]:
            break
        retrieve(query)
