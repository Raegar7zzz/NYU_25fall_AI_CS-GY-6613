"""
app_gradio.py
--------------
• Web界面输入英文问题
• 检索Qdrant字幕片段
• 组织成简易回答
• 网页显示
"""

import gradio as gr
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ------------------ 配置 ------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6334
COLLECTION_NAME = "video_chunks"

# ------------------ 连接Qdrant + 加载模型 ------------------

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ 核心检索 + 组织回答 ------------------

def search_and_answer(query_text, top_k=5):
    # 1. 文本嵌入
    query_vec = text_model.encode(query_text, normalize_embeddings=True)
    text_padding = np.zeros(512, dtype=np.float32)
    query_vec_full = np.concatenate([query_vec, text_padding])

    # 2. Qdrant检索
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec_full.tolist(),
        limit=top_k,
    )

    # 3. 组织回答
    if not hits:
        return "No relevant video clips found."

    subtitles = []
    timestamps = []
    videos = []
    for hit in hits:
        payload = hit.payload
        subtitles.append(payload.get("subtitle", ""))
        timestamps.append(f"{payload.get('timestamp', 0)}s")
        videos.append(payload.get("video_name", "Unknown"))

    # 4. 简单生成一段自然回答（串字幕）
    answer = "Based on the video segments, here is the information:\n\n"
    for i, subtitle in enumerate(subtitles):
        answer += f"- ({timestamps[i]}) {subtitle}\n"

    return answer

# ------------------ 搭建Gradio界面 ------------------

iface = gr.Interface(
    fn=search_and_answer,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question about the videos..."),
    outputs="text",
    title="Chat with Your Video Library",
    description="Ask about course videos, and get answers grounded in actual video content."
)

# ------------------ 启动 ------------------

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
