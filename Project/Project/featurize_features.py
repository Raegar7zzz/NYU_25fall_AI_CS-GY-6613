"""
featurize_features.py
---------------------
• 从 MongoDB 读取帧图像 + 字幕
• 提取文本 (384 dim) + 图像 (512 dim) 嵌入
• 拼接成 896-dim 向量写入 Qdrant (gRPC 端口 6334)
"""

import base64
from io import BytesIO

import numpy as np
import pymongo
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import open_clip

# ------------------ 参数 ------------------

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "video_rag_db"
COLL_FRAMES = "video_frames"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6334          # gRPC 端口
COLLECTION_NAME = "video_chunks"

SIZE_TEXT = 384             # all-MiniLM-L6-v2 输出维度
SIZE_IMAGE = 512            # OpenCLIP ViT-B-32 输出维度
VECTOR_DIM = SIZE_TEXT + SIZE_IMAGE  # 896

# ------------------ 连接数据库 ------------------

mongo = pymongo.MongoClient(MONGO_URI)[DB_NAME][COLL_FRAMES]
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)

# 若旧集合已存在 —— 先删除再创建，避免维度不一致
if COLLECTION_NAME in [c.name for c in qdrant.get_collections().collections]:
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# ------------------ 加载模型 ------------------

text_model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device)

# ------------------ 处理并写入 ------------------

points = []
for idx, doc in enumerate(mongo.find()):      # 用 idx 作为唯一整数 ID
    subtitle = doc.get("subtitle", "")

    # ---- 文本向量 ----
    text_vec = text_model.encode(subtitle, normalize_embeddings=True)

    # ---- 图像向量 ----
    img_bytes = base64.b64decode(doc["frame_image_base64"])
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_vec = clip_model.encode_image(img_tensor).squeeze().cpu().numpy()
        img_vec /= np.linalg.norm(img_vec)

    # ---- 拼接 ----
    combined_vec = np.concatenate([text_vec, img_vec])  # 896-dim

    points.append(
        PointStruct(
            id=idx,                       # 必须为 int 或 UUID 字符串
            vector=combined_vec.tolist(),
            payload={
                "video_name": doc.get("video_name"),
                "timestamp": doc.get("timestamp_sec"),
                "subtitle": subtitle,
            },
        )
    )

# 写入 Qdrant
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ Inserted {len(points)} vectors (dim={VECTOR_DIM}) into Qdrant!")
