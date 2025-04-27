import cv2
import pysrt
import pymongo
import base64
import datetime

# -------------- 配置 --------------
VIDEO_PATH = "test_video_long.mp4"  # 视频路径
SRT_PATH = "test_video_long.srt"    # 字幕路径
MONGO_URI = "mongodb://localhost:27017/"  # 本地MongoDB地址
DB_NAME = "video_rag_db"
COLLECTION_NAME = "video_frames"

FRAME_INTERVAL = 1  # 每隔多少秒取一帧

# -------------- 初始化 MongoDB --------------
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# 如果想每次跑之前清空，可以取消下面注释
# collection.delete_many({})

# -------------- 解析字幕 --------------
subs = pysrt.open(SRT_PATH)

# 帮助函数：根据时间戳找到对应字幕
def find_subtitle(current_seconds):
    for sub in subs:
        start = sub.start.ordinal / 1000.0  # 毫秒转秒
        end = sub.end.ordinal / 1000.0
        if start <= current_seconds <= end:
            return sub.text
    return ""

# -------------- 处理视频 --------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Video FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")

current_time = 0

while current_time <= duration:
    # 设置到指定时间
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # 单位是毫秒
    ret, frame = cap.read()
    if not ret:
        break

    # 编码成JPEG再转base64，方便存数据库
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = base64.b64encode(buffer).decode('utf-8')

    # 查找对应字幕
    subtitle_text = find_subtitle(current_time)

    # 保存到MongoDB
    record = {
        "video_name": VIDEO_PATH,
        "timestamp_sec": current_time,
        "frame_image_base64": frame_bytes,
        "subtitle": subtitle_text
    }
    collection.insert_one(record)

    print(f"Inserted frame at {current_time:.2f}s with subtitle: {subtitle_text}")

    # 下一帧
    current_time += FRAME_INTERVAL

cap.release()
print("ETL completed!")
