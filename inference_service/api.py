from fastapi import FastAPI, File, UploadFile
from typing import List, Union
from pydantic import BaseModel
import numpy as np
import cv2
from remote_infer_grpc import ort_v5
import os
from dotenv import load_dotenv
import ast

import uuid
import boto3
import base64
import io
from PIL import Image

load_dotenv()

DEBUG = os.getenv('DEBUG', 1)
DEBUG_LOG_IMAGES = "./log/images/"
DEBUG_LOG_VARS = "./log/vars/"

IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.65))
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.15))

GRPC_HOST = os.getenv('GRPC_HOST', 'modelmesh-serving.object-detection')
GRPC_PORT = os.getenv('GRPC_PORT', '8033')
MODEL_NAME = os.getenv('MODEL_NAME', 'object-detection')

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "images")
AWS_S3_INTERNAL_ENDPOINT = os.environ.get("AWS_S3_INTERNAL_ENDPOINT", "")
AWS_S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT", "")

class Box(BaseModel):
    xMax: float
    xMin: float
    yMax: float
    yMin: float

class Detection(BaseModel):
    box: Box
    class_: str
    label: str
    score: float

    class Config:
        allow_population_by_field_name = True
        fields = {
            'class_': 'class'
        }

class Detections(BaseModel):
    detections: List[Detection]
    image_url: str

app = FastAPI()

def push_to_bucket(safe_uid, random_uuid, img_data):
    bucket_path = f"{safe_uid}/{random_uuid}.jpg"
    is_success, buffer = cv2.imencode(".jpg", img_data)
    io_buf = io.BytesIO(buffer)
    s3_route = AWS_S3_INTERNAL_ENDPOINT if AWS_S3_INTERNAL_ENDPOINT else AWS_S3_ENDPOINT
    s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID, 
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             endpoint_url=s3_route)
    s3_client.upload_fileobj(io_buf, AWS_S3_BUCKET, bucket_path)
    return bucket_path

@app.post("/predictions", response_model=Detections)
async def predictions(userId: Union[str, None] = None, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if not userId:
        userId = "anonymous"
    uid_b64 = base64.b64encode(str.encode(userId)).decode("utf-8")
    random_uuid = str(uuid.uuid4())
    request_uuid = uid_b64 + "_" + random_uuid
    bucket_path = push_to_bucket(uid_b64, random_uuid, img_data)
    raw_detections = infer(img_data, CONF_THRESHOLD, IOU_THRESHOLD, request_uuid)
    result = Detections(detections=[], image_url = f"{AWS_S3_ENDPOINT}/{AWS_S3_BUCKET}/{bucket_path}")
    for output in raw_detections:
        box = Box(xMin=output["xMin"], xMax=output["xMax"], yMin=output["yMin"], yMax=output["yMax"])
        detection = Detection(box=box, class_=output["class"],
            label=output["class"].capitalize(), score=output["score"]) # Label is the value displayed to user
        result.detections.append(detection)
    if DEBUG:
        from debug_utility import log_inference, draw_box_and_save
        log_inference(request_uuid, img_data, raw_detections, img_dir=DEBUG_LOG_IMAGES, var_dir=DEBUG_LOG_VARS)
        draw_box_and_save(request_uuid, img_data, raw_detections, DEBUG_LOG_IMAGES)
    return result

if __name__ == "__main__":
    import uvicorn
    MODEL_IMAGE_SIZE = (800,600)
    infer=ort_v5(GRPC_HOST, GRPC_PORT, MODEL_NAME, MODEL_IMAGE_SIZE)
    if DEBUG:
        os.makedirs(DEBUG_LOG_IMAGES, exist_ok=True)
        os.makedirs(DEBUG_LOG_VARS, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)