from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel
import numpy as np
import cv2
from remote_infer_grpc import ort_v5
import os
from dotenv import load_dotenv
import ast
load_dotenv()

import io
import boto3
import base64
import uuid

GRPC_HOST = os.getenv('GRPC_HOST', 'modelmesh-serving.object-detection')
GRPC_PORT = os.getenv('GRPC_PORT', '8033')
MODEL_NAME = os.getenv('MODEL_NAME', 'object-detection')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.2))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.5))
CLASSES_FILE = 'classes.yaml'

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
    try:
        bucket_path = f"{safe_uid}/{random_uuid}.jpg"
        is_success, buffer = cv2.imencode(".jpg", img_data)
        io_buf = io.BytesIO(buffer)
        s3_route = AWS_S3_INTERNAL_ENDPOINT if AWS_S3_INTERNAL_ENDPOINT else AWS_S3_ENDPOINT
        s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID, 
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                endpoint_url=s3_route)
        s3_client.upload_fileobj(io_buf, AWS_S3_BUCKET, bucket_path)
    except Exception as e:
        print(f"Exception while pushing to bucket: {e}")
    return bucket_path

@app.post("/predictions", response_model=Detections)
async def predictions(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    out = infer(img_data, CONF_THRESHOLD, IOU_THRESHOLD)
    classes = infer.class_name()
    raw_detections = out.tolist()
    userId = "anonymous"
    uid_b64 = base64.b64encode(str.encode(userId)).decode("utf-8")
    random_uuid = str(uuid.uuid4())
    request_uuid = uid_b64 + "_" + random_uuid
    bucket_path = push_to_bucket(uid_b64, random_uuid, img_data)
    result = Detections(detections=[], image_url = f"{AWS_S3_ENDPOINT}/{AWS_S3_BUCKET}/{bucket_path}")
    for raw_detection in raw_detections:
        # Boxes are returned in xMax,xMin,yMax,yMin coordinates on the 640x640 image
        box = Box(xMax=raw_detection[2]/640, xMin=raw_detection[0]/640, yMax=raw_detection[3]/640, yMin=raw_detection[1]/640)
        class_number = int(raw_detection[5])
        detection = Detection(box=box, 
            class_=classes[class_number], label=classes[class_number].capitalize(), score=raw_detection[4])
        print(detection)
        result.detections.append(detection)
    return result

if __name__ == "__main__":
    import uvicorn
    infer=ort_v5(GRPC_HOST, GRPC_PORT, MODEL_NAME, 640, CLASSES_FILE)
    uvicorn.run(app, host="0.0.0.0", port=8000)