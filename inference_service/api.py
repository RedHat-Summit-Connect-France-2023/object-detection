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

load_dotenv()

DEBUG = os.getenv('DEBUG', 1)
DEBUG_LOG_IMAGES = "./log/images/"
DEBUG_LOG_VARS = "./log/vars/"

IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.65))
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.15))

GRPC_HOST = os.getenv('GRPC_HOST', 'modelmesh-serving.object-detection')
GRPC_PORT = os.getenv('GRPC_PORT', '8033')
MODEL_NAME = os.getenv('MODEL_NAME', 'object-detection')

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

app = FastAPI()

@app.post("/predictions", response_model=Detections)
async def predictions(userId: Union[str, None] = None, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if not userId:
        userId = "anonymous"
    safe_uid = "".join([c for c in userId if c.isalpha() or c.isdigit() or c in ["@", "."]]).rstrip()
    request_uuid = safe_uid + "_" + str(uuid.uuid4())
    raw_detections = infer(img_data, CONF_THRESHOLD, IOU_THRESHOLD, request_uuid)
    result = Detections(detections=[])
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