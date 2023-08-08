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

import pickle
import uuid

DEBUG = os.getenv('DEBUG', 1)
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.65))

CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.2))
GRPC_HOST = os.getenv('GRPC_HOST', 'modelmesh-serving.object-detection')
GRPC_PORT = os.getenv('GRPC_PORT', '8033')
MODEL_NAME = os.getenv('MODEL_NAME', 'object-detection')
COUPON_VALUE = ast.literal_eval(os.getenv('COUPON_VALUE', '[5,10,15]'))

class Box(BaseModel):
    xMax: float
    xMin: float
    yMax: float
    yMin: float

class Detection(BaseModel):
    box: Box
    cValue: float
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


def log_inference(uuid, image, outputs):

    def save_pickle(var, file_name):
        with open(f'./log/vars/{file_name}.pickle', 'wb') as f:
            pickle.dump(var, f)
        return None
    
    cv2.imwrite("./log/images/" + str(uuid) + '.jpg', image)
    outputs_file_name = f"outputs_{str(uuid)}"
    save_pickle(outputs, outputs_file_name)
    return None

@app.post("/predictions", response_model=Detections)
async def predictions(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    request_uuid = uuid.uuid4()
    raw_detections = infer(img_data, CONF_THRESHOLD, IOU_THRESHOLD, request_uuid)
    result = Detections(detections=[])
    for output in raw_detections:
        box = Box(xMin=output["xMin"], xMax=output["xMax"], yMin=output["yMin"], yMax=output["yMax"])
        detection = Detection(box=box,cValue=5, class_=output["class"],
            label=output["label"], score=output["score"])
        result.detections.append(detection)
    if DEBUG:
        log_inference(request_uuid, img_data, raw_detections)
    return result

if __name__ == "__main__":
    import uvicorn
    MODEL_IMAGE_SIZE = (800,600)
    infer=ort_v5(GRPC_HOST, GRPC_PORT, MODEL_NAME, MODEL_IMAGE_SIZE)
    uvicorn.run(app, host="0.0.0.0", port=8000)