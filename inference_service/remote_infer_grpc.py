import random
import numpy as np
import cv2
import torch
import torchvision
import argparse
import requests
import grpc
import grpc_predict_v2_pb2_grpc
import grpc_predict_v2_pb2
import yaml
import time

from transformers import YolosFeatureExtractor
import transformers
import json
import pickle
from PIL import Image

LOG_DIR = "./log"
FEATURE_EXTRACTOR_DIR = "./feature_extractor"
MAX_RETURNED_REDICTIONS = 3

class ort_v5:
    def __init__(self, grpc_host, grpc_port, model_name, model_img_size):
        self.host = grpc_host
        self.port = grpc_port
        self.model_name = model_name
        self.model_img_size=model_img_size
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}", options = options)
        self.stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(self.channel)

    def __call__(self, img_data, conf_thres, iou_thres, uuid):
        """
        Makes a prediction on a given image by calling an inference endpoint served by ModelMesh.

        The model is based on YoloV5 (https://github.com/ultralytics/yolov5), exported as ONNX, and served
        using OpenVino Model Server.
        """
        start_call = time.time()

        # image preprocessing
        original_img_size = img_data.shape
        image, ratio, dwdh = self.letterbox(img_data) # Resize and pad image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature_extractor = YolosFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR_DIR)
        inputs = feature_extractor(images=image, return_tensors="pt")
        flatten_inputs = inputs.data["pixel_values"].numpy().flatten().tolist()

        # request content building
        inputs = []
        inputs.append(grpc_predict_v2_pb2.ModelInferRequest().InferInputTensor())
        inputs[0].name = "pixel_values"
        inputs[0].datatype = "FP32"
        inputs[0].shape.extend([1, 3, 1066, 800]) # Expected size of the image after feature extractor
        inputs[0].contents.fp32_contents.extend(flatten_inputs)

        # request building
        request = grpc_predict_v2_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.inputs.extend(inputs)

        # Call the gRPC server and get the response
        try:
            start_inference_time = time.time()
            response = self.stub.ModelInfer(request)
            end_inference_time = time.time()
            inference_time = end_inference_time-start_inference_time
        except grpc.RpcError as e:
            if e.code() == StatusCode.UNAVAILABLE:
                raise Exception("Failed to connect to gRPC server")
            else:
                raise Exception(f"Failed to call gRPC server: {e.details()}")
        
        # unserialize response content
        result_arr_logits = np.frombuffer(response.raw_output_contents[0], dtype=np.float32).reshape(1,100,47) # Model has 47 labels
        result_arr_boxes = np.frombuffer(response.raw_output_contents[1], dtype=np.float32).reshape(1,100,4) # 4 coords
        logits = torch.tensor(result_arr_logits)
        pred_boxes = torch.tensor(result_arr_boxes)
        DETR = transformers.models.detr.modeling_detr.DetrObjectDetectionOutput(logits=logits, pred_boxes=pred_boxes)
        target_sizes = [(800, 600)]
        results = feature_extractor.post_process_object_detection(DETR, threshold=conf_thres, target_sizes=target_sizes)[0]

        # Non-maximum Suppression (NMS)
        indices = torchvision.ops.nms(results["boxes"], results["scores"], iou_thres)
        nms_results = {k: torch.index_select(v, 0, indices) for k,v in results.items()}

        # Format output
        config_file = "./config/reverse-config.json"
        with open(config_file, 'r') as j:
            config = json.loads(j.read())
        outputs = []
        for score, label, box in zip(nms_results["scores"], nms_results["labels"], nms_results["boxes"]):
            class_ = config[str(label.item())]
            if not "sub_" in class_ and class_ != "ignore":
                x, y, x2, y2 = self.rebase_coord(box.tolist(), dwdh, ratio, original_img_size)
                item = {
                    "xMin": x , "xMax": x2, "yMin": y, "yMax": y2,
                    "label": str(label.item()), "class": class_, "score": float(score.item())}
                outputs.append(item)
                if len(outputs) >= MAX_RETURNED_REDICTIONS:
                    break
        
        end_call = time.time()
        call_time = end_call - start_call
        self.stdout(uuid, outputs, call_time, inference_time)
        return outputs

    def rebase_coord(self, box, dwdh, ratio, img_size):
        x, y, x2, y2 = box
        dw, dh = dwdh
        height = img_size[0]
        width = img_size[1]
        x, x2 = (x - dw) / ratio, (x2 - dw) / ratio
        y, y2 = (y - dh) / ratio, (y2 - dh) / ratio
        x, x2 = max(x, 0), min(x2, width)
        y, y2 = max(y, 0), min(y2, height)
        x, x2 = x / width, x2 / width
        y, y2 = y / height, y2 / height
        return x, y, x2, y2

    def letterbox(self, im, color=(114, 114, 114), scaleup=True):
        # Resize and pad image
        shape = im.shape[:2]  # current shape [height, width]
        new_shape= self.model_img_size

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def stdout(self, uuid, outputs, call_time, inference_time):
        print(f"--- Processed UUID: {str(uuid)} in {call_time:.2f} seconds ---")
        for elem in outputs:
            class_ = elem["class"]
            score = elem["score"]
            print(f"Detected {class_} with condifence {score}")
        print(f"Inference took {inference_time:.2f} seconds")
        print("------")
        return None