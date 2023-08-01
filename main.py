import os
import uuid
from typing import List, Dict, Any, Union

from PIL import Image
from torchvision.ops import box_convert
import io
import boto3
import cv2
import numpy as np
import supervision as sv
import torch
from fastapi import FastAPI
import requests
import logging
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

app = FastAPI()
headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0'}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

HOME = os.getcwd()

# Get the base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# specify config path
CONFIG_PATH = os.path.join(BASE_DIR, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

# specify weight path
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

# load model
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# specify image path
FILE_NAME = "frame.jpg"
FOLDER = "data"

# s3 bucket name
BUCKET_NAME = "ai-engine-cropped-image-bucket"

# specify classes or prompt
TEXT_PROMPT = "chair, shirt, trousers, pants, skirt, hoodie, sweatpants, suit, blouse, shoes, glasses, car, cap, " \
              "beanie, wristwatch, jewelry, bag, furniture, table, plate, computer, phone"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


@app.get("/root/")
async def root():
    return 'Hello World'


@app.get("/process-test/")
async def process_test(url: str):
    image_path = os.path.join(HOME, download_image(url, FILE_NAME, FOLDER))
    return {
        "statusCode": 200,
        "header": "application/json",
        "body": image_path
    }


@app.get("/process-frame/")
async def process_frame(url: str):
    detections = []
    image_path = download_image(url, FILE_NAME, FOLDER)
    logger.info("Processing image: %s", image_path)
    if not image_path:
        return {
            "statusCode": 400,
            "header": "application/json",
            "body": {
                "message": "The image could not be downloaded."
            }
        }

    image_source, image = load_image(image_path)

    # Perform inference
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    detections = annotate_and_crop(image_source, boxes, logits, phrases)

    return {
        "statusCode": 200,
        "header": "application/json",
        "body": detections
    }


def download_image(url, filename, folder):
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
            print(f"Image downloaded and saved: {filepath}")
            return filepath
    else:
        print(f"Failed to download image from URL: {url}")
        return None


def annotate_and_crop(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> List[
    Dict[str, Union[Dict[str, Any], str, Any]]]:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # annotated_file_path = os.path.join(FOLDER, "annotated.jpg")
    # cv2.imwrite(annotated_file_path, annotated_frame)
    #
    # annotated_url = upload_annotated_frame_to_s3(f"{str(uuid.uuid4())}_annotated", BUCKET_NAME, annotated_frame)

    image_source = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    detections = []
    for box, logit, phrase in zip(xyxy, logits, phrases):
        xmin, ymin, xmax, ymax = box

        # Convert the bounding box coordinates to integers
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        cropped_image = image_source[ymin:ymax, xmin:xmax]

        if cropped_image.size != 0:
            url = upload_to_s3(f"{phrase}_{str(uuid.uuid4())}", BUCKET_NAME, cropped_image)

            # Convert the bounding box coordinates to percentages
            left = xmin / w
            top = ymin / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            # create detection obj
            detection = {
                "label": phrase,
                "confidence": f"{logit:.2f}",
                # "annotated_url": annotated_url,
                "url": url,
                "bounding_box": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            }
            detections.append(detection)

    return detections


def upload_to_s3(label: str, bucket: str, cropped_image: np.ndarray) -> str:
    file_name = f"{label}.jpg"
    print(file_name)
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        s3 = session.resource('s3')

        # Convert cropped_image to bytes
        _, image_bytes = cv2.imencode(".jpg", cropped_image)
        byte_stream = io.BytesIO(image_bytes.tobytes())

        # Upload to s3
        s3.Object(bucket, file_name).put(Body=byte_stream)
    except Exception as e:
        raise Exception(f"Failed to upload image to S3: {file_name} with error: {e}")

    url = f"https://{bucket}.s3.us-east-2.amazonaws.com/{file_name}"
    print(url)
    return url


def upload_annotated_frame_to_s3(label: str, bucket: str, annotated_image: np.ndarray) -> str:
    file_name = f"{label}.jpg"
    print(file_name)
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        s3 = session.resource('s3')

        # Convert cropped_image to bytes
        _, image_bytes = cv2.imencode(".jpg", annotated_image)
        byte_stream = io.BytesIO(image_bytes.tobytes())

        # Upload to s3
        s3.Object(bucket, file_name).put(Body=byte_stream)
    except Exception as e:
        raise Exception(f"Failed to upload image to S3: {file_name} with error: {e}")

    url = f"https://{bucket}.s3.us-east-2.amazonaws.com/{file_name}"
    print(url)
    return url
