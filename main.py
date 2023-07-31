import os
import boto3
import supervision as sv
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
    if not image_path is None:
        image_source, image = load_image(image_path)

        # perform inference
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        print(boxes, logits, phrases)

        for box, logit, phrase in zip(boxes, logits, phrases):
            label = phrase
            left, top, width, height = box

            right = left + width
            bottom = top + height

            try:
                url = crop_and_save_image(
                    image_path=image_path,
                    box=box, bucket=BUCKET_NAME,
                    label=label)

                detection = {
                    "label": label,
                    "url": url,
                    "bounding_box": {
                        "left": left.item(),
                        "right": right.item(),
                        "top": top.item(),
                        "bottom": bottom.item()
                    }
                }

                detections.append(detection)

                print("Label: ", label)
                print("Url: ", url)
                print("Bounding Box Left: ", left)
                print("Bounding Box Right: ", right)
                print("Bounding Box Top: ", top)
                print("Bounding Box Bottom: ", bottom)
                print()

            except Exception as e:
                logger.error(e)
                return {
                    "statusCode": 500,
                    "header": "application/json",
                    "body": {
                        "message": "An error occurred while processing the frame."
                    }
                }

    else:
        return {
            "statusCode": 400,
            "header": "application/json",
            "body": {
                "message": "The image could not be downloaded."
            }
        }

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


def crop_and_save_image(image_path: str, box: torch.Tensor, bucket: str, label: str) -> str:
    """Crops and saves each bounding box image out of the frame and returns the S3 URL of the cropped image."""
    h, w, _ = image_path.shape

    boxes = convert_to_xyxy(boxes=box)
    left, top, right, bottom = boxes
    cropped_image = image_path[top:bottom, left:right]

    file_name = f"{label}_{left}_{top}_{right}_{bottom}.jpg"
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
        )
        s3 = session.resource('s3')
        s3.Object(bucket, file_name).put(Body=cropped_image)
    except Exception as e:
        raise Exception(f"Failed to upload image to S3: {file_name} with error: {e}")

    url = f"https://s3.amazonaws.com/{bucket}/{file_name}"
    return url


def convert_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from the "cxcywh" format to the "xyxy" format."""
    h, w, _ = boxes.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = torch.stack([boxes[:, :2], boxes[:, 2:] + boxes[:, :2]], dim=1)
    return xyxy

