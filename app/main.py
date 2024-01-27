from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

import base64
import io
from PIL import Image

from stable_diffusion import StableDiffusionModel

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    get_nn_model()


def get_nn_model():
    StableDiffusionModel.get_model()
    return StableDiffusionModel


def images_to_base64(images) -> list:
    """
    Converts a list of images to a list of base64-encoded strings.

    Args:
      images (list): A list of images in numpy array format.

    Returns:
      list: A list of base64-encoded strings representing the input images.
    """
    images_base64 = []
    for image in images:
        buff = io.BytesIO()
        image.save(buff, format="PNG")
        image_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        images_base64.append(image_base64)
    return images_base64


class Txt2ImgRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20
    width: int = 512
    height: int = 512
    batch_size: int = 1


class Txt2ImgResponse(BaseModel):
    images: list[str]


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/txt2img", response_model=Txt2ImgResponse)
def txt2img(request: Txt2ImgRequest, pipeline=Depends(get_nn_model)) -> Txt2ImgResponse:
    print(request)
    try:
        images = pipeline.generate_images(
            num_inference_steps=request.num_inference_steps,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            batch_size=request.batch_size,
        )
        base64_images = Txt2ImgResponse(images=images_to_base64(images))
        return base64_images
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
