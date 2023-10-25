"""
This module contains the FastAPI application for the Stable Diffusion XL model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from app.stable_diffusion import StableDiffusionXL

# FastAPI app
app = FastAPI()


class Txt2ImgRequest(BaseModel):
    """
    A request object for generating images from text prompts.

    Attributes:
      prompt (str): The text prompt to generate an image from.
      num_inference_steps (int, optional): The number of inference steps
      to use when generating the image. Defaults to 20.
      batch_size (int, optional): The batch size to use when generating
      the image. Defaults to 1.
    """
    prompt: str
    num_inference_steps: int = 20
    batch_size: int = 1


class Txt2ImgResponse(BaseModel):
    """
    Represents a response object containing a list of images
    generated from text.

    Attributes:
      images (list): A list of images generated from the input text.
    """
    images: list


def get_pipe():
    """
    Returns a StableDiffusionXL object with adapters set.

    :return: StableDiffusionXL object
    """
    pipeline = StableDiffusionXL()
    pipeline.set_adapters()
    return pipeline


# Dependency to get the pipe
pipe_dependency = get_pipe()


@app.post("/txt2img", response_model=Txt2ImgResponse)
def txt2img(request: Txt2ImgRequest,
            pipe: StableDiffusionXL = pipe_dependency) -> dict:
    """
    Converts text to images using the Stable Diffusion XL model.

    Args:
      request (Txt2ImgRequest): An object containing the request parameters.
      pipe (StableDiffusionXL): An instance of the StableDiffusionXL class.

    Returns:
      A dictionary containing the base64-encoded images.
    """
    try:
        images = pipe.generate_images(
            request.num_inference_steps, request.prompt, request.batch_size)
        base64_images = pipe.images_to_base64(images)
        return {"images": base64_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def run():
    """
    Runs the application using Uvicorn server on host 0.0.0.0 and port 3000.
    """
    uvicorn.run(app, host="0.0.0.0", port=3000)
