from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from stable_diffusion import StableDiffusionXL

app = FastAPI()

def get_nn_model():
    StableDiffusionXL.get_model()
    StableDiffusionXL.set_adapters()
    return StableDiffusionXL

class Txt2ImgRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20
    batch_size: int = 1

class Txt2ImgResponse(BaseModel):
    images: list[str]


@app.get("/ping")
def ping() -> str:
    return "pong"

@app.post("/txt2img", response_model=Txt2ImgResponse)
def txt2img(request: Txt2ImgRequest, pipeline = Depends(get_nn_model)) -> Txt2ImgResponse:
    try:
        images = pipeline.generate_images(request.num_inference_steps, request.prompt, request.batch_size)
        base64_images = pipeline.images_to_base64(images)
        return {"images": base64_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
