from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from stable_diffusion import StableDiffusionXL

app = FastAPI()


class Txt2ImgRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20
    batch_size: int = 1

class Txt2ImgResponse(BaseModel):
    images: list[str]

@app.on_event("startup")
def init_pipeline():
    global pipeline
    pipeline = StableDiffusionXL()
    pipeline.set_adapters()
    return pipeline

@app.get("/ping")
def ping() -> str:
    return "pong"

@app.post("/txt2img", response_model=Txt2ImgResponse)
def txt2img(request: Txt2ImgRequest) -> Txt2ImgResponse:
    try:
        images = pipeline.generate_images(
            request.num_inference_steps, request.prompt, request.batch_size)
        base64_images = pipeline.images_to_base64(images)
        return {"images": base64_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)
