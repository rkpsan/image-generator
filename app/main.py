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


pipe_instance = None
global pipe_instance

def get_pipe():
    if not pipe_instance:
        pipe_instance = StableDiffusionXL()
        pipe_instance.set_adapters()
    return pipe_instance

@app.get("/ping")
def ping() -> str:
    return "pong"

@app.post("/txt2img", response_model=Txt2ImgResponse)
def txt2img(request: Txt2ImgRequest,
            pipe: StableDiffusionXL = Depends(get_pipe)) -> Txt2ImgResponse:
    try:
        images = pipe.generate_images(
            request.num_inference_steps, request.prompt, request.batch_size)
        base64_images = pipe.images_to_base64(images)
        return {"images": base64_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run():
    uvicorn.run(app, host="0.0.0.0", port=3000)
