from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_txt2img():
    request_body = {
        "num_inference_steps": 10,
        "prompt": "A cat sitting on a mat",
        "batch_size": 1
    }
    response = client.post("/txt2img", json=request_body)
    print(response.status_code == 200)
    breakpoint()
    print(len(response.json()["images"]) == 1)

test_txt2img()