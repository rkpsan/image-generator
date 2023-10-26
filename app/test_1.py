import httpx

def test_ping():
    with httpx.Client(base_url="http://localhost:8000", timeout=40.0) as client:
        request_body = {
        "num_inference_steps": 10,
        "prompt": "A cat sitting on a mat",
        "batch_size": 1
        }
        response = client.post("/txt2img", json=request_body)
        assert response.status_code == 200
        assert len(response.json()["images"]) == 1