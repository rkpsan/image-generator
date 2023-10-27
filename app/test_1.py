import base64
from io import BytesIO
import httpx
from PIL import Image

def test_timeout():
    with httpx.Client(base_url="http://localhost:8000", timeout=400.0) as client:
        request_body = {
        "num_inference_steps": 20,
        "prompt": "A vector-style design an apple, enclosed within a circular border (white background)",
        "batch_size": 5
        }
        response = client.post("/txt2img", json=request_body)
        assert response.status_code == 200
        assert len(response.json()["images"]) == request_body["batch_size"]
        for i, image in enumerate(response.json()["images"]):
            # Decode the base64 string
            image_bytes = base64.b64decode(image)

            # Create a BytesIO object
            image_io = BytesIO(image_bytes)

            # Open the image using PIL
            image = Image.open(image_io)

            # Save the image to a file with a unique name based on the index
            image.save(f"output_{i}.png")
            
def multiple_requests():
    with httpx.Client(base_url="http://localhost:8000", timeout=400.0) as client:
        request_body = {
        "num_inference_steps": 10,
        "prompt": "A vector-style design where the bear is raising a cup in a toast gesture, enclosed within a circular border (white background) LogoRedmondV2",
        "batch_size": 1
        }
        for i in range(5):
            response = client.post("/txt2img", json=request_body)
            assert response.status_code == 200
            assert len(response.json()["images"]) == request_body["batch_size"]
            image = response.json()["images"][0]
            # Decode the base64 string
            image_bytes = base64.b64decode(image)

            # Create a BytesIO object
            image_io = BytesIO(image_bytes)

            # Open the image using PIL
            image = Image.open(image_io)

            # Save the image to a file with a unique name based on the index
            image.save(f"output_{i}.png")
        