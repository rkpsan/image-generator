import os
import datetime
from PIL import Image
from io import BytesIO
import base64
import httpx

def save_images(response):
    dir_name = 'test_images'
    
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time to create a unique subdirectory name
    sub_dir_name = now.strftime("%Y%m%d_%H%M%S")

    # Create the subdirectory inside the main directory
    os.makedirs(os.path.join(dir_name, sub_dir_name), exist_ok=True)

    for i, image in enumerate(response.json()["images"]):
        # Decode the base64 string
        image_bytes = base64.b64decode(image)

        # Create a BytesIO object
        image_io = BytesIO(image_bytes)

        # Open the image using PIL
        image = Image.open(image_io)

        # Save the image to a file with a unique name based on the index
        # The image is saved in the unique subdirectory created earlier
        image.save(os.path.join(dir_name, sub_dir_name, f"output_{i}.png"))

def test_without_lora():
    with httpx.Client(base_url="http://localhost:8000", timeout=400.0) as client:
        request_body = {
        "num_inference_steps": 20,
        "prompt": "A vector-style design an apple, enclosed within a circular border (white background)",
        "batch_size": 5
        }
        response = client.post("/txt2img", json=request_body)
        assert response.status_code == 200
        assert len(response.json()["images"]) == request_body["batch_size"]
        save_images(response)
        
def test_with_lora():
    with httpx.Client(base_url="http://localhost:8000", timeout=400.0) as client:
        request_body = {
        "num_inference_steps": 20,
        "prompt": "A vector-style design an apple, enclosed within a circular border (white background), LogoRedmondV2",
        "batch_size": 5
        }
        response = client.post("/txt2img", json=request_body)
        assert response.status_code == 200
        assert len(response.json()["images"]) == request_body["batch_size"]
        save_images(response)
            
def multiple_requests():
    with httpx.Client(base_url="http://localhost:8000", timeout=400.0) as client:
        request_body = {
        "num_inference_steps": 20,
        "prompt": "A vector-style design an apple, enclosed within a circular border (white background)",
        "batch_size": 1
        }
        for i in range(5):
            response = client.post("/txt2img", json=request_body)
            assert response.status_code == 200
            assert len(response.json()["images"]) == request_body["batch_size"]
            
            # Call the save_images function
            save_images(response)
        