"""
Stable Diffusion model for generating images from prompts.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
import transformers

transformers.utils.move_cache()



class StableDiffusionModel:
    """
    A class for generating images using the Stable Diffusion model.

    Attributes:
    - pipe (StableDiffusionPipeline): The pipeline used for
    generating images.

    Methods:
    - __init__(): Initializes the StableDiffusion object.
    - set_adapters(): Loads LORA weights for the adapter, and sets him
    on the pipeline.
    - images_to_base64(images): Converts a list of images to
    a list of base64-encoded strings.
    - generate_images(num_inference_steps, prompt, batch_size=1):
    Generates a batch of images using the given prompt and number of
    inference steps.
    """

    model = None

    @classmethod
    def get_model(cls):
        """
        Returns the DiffusionPipeline model for stable diffusion.

        Frozen start takes about 10 GB to load, so we load it once and
        reuse it for all requests.

        The model will be loaded onto the GPU for faster computation.

        Returns:
            DiffusionPipeline: The pre-trained DiffusionPipeline model for stable diffusion.
        """
        if cls.model is None:
            # Load the model from folder, 4 GB
            cls.pipe = StableDiffusionPipeline.from_single_file(
                "https://huggingface.co/rkpsmix/epicrealism_naturalSinRC1VAE/blob/main/epicrealism_naturalSinRC1VAE.safetensors",
                load_safety_checker=False,
                torch_dtype=torch.float16,
                token=os.getenv("HUGGINGFACE_TOKEN"),
            ).to("cuda")
            # cls.pipe.unet = torch.compile(
            #     cls.pipe.unet, mode="reduce-overhead", fullgraph=True
            # )
            cls.model = cls.pipe
        return cls.model

    @classmethod
    def set_adapter(cls, repo, adapter_name, weight_name, weight) -> None:
        """
        Loads LORA weights, adapter, and sets on the pipeline.
        """

        cls.model.load_lora_weights(
            repo,
            weight_name=weight_name,
            adapter_name=adapter_name,
        )

        cls.model.set_adapter(
            adapter_name,
            adapter_weights=weight,
        )

        print("Adapter set")

    @classmethod
    def generate_images(cls, num_inference_steps, prompt, width, height, batch_size):
        """
        Generates a batch of images using the given prompt and number
        of inference steps.

        Args:
          num_inference_steps (int): The number of inference steps to use.
          prompt (str): The prompt to use for generating the images.
          batch_size (int, optional): The number of images to generate
          in a batch. Defaults to 1.

        Returns:
          List of images generated using the given prompt and number
          of inference steps.
        """
        images = []
        for _ in range(batch_size):
            image = cls.model(
                prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                cross_attention_kwargs={"scale": 1.0},
            ).images[0]
            images.append(image)
        return images
