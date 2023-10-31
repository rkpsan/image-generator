"""
Stable Diffusion XL model for generating images from prompts.
"""

import base64
import io
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from PIL import Image


class StableDiffusionXL:
    """
    A class for generating images using the Stable Diffusion XL model.

    Attributes:
    - vae (AutoencoderKL): The VAE model used by the pipeline.
    - pipe (StableDiffusionXLPipeline): The pipeline used for
    generating images.

    Methods:
    - __init__(): Initializes the StableDiffusionXL object.
    - set_adapters(): Loads LORA weights for the LogoRedmondV2,
    StickersRedmond,and ColoringBookRedmond adapters, and sets them
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

        Frozen start takes about 16 GB to load, so we load it once and
        reuse it for all requests.

        If the model has not been initialized, it will be loaded from the pre-trained models
        "madebyollin/sdxl-vae-fp16-fix" and "stabilityai/stable-diffusion-xl-base-1.0".
        The model will be loaded onto the GPU for faster computation.

        Returns:
            DiffusionPipeline: The pre-trained DiffusionPipeline model for stable diffusion.
        """
        if cls.model is None:
            cls.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )

            cls.euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )

            # Load the model from HuggingFace, 6 GB
            cls.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                scheduler=cls.euler_anc,
                vae=cls.vae,
                torch_dtype=torch.float16,
            ).to("cuda")
            cls.pipe.unet = torch.compile(
                cls.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            cls.model = cls.pipe
        return cls.model

    @classmethod
    def set_adapters(cls) -> None:
        """
        Loads LORA weights for the LogoRedmondV2, StickersRedmond,
        and ColoringBookRedmond adapters, and sets them on the pipeline.
        """

        cls.model.load_lora_weights(
            "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
            weight_name="LogoRedmondV2-Logo-LogoRedmAF.safetensors",
            adapter_name="LogoRedmondV2",
        )
        cls.model.load_lora_weights(
            "artificialguybr/StickersRedmond",
            weight_name="StickersRedmond.safetensors",
            adapter_name="Stickers",
        )
        cls.model.load_lora_weights(
            "artificialguybr/ColoringBookRedmond",
            weight_name="ColoringBookRedmond-ColoringBookAF.safetensors",
            adapter_name="ColoringBookRedmond"
        )

        cls.model.set_adapters(
            [
                "LogoRedmond",
                "Stickers",
                "ColoringBook"
            ],
            adapter_weights=[1, 1, 0.5],
        )

        print("Adapters set")

    @staticmethod
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

    @classmethod
    def generate_images(cls, num_inference_steps, prompt, batch_size=1):
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
                num_inference_steps=num_inference_steps,
                cross_attention_kwargs={"scale": 1.0},
            ).images[0]
            images.append(image)
        return images
