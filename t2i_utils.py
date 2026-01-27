import torch
from diffusers import StableDiffusionPipeline

@torch.no_grad()
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")
    return pipe


def generate_image(pipe, prompt, steps, guidance):
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    return image