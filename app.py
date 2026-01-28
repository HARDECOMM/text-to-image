import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import uuid
import os

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="🖼️ Text to Image Generator (CPU)",
    layout="centered"
)

st.title("🖼️ Text to Image Generator")
st.caption("Stable Diffusion • CPU only • Free tier friendly")

# -------------------------------------------------
# Load model safely (cached, CPU only)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipe = pipe.to("cpu")
    return pipe


pipe = load_model()

# -------------------------------------------------
# UI controls
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 10, 20, 12)
    guidance = st.slider("Guidance Scale", 1.0, 8.0, 7.0)

prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder="A futuristic city at sunset, ultra realistic, 4k"
)

generate = st.button("🚀 Generate Image")

# -------------------------------------------------
# Inference
# -------------------------------------------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a text prompt.")
    else:
        with st.spinner("Generating image on CPU... Please wait ⏳"):
            try:
                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                ).images[0]

                # Save locally
                os.makedirs("outputs", exist_ok=True)
                filename = f"outputs/{uuid.uuid4().hex}.png"
                image.save(filename)

                st.success("Image generated successfully!")
                st.image(image, caption=prompt, use_column_width=True)

            except RuntimeError as e:
                st.error("Generation failed due to memory limits.")
                st.code(str(e))
