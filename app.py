import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os
import uuid

st.set_page_config(page_title="Text to Image (CPU)", layout="centered")
st.title("🖼️ Text to Image Generator (CPU)")
st.caption("Free tier • CPU only • Stable")

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 5, 15, 8)
    guidance = st.slider("Guidance Scale", 1.0, 7.5, 6.0)

prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder="A car running on a highway, cinematic"
)

generate = st.button("🚀 Generate Image")

# ----------------------------
# Lazy model loader
# ----------------------------
@st.cache_resource
def load_pipeline():
    return StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cpu")


# ----------------------------
# Inference
# ----------------------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.info("⏳ First run may take 2–3 minutes on CPU. Please wait.")

        try:
            pipe = load_pipeline()

            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance
            ).images[0]

            os.makedirs("outputs", exist_ok=True)
            path = f"outputs/{uuid.uuid4().hex}.png"
            image.save(path)

            st.success("✅ Image generated!")
            st.image(image, caption=prompt, use_column_width=True)

        except Exception as e:
            st.error("❌ Generation failed (CPU limit reached).")
            st.code(str(e))