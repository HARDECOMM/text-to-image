import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import time
import os
import uuid

st.set_page_config(page_title="Text to Image (CPU)", layout="centered")
st.title("🖼️ Text to Image Generator (CPU • Fast)")
st.caption("Stable Diffusion Turbo • Free tier • CPU only")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 1, 4, 2)
    guidance = st.slider("Guidance Scale", 0.0, 2.0, 1.0)

prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder="A car running on a highway, cinematic"
)

generate = st.button("🚀 Generate Image")

# ---------------- Model Loader ----------------
@st.cache_resource
def load_pipeline():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    return pipe

# ---------------- Inference ----------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        progress = st.progress(0)
        status = st.empty()

        # Fake progress (keeps Streamlit healthy)
        for i in range(40):
            progress.progress(i + 1)
            status.text("Initializing model...")
            time.sleep(0.03)

        try:
            pipe = load_pipeline()

            for i in range(40, 70):
                progress.progress(i + 1)
                status.text("Running inference on CPU...")
                time.sleep(0.02)

            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance
            ).images[0]

            os.makedirs("outputs/frames", exist_ok=True)
            img_id = uuid.uuid4().hex
            img_path = f"outputs/frames/{img_id}_frame_000.png"
            image.save(img_path)

            for i in range(70, 100):
                progress.progress(i + 1)
                status.text("Finalizing output...")
                time.sleep(0.02)

            progress.empty()
            status.empty()

            st.success("✅ Image generated successfully")
            st.image(image, caption=prompt, use_column_width=True)

            st.info("🧩 Frame saved for MMAction2-style processing")
            st.code(img_path)

        except Exception as e:
            st.error("❌ Generation failed on CPU")
            st.code(str(e))
