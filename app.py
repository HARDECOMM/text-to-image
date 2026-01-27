import streamlit as st
from t2i_utils import load_model, generate_image

st.set_page_config(page_title="Text to Image Generator", layout="centered")

st.title("🖼️ Text to Image Generator")
st.write("Generate images from text using Stable Diffusion (CPU, Free)")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference Steps", 10, 50, 25)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder="A futuristic city at sunset, ultra realistic, 4k"
)

generate_btn = st.button("🚀 Generate Image")

@st.cache_resource
def get_pipe():
    return load_model()

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Loading model (first time may take a minute)..."):
            pipe = get_pipe()

        with st.spinner("Generating image... please wait"):
            image = generate_image(pipe, prompt, steps, guidance)

        st.success("Done!")
        st.image(image, caption="Generated Image", use_column_width=True)