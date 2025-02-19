import streamlit as st
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image

# Load the model and scheduler
model = UNet2DModel.from_pretrained("zaibutcooler/umi")
scheduler = DDPMScheduler.from_pretrained("zaibutcooler/umi")

# Create the pipeline
pipeline = DDPMPipeline(
    unet=model,
    scheduler=scheduler,
)

# Streamlit UI
st.title("DDPM Image Generation")
batch_size = st.slider("Select batch size", min_value=1, max_value=8, value=4)

if st.button("Generate Images"):
    with st.spinner("Zai's too broke to rent a GPU, so grab a coffee while you wait..."):
        images = pipeline(batch_size=batch_size).images

    st.success("Images generated successfully!")
    cols = st.columns(batch_size)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(Image.fromarray(img), use_column_width=True)