import streamlit as st
import cv2
from masks_to_text import BlipImageCaptioning
from text_to_embeddings import TextToEmbeddings
from PIL import Image
import numpy as np

# Initialize classes outside of the button press to avoid reinitialization
@st.cache_resource
def init_models():
    blip = BlipImageCaptioning()
    uae = TextToEmbeddings()
    return blip, uae


st.set_page_config("VODKA", "🥷", layout="wide")

blip, uae = init_models()

css = """
<style>
img {
    padding: 10%;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


st.header("Voting Over Distilled Knowledge Associations (VODKA)")

files = st.file_uploader(
    label="Upload the cutouts",
    type=["jpg", "png"],
    accept_multiple_files=True,
    help="Upload the top 5 cutouts.",
)

prompt = st.text_input(
    "Enter the prompt for segmentation",
    help="Provide a descriptive prompt to guide the segmentation process.",
)

if st.button(label="Process", help="Click to start the segmentation process."):
    if files is not None and prompt:
        with st.spinner("Processing image..."):
            images_cv2 = []

            for file in files:
                image = Image.open(file)
                image_np = np.array(image)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                images_cv2.append(image_cv2)

        with st.spinner("Generating descriptions..."):
            captions = [blip.generate_description(cutout) for cutout in images_cv2]
            # st.write("Captions:")
            # st.write(captions)

        with st.spinner("Calculating similarity..."):
            similarities = [uae.get_similarity(prompt, caption) for caption in captions]
            # st.write("Similarity Scores:")
            # st.write(similarity)

        st.image(
            files,
            width=250,
            caption=[f"{c}. {s}" for c, s in zip(captions, similarities)],
        )

    else:
        st.error("Please upload an image and enter a prompt to proceed.")



