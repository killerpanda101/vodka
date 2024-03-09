import streamlit as st

css = """
<style>
img {
    padding: 10%;
}
"""

st.set_page_config("VODKA", "🥷", layout="wide")

st.header("Voting Over Distilled Knowledge Associations (VODKA)")

st.file_uploader(
    label="Upload the image you want segmented", type=["jpg"], help="sfdsfa"
)
st.text_input("Enter the prompt for segmentation", help="dgagagr")
st.button(label="Process", help="gsdgfgdf")
images = ["./images/telephone_booth.jpg"] * 5
st.image(
    images,
    width=250,
    caption=[
        "some generic text, some generic text, some generic text, some generic text"
    ]
    * len(images),
)
st.write(css, unsafe_allow_html=True)
