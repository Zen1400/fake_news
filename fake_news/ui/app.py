from email.mime import image
import streamlit as st
from fake_news.interface.main import pred
import pytesseract as tess
from PIL import Image


"""
# Welcome to your fake news detection website
"""

st.markdown("## Please upload an image of your article below")

st.set_option('deprecation.showfileUploaderEncoding', False)

file= st.file_uploader("Upload a jpg image", type="jpg")


if file is not None :
    image = Image.open(file)
    text = tess.image_to_string(image)
    "## We are processing your article, thank you for your patience"
    prediction = pred(text)
    st.markdown(f"""## The probability that this article is true is :
                {prediction[0][0]}""")
