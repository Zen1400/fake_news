import pytesseract as tess
from PIL import Image

img = Image.open("raw_data/image_m1.jpg")
text = tess.image_to_string(img)

print(text)
