import os
from PIL import Image

root_path = os.getcwd()
assets_path = f"{root_path}/app/assets"

def main():
    img = Image.open(f"{assets_path}/img-1.png")
    img.show()

    # converting to grayscale
    gray_img = img.convert('L')
    gray_img.show()



main()