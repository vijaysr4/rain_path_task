from PIL import Image

img = Image.open("ihc.jpg")

img.save("ihc.tiff", format="TIFF", compression="tiff_lzw")
