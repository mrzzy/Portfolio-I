#
# util.py
# Utilities
#

from io import BytesIO
from PIL import Image

## Utilities
# Convert the given image data into a PIL image 
# Returns the converted image
def convert_image(img_data):
    buffer = BytesIO(img_data)
    return Image.open(buffer)

# Read the file at the given path.
# Converts the contents of the file to base64 encoding
# Returnsf the base64 contents
def read_file(path):
    with open(path, "rb") as f:
        content = f.read()
        return content

