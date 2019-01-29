#
# util.py
# Utilities
#

from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

## Utilities
# Encode the given Pillow image into a base64 encoded string 
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    img_str = b64encode(img_data).decode("utf-8")
    return img_str

# Decode the given base64 encoded image string into a PIL image 
# Returns the converted image
def decode_image(img_str):
    img_data = b64decode(img_str)
    buffer = BytesIO(img_data)
    return Image.open(buffer)

# Read the file at the given path.
# Converts the contents of the file to base64 encoding
# Returnsf the base64 contents
def read_file(path):
    with open(path, "rb") as f:
        content = f.read()
        return content

# Apply the given setting overrides to the given default settings
# Returns the default settings with the overrides applyed
def apply_settings(overrides, defaults):
    settings = dict(defaults) # Make shallow copy to make changes on
    
    for setting, value in overrides.items():
        settings[setting] = value
    
    return settings
