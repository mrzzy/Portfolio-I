#
# client.py
# Style Transfer Client
#

import api
import uuid
import time
import requests
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from util import read_file, convert_image

# Parse command line args for the client creating program options
# Returns the parsed program options
def parse_args():
    # Setup parser and parse arguments
    parser = argparse.ArgumentParser(description="""Style Transfer Client
    - Performs style transfer through style transfer server
    """)
    parser.add_argument("-v", action="store_true", help="produce verbose output")
    parser.add_argument("-c", nargs="?", type=float, help="how much weight to content reproduction")
    parser.add_argument("-s", nargs="?", type=float, help="how much weight to style reproduction")
    parser.add_argument("-d", nargs="?", type=float, help="how much weight to given produce a smooth image")
    parser.add_argument("-n", nargs="?", type=int, help="how many iterations of style transer to perform")
    parser.add_argument("content", help="path to the content image. \
                        Use special value ':camera' to obtain image from camera")
    parser.add_argument("style", help="path to the style image")
    #TODO; add options for stylefn settings
    args = parser.parse_args()

    # Parse style transfer settings
    settings = {}
    if not args.c is None: settings[api.SETTING_CONTENT_WEIGHT_KEY] = args.c
    if not args.s is None: settings[api.SETTING_STYLE_WEIGHT_KEY] = args.s
    if not args.d is None: settings[api.SETTING_DENOISE_WEIGHT_KEY] = args.d
    if not args.n is None: settings[api.SETTING_NUMBER_EPOCHS_KEY] = args.n

    # Build Program options
    options = { 
        "content_path": args.content if args.content != ":camera" else None,
        "use_camera": True if args.content == ":camera" else False,
        "style_path": args.style,
        "verbose": args.v,
        "settings": settings
    }
    return options

if __name__ == "__main__":
    # Read program options
    options = parse_args()
    
    # Load style and content image data
    if options["use_camera"]:
        # TODO: use opencv to obtain image from webcam
        raise NotImplementedError
    else:
        content_path, style_path = options["content_path"], options["style_path"]
        content_data = read_file(content_path)
        style_data = read_file(style_path)
    
    # Build style transfer payload
    tag_id = str(uuid.uuid4())
    settings = options["settings"]
    payload = api.pack_payload(content_data, style_data, tag_id, settings)

    # Trigger style transfer on server with api
    if options["verbose"]: print("Sending style transfer request to server... ",
                                 end="")
    
    r = requests.post(api.SERVER_URL + "/api/style", json=payload)
    
    
    # Wait for success response from server
    if options["verbose"]: print("Waiting for response from server ", end="")
    has_pastiche = False
    while not has_pastiche:
        r = requests.get(api.SERVER_URL + "/api/pastiche/" + tag_id)
        if r.status_code == api.STATUS_OK:
            print(" Done!")
            has_pastiche = True
        elif r.status_code == api.STATUS_NOT_READY:
            print("#", end="", flush=True)
            time.sleep(1)
        else:
            raise ValueError("FATAL: something went wrong")

    
    # Read pastiche image from server
    if options["verbose"]: print("Loading pastiche from server... ", end="")
    r = requests.get(api.SERVER_URL + "/api/pastiche/" + tag_id)
    if r.status_code == api.STATUS_OK:
        pastiche_image = convert_image(r.content)
    else:
        raise ValueError("FATAL: something went wrong")
    
    # Show pastiche image
    plt.imshow(np.asarray(pastiche_image))
    plt.show()
