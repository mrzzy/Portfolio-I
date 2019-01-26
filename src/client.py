#
# client.py
# Style Transfer Client
#

import api
import time
import requests
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from util import decode_image

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
    parser.add_argument("-r", nargs="?", type=int, help="the resolution to perform style transfer (r x r)")
    parser.add_argument("server", help="<address>:<port> the address and port of the style transfer server")
    parser.add_argument("content", help="path to the content image.")
    parser.add_argument("style", help="path to the style image")
    args = parser.parse_args()

    # Construct style transfer settings
    settings = {}
    if not args.c is None: setting["content_weight"] = args.c
    if not args.s is None: settings["style_weight"] = args.s
    if not args.d is None: settings["denoise_weight"] = args.d
    if not args.n is None: settings["n_epochs"] = args.n
    if not args.r is None: settings["image_shape"] = (args.r, args.r, 3)

    # Build program options
    options = {
        "content_path": args.content,
        "style_path": args.style,
        "settings": settings,
        "verbose": args.v,
        "server": args.server
    }

    return options

if __name__ == "__main__":
    # Read program options
    options = parse_args()
    server = options["server"]
    verbose = options["verbose"]
    
    # Read content & style images
    content_image = Image.open(options["content_path"])
    style_image = Image.open(options["style_path"])

    # Trigger style transfer by sending request to server
    settings = options["settings"]
    if verbose: print("Senting style transfer request to server...")
    request = api.TransferRequest(content_image, style_image, settings)
    r = requests.post("http://" + server + "/api/style", data=request.serialise());
    response = api.TransferResponse.parse(r.text)
    if verbose: print("Server assigns task id: ", response.ID)
