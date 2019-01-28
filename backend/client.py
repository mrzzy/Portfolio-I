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
from io import BytesIO

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
    parser.add_argument("-l", nargs="?", type=float, help="the learning rate to pass to the optimizer")
    parser.add_argument("-r", nargs="?", type=int, help="the resolution to perform style transfer (r x r)")
    parser.add_argument("-o", nargs="?", type=str, help="the path in which to output the generated pastiche")
    parser.add_argument("server", help="<address> the address of the style transfer server")
    parser.add_argument("content", help="path to the content image.")
    parser.add_argument("style", help="path to the style image")
    args = parser.parse_args()

    # Construct style transfer settings
    settings = {}
    if not args.c is None: setting["content_weight"] = args.c
    if not args.s is None: settings["style_weight"] = args.s
    if not args.d is None: settings["denoise_weight"] = args.d
    if not args.n is None: settings["n_epochs"] = args.n
    if not args.l is None: settings["learning_rate"] = args.l
    if not args.r is None: settings["image_shape"] = (args.r, args.r, 3)

    # Build program options
    options = {
        "content_path": args.content,
        "style_path": args.style,
        "pastiche_path": args.o if not args.o is None else "pastiche.jpg",
        "settings": settings,
        "verbose": args.v,
        "server": args.server
    }

    return options

# Display a progress bar for the given progress (1.0-0.0)
def display_progress(progress, bar_len=80):
    n_bars = int(progress * bar_len)
    n_not_bars = bar_len - n_bars
    progress_percent = progress * 100.0
    print(n_bars * "#" + n_not_bars * " " + " {:.1f}%".format(progress_percent), end="\r")


if __name__ == "__main__":
    # Read program options
    options = parse_args()
    server = options["server"] + ":" + str(api.SERVER_PORT)
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
    task_id = response.ID
    if verbose: print("Server assigns task id: ", task_id)

    # Wait for server to complete style tran + ":" + apsfer
    is_pastiche_ready = False
    while not is_pastiche_ready:
        if verbose: print("Requesting transfer status from server...")
        r = requests.get("http://" + server + "/api/status/" + task_id)
        if r.status_code == 404:
            raise Exception("FATAL: Server disowned style transfer task")
        elif r.status_code == 500:
            raise Exception("FATAL: Server encountered internal error processing style transfer task")
    
        # Read server status response
        response = api.StatusResponse.parse(r.text)
        progress = response.progress
        display_progress(progress)
        
        # Stop waiting if style transfer is completed
        if progress == 1.0: break
        time.sleep(1)

    # Retrieve pastiche from server
    if verbose: print("Requesting pastiche from server...")
    r = requests.get("http://" + server + "/api/pastiche/" + task_id)
    if r.status_code == 404:
        raise Exception("FATAL: Server disowned style transfer task")
    elif r.status_code == 500:
        raise Exception("FATAL: Server encountered internal error processing style transfer task")
    elif r.status_code == 202:
        raise Exception("FATAL: Server says pastiche not yet ready, but status request indicates that it is")
    
    # Write pastiche to disk
    pastiche_image = Image.open(BytesIO(r.content))
    pastiche_image.save(options["pastiche_path"])
