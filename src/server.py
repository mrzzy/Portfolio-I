#
# server.py
# Style Transfer Server
#

import os
import api
import json
import styleopt

from PIL import Image
from flask import Flask, request
from multiprocessing import Process, Queue

from client import read_file
from util import convert_image, read_file

## Tasking
# Style transfer worker that runs style transfers task as defined by the 
# payloads queued
class StyleWorker:
    def __init__(self, queue=Queue(), verbose=True):
        self.queue = queue
        self.verbose = verbose

        # Setup workers process
        self.process = Process(target=self.run)
        self.process.start()
    
    # Run loop of worker
    def run(self):
        while True:
            # Perform style transfer for payload
            payload = self.queue.get()
    
            # Unpack and setup style transer for payload
            content_data, style_data, tag, settings = api.unpack_payload(payload)
            content_image = convert_image(content_data)
            style_image = convert_image(style_data)

            # Perform style transfer
            if self.verbose: print("[StyleWorker]: processing payload: ", tag)
            n_epochs = 100
            if api.SETTING_NUMBER_EPOCHS_KEY in settings:  
                n_epochs = settings[api.SETTING_NUMBER_EPOCHS_KEY]
            pastiche_image = styleopt.transfer_style(content_image, style_image,
                                                     n_epochs=n_epochs,
                                                     settings=settings,
                                                     verbose=self.verbose)
        
            # Save results of style transfer
            if self.verbose: print("[StyleWorker]: completed payload: ", tag)
            if not os.path.exists("static/pastiche"): os.mkdir("static/pastiche")
            pastiche_image.save("static/pastiche/{}.jpg".format(tag))
            
worker = StyleWorker()
            
## Routes
app = Flask(__name__, static_folder="static")
# Default route "/" displays server running message, used to check server status
@app.route("/", methods=["GET"])
def route_status():
    return app.send_static_file("status.html")

## REST API
# Rest API route "/api/style" triggers style transfer given POST body payload 
@app.route("/api/style", methods=["POST"])
def route_api_style():
    global worker
    print("[API call]: /api/style")
    payload = request.get_json()
    # Queue payload to perform style transfer on worker
    if not worker: worker = StyleWorker()
    worker.queue.put(payload)
    # Reply okay status
    return json.dumps({"sucess": True}), api.STATUS_OK, {'ContentType':'application/json'}

# Rest API route "/api/pastiche/<tag>" attempts to retrieve pastiche for the
# given tag
@app.route("/api/pastiche/<tag>", methods=["GET"])
def route_api_pastiche(tag):
    global worker
    print("[API call]: /api/pasitche for tag", tag)
    # Check if pastiche has been generated for id
    pastiche_path = "pastiche/{}.jpg".format(tag)
    
    if os.path.exists("static/" + pastiche_path):
        # Repond with pastiche for tag id 
        return app.send_static_file(pastiche_path), api.STATUS_OK
    elif not worker.process.is_alive():
        print("FATAL ERROR: style transfer worker crashed")
        worker = None
        return (json.dumps({"error": "Style transfer worker crashed"}), 
                api.STATUS_FAIL, {'ContentType':'application/json'})
    else:
        return (json.dumps({"error": "Resource not available yet"}), 
                api.STATUS_NOT_READY, {'ContentType':'application/json'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=api.SERVER_PORT)
