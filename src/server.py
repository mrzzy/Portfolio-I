#
# server.py
# Style Transfer Server
#

import os
import api
import uuid
import styleopt

from PIL import Image
from flask import Flask, request
from multiprocessing import Process, Queue

## Tasking
# Style transfer worker that runs style transfers task as defined by the 
# payloads queued
class TransferWorker:
    def __init__(self, queue=Queue(), verbose=True):
        self.queue = queue
        self.verbose = verbose

        # Setup workers process
        self.process = Process(target=self.run)
        self.process.start()
    
    # Enqeue a new style transfer task parameterised by the given style 
    # transfer request. Returns an uuid that uniquely identifies the task
    def enqueue(self, request):
        # Create task for request
        task_id = str(uuid.uuid4())
        task = {
            "request": request,
            "ID": task_id
        }

        self.queue.put(task)

        return task_id
        
    # Run loop of worker
    def run(self):
        while True:
            # Perform style transfer for style transfer requst
            task = self.queue.get()
            request = task["request"]
            task_id = task["ID"]
    
            # Unpack style transfer request
            content_image = request.content_image
            style_image = request.style_image
            settings = request.settings

            # Perform style transfer
            if self.verbose: print("[TransferWorker]: processing payload: ", task_id)
            pastiche_image = styleopt.transfer_style(content_image, style_image, 
                                            settings=settings,
                                            callbacks=[styleopt.callback_progress])
        
            # Save results of style transfer
            if self.verbose: print("[TransferWorker]: completed payload: ", task_id)
            if not os.path.exists("static/pastiche"): os.mkdir("static/pastiche")
            pastiche_image.save("static/pastiche/{}.jpg".format(task_id))
            
    
worker = None

# Server Routes
app = Flask(__name__, static_folder="static")
# Default route "/" displays server running message, used to check server if 
# server is running properly
@app.route("/", methods=["GET"])
def route_test():
    return app.send_static_file("test.html")

## REST API
# Rest API route "/api/style" triggers style transfer given POST style transfer
# request payload
@app.route("/api/style", methods=["POST"])
def route_api_style():
    global worker
    print("[REST]: /api/style")
    # Read style transfer request from body
    transfer_request = api.TransferRequest.parse(request.data)

    # Queue request to perform style transfer on worker
    if not worker: worker = TransferWorker()
    task_id = worker.enqueue(transfer_request)
    
    # Return response to requester
    response = api.TransferResponse(task_id)
    return response.serialise(), 200, {'ContentType':'application/json'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8989)
