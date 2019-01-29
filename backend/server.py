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
from multiprocessing import Process, Queue, Manager

## Tasking
# Style transfer worker that runs style transfers task as defined by the 
# payloads queued
class TransferWorker:
    def __init__(self, queue=Queue(), verbose=True):
        self.queue = queue
        self.verbose = verbose
        
        # Setup shared style transfer process log
        manager = Manager()
        self.log = manager.dict()

        # Setup worker process
        self.process = Process(target=self.run)
        self.process.start()

        # Setup directory to generated pastiches
        if not os.path.exists("static/pastiche"): os.mkdir("static/pastiche")
    
    # Enqeue a new style transfer task parameterised by the given style 
    # transfer request. Returns an uuid that uniquely identifies the task
    def enqueue(self, request):
        # Create task for request
        task_id = str(uuid.uuid4())
        task = {
            "request": request,
            "ID": task_id
        }

        # Queue task for style transfer
        self.log[task_id] = 0.0
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
            # Callback to record status of style transfer in worker log
            def callback_status(graph, feed, i_epoch):
                n_epoch = graph.settings["n_epochs"]
                self.log[task_id] = i_epoch / n_epoch
                
            if self.verbose: print("[TransferWorker]: processing task: ", task_id)
            try:
                pastiche_image = styleopt.transfer_style(content_image, style_image, 
                                                settings=settings,
                                                callbacks=[styleopt.callback_progress,
                                                           styleopt.callback_tensorboard,
                                                           callback_status])
            except Exception as e:
                # Style transfer failed for some reason
                print("[TransferWorker]: FATAL: style transfer failed for task:",
                      task_id)
                print(repr(e))
                
                self.log[task_id] = -1.0 # Mark failure for task in log
                continue # Abadon and work on next job
        
            # Save results of style transfer
            if self.verbose: print("[TransferWorker]: completed payload: ", task_id)
        
            pastiche_image.save("static/pastiche/{}.jpg".format(task_id))

    # Check the status of the worker task specified by task_id
    # Returns None if no task for the given task_id is found
    # Returns -1.0 if style transfer task failed for some reason
    def check_status(self, task_id):
        if not task_id in self.log: return None
        else: return self.log[task_id]
    
worker = TransferWorker()

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
    print("[REST]: /api/style")
    # Read style transfer request from body
    transfer_request = api.TransferRequest.parse(request.data)

    # Queue request to perform style transfer on worker
    task_id = worker.enqueue(transfer_request)
    
    # Return response to requester
    response = api.TransferResponse(task_id)
    return response.serialise(), 200, {'ContentType':'application/json'}

# Rest API route "/api/status" retrieves the current status of style transfer
# for the given task_id.
@app.route("/api/status/<task_id>", methods=["GET"])
def route_api_status(task_id):
    print("[REST]: /api/status")

    # Query work current status 
    progress = worker.check_status(task_id)
    if progress == None:
        status_code = 404 # Task for the given ID not found
    elif progress == -1.0:
        status_code = 500 # Internal server error in style transfer
    else:
        status_code = 200

    # Return status response to request 
    response = api.StatusResponse(progress)
    return response.serialise(), status_code, {'ContentType':'application/json'}

# Rest API route "/api/pastiche" retrieves the pastiche 
# for the given task_id.
@app.route("/api/pastiche/<task_id>", methods=["GET"])
def route_api_pastiche(task_id):
    print("[REST]: /api/pastiche")

    # Query work current status 
    progress = worker.check_status(task_id)
    if progress == None:
        status_code = 404 # Task for the given ID not found
        return "", status_code
    elif progress == -1.0:
        status_code = 500 # Internal server error in style transfer
        return "", status_code
    elif progress >= 0.0 and progress < 1.0:  
        status_code = 202 # Style transfer genrated pastiche not yet ready
        return "", status_code
    else:
        status_code = 200
        return app.send_static_file("pastiche/{}.jpg".format(task_id)), status_code

# Cross origin pain in the ass
@app.after_request 
def handle_cors(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=api.SERVER_PORT)
