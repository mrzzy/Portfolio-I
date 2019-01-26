#
# Pastiche
# REST API 
#

import json
import uuid
from abc import ABC, abstractmethod
from util import encode_image, decode_image


# Represents a style transfer request
class TransferRequest:
    CONTENT_IMAGE_KEY = "content_image"
    STYLE_IMAGE_KEY = "style_image"
    SETTINGS_KEY = "settings"
    
    # Construct a new Transfer Request for the given content and style image 
    # and style transfer settings
    def __init__(self, content_image, style_image, settings):
        self.content_image = content_image
        self.style_image = style_image
        self.settings = settings

    # Parse a style transfer request from the given json
    @classmethod
    def parse(cls, request_json):
        # Parse payload
        payload = json.loads(request_json)
        # Extract content & style images
        content_image = decode_image(
            payload[cls.CONTENT_IMAGE_KEY])
        style_image = decode_image(
            payload[cls.STYLE_IMAGE_KEY])
        # Extract settings
        settings = payload[cls.SETTINGS_KEY]
    
        return cls(content_image, style_image, settings)

    # Serialise this style transfer request to network transmittable JSON
    def serialise(self):
        contents = {
            self.CONTENT_IMAGE_KEY: encode_image(self.content_image),
            self.STYLE_IMAGE_KEY: encode_image(self.style_image),
            self.SETTINGS_KEY: self.settings
        }
    
        return json.dumps(contents)

# Represents a style transfer response
class TransferResponse:
    ID_KEY = "id"
    
    # Construct a style transfer response given ID
    def __init__(self, ID):
        self.ID = ID 
    
    @classmethod
    # Parse a style transfer response from the given json
    def parse(cls, response_json):
        contents = json.loads(response_json)
        ID = contents[cls.ID_KEY]
        
        return cls(ID)
        
    
    # Serialise this style transfer response to network transmittable JSON
    def serialise(self):
        contents = {
            self.ID_KEY: self.ID
        }
    
        return json.dumps(contents)


# Represents a style transfer status response
class StatusResponse:
    PROGRESS_KEY = "progress"
    
    # Construct a style transfer response given current progress (0.0 - 1.0)
    def __init__(self, progress):
        self.progress = progress 
    
    @classmethod
    # Parse a style transfer status request from the given json
    def parse(cls, response_json):
        contents = json.loads(response_json)
        progress = contents[cls.PROGRESS_KEY]
        
        return cls(progress)
    
    # Serialise this style transfer status response to network transmittable JSON
    def serialise(self):
        contents = {
            self.PROGRESS_KEY: self.progress
        }
        return json.dumps(contents)
