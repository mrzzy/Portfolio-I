#
# reader.py
# Card Reader 
# QNS 1 
# P1 CA2
#

import requests
import json
import hashlib
import atexit
import RPi.GPIO as GPIO
import time
from pirc522 import RFID
from multiprocessing import Pool, Manager

FIREBASE_DB_URL = "https://cardreader-93045.firebaseio.com/"

GPIO.setmode(GPIO.BCM)

# Represents a scanner for RFID cards
class Scanner:
    def __init__(self):
        self.backend = RFID()

    # Waits for an card to be placed on the rfid reader
    # and reads the RFID on the card
    # Returns the RFID on the card as a string
    def read(self):
        while True:
            print("waiting for card...")
            # Wait for a card tag to be place on the tag
            self.backend.wait_for_tag()
            # Request card tag
            (error, tag_type) = self.backend.request()
            if not error:
                # Retreive RFID UID for card tag 
                (error, uid) = self.backend.anticoll()
                if not error:
                    rfid = "".join([str(idf) for idf in uid])
                    print("read rfid: ", rfid)
                    return rfid

# Represents a remote key value database backed by a firebase
class RemoteDB:
    def __init__(self, storage_file="db.json"):
        self.storage_url = "{}/{}".format(FIREBASE_DB_URL, storage_file)
        self.data = {} #  Lazy load data only on retrieve
        self.hasChanges = False
    
    # Commit changes made to local data into the remote database 
    def commit(self):
        if self.hasChanges:
            self.unlink()
            data_json = json.dumps(self.data)
            requests.post(self.storage_url, data_json)
            self.hasChanges = False
    
    # Load the data from the remote database,
    def load(self):
        # Commit changes if any since any local changes will overwritten on 
        # update
        self.commit()
        
        # Retrieve data from remote database
        r = requests.get(self.storage_url)
        if r.status_code == 200:
            self.data = list(r.json().values())[0]

    # Stores the given key and value pair into the remote database
    # Returns True if storage successful, otherwise false
    def store(self, key, value):
        self.data[key] = value
        self.hasChanges = True
    
    # Retrieves the value for the given key and returns its
    # Raises KeyError if no value is found for the given key
    def retrieve(self, key):
        # Check if key in local data, if not load remote data
        if not key in self.data:
            self.load()
        return self.data[key]
    
    # Delete the key value for the given key
    def delete(self, key):
        del self.data[key]
        self.hasChanges = True

    # Delete all data on the remote database
    def unlink(self):
        requests.delete(self.storage_url)
    
# Represents Authenticator which would decide whether to accept or reject an RFID
class Authenticator:
    # Create a new authenticator object 
    # The restore flag detemine if the authenticator restores its cloud backed 
    # state
    def __init__(self, restore=True):
        self.known_hashes = [] # List of accepted hashes
        self.db = RemoteDB() # Clound backed DB
        if restore:
            try:
                self.known_hashes = self.db.retrieve("authenticator-hashes")
            except KeyError:
                print("Failed to restore...")
            
    # Convert the given str into a SHA224 hash and returns it
    def hashify(self, s):
        # Convert rfid to hash
        digest = hashlib.sha224(s.encode('utf-8')).hexdigest()
        return digest
        
    # Verifies the given RFID is valid: part of the registered RFIDs
    # Returns True if the RFID is valid, otherwise false
    def verify(self, rfid):
        # Check if part of registered RFIDs
        return True if self.hashify(rfid) in self.known_hashes else False

    # Register the RFID into the of reconignised RFIDs
    # Commit specifies whether to commit state to the cloud
    def register(self, rfid, commit=True):
        if not rfid in self.known_hashes:
            # Add RFID hash to known hashes
            id_hash = self.hashify(rfid)
            self.known_hashes.append(id_hash)
            
            if commit:
                # Store known hashes in the cloud
                self.db.store('authenticator-hashes', self.known_hashes)
                self.db.commit()
                
# Represents an GPIO based button
class Button:
    # Create a new button on the given PIN
    def __init__(self, pin, handlers):
        self.handlers = handlers
        # Setup button pin
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.pin = pin
    
        # Listen for button events asyncronously
        pool = Pool(1)
        pool.apply_async(self.listen)
        
    
    # Listen for button events
    def listen(self):
        while True:
            # Button is press when state is low
            state = GPIO.input(self.pin)
            if state == GPIO.LOW: self.trigger()

    # Trigger the button, running all its event handlers
    def trigger(self):
        for handler in self.handlers: handler()

# Rerpresents an GPIO based LED
class LED:
    # Create a new LED on the given PIN
    def __init__(self, pin):
        # Setup LED pin
        GPIO.setup(pin, GPIO.OUT)
        self.pin = pin

    # Turn on the LED 
    def on(self):
        GPIO.output(self.pin, GPIO.HIGH)

    # Turn off the LED 
    def off(self):
        GPIO.output(self.pin, GPIO.LOW)
        

if __name__ == "__main__":
    green_led = LED(20)
    red_led = LED(21)
            
    def handle_press():
        print("Button pressed!")
    
    button = Button(19, handlers=[ handle_press ])

    while True:
        green_led.on()
        red_led.off()
        time.sleep(1)
        red_led.on()
        green_led.off()
        time.sleep(1)

    #while True: # Run Loop
    #    rfid = scanner.read()
    #    if button.is_pressed:
    #        # Registration mode
    #        auth.register(rfid)
    #        print("registered. {}".format(rfid))
    #    else:
    #        # Check mode
    #        if auth.verify(rfid):
    #            green_led.on()
    #            time.sleep(5)
    #            green_led.off()
    #            print("Approved.")
    #        else:
    #            red_led.on()
    #            time.sleep(5)
    #            red_led.off()
    #            print("Rejected.")
    #    
                
        
        
