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
#from pirc522 import RFID
from gpiozero import Button, LED

FIREBASE_DB_URL = "https://cardreader-93045.firebaseio.com/"

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
                (error, uid) = rdr.anticoll()
                if not error:
                    rfid = "".join(uid)
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
        # Add RFID hash to known hashes
        id_hash = self.hashify(rfid)
        self.known_hashes.append(id_hash)
        
        if commit:
            # Store known hashes in the cloud
            self.db.store('authenticator-hashes', self.known_hashes)
            self.db.commit()
            

if __name__ == "__main__":
    red_led = LED(21)
    gren_led = LED(20)
    button = Button(26)

    button.when_pressed = red_led.on
