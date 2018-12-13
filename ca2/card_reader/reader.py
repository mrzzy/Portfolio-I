#
# reader.py
# Card Reader 
# QNS 1 
# P1 CA2
#

from pirc522 import RFID

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

if __name__ == "__main__":
    scanner = Scanner()
    while True:
        scanner.read()
