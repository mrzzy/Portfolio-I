#
# traffic_light.py
# Portfolio I - Lab 1-2
# Similates a Traffic light system with the raspberry pi
#

import lcddriver
import time
from datetime import datetime, timedelta
from gpiozero import LED

# Pinout constants
# TODO: fill this up to work 
PIN_RED_LED = 0
PIN_AMBER_LED = 0
PIN_GREEN_LED = 0


# Displays the given traffic light signal by lighting up the LED specifed by led_pin
# and writes the given message to the lcd display for the given duration seconds
def display_signal(led_pin, message, duration):
    begin = datetime.now() # record starting time
    
    # Light up red LED
    led = LED(led_pin)
    led.on()
    
    # Write message on the LCD
    lcd = lcddriver.lcd()
    lcd.lcd_display_string(message)
    
    # Sleep until the duration specified has elasped
    # skip time spent lighting up the LED and writing walk
    sleep_seconds = timedelta(seconds=duration) - (begin - datetime.now())
    time.sleep(sleep_seconds)
    
    # Cleanup: Turn off lead and clear display
    led.off()
    lcd.lcd_clear()


try:
    while True:
        # Light up green traffic light signal
        display_signal(PIN_GREEN_LED, "DO NOT WALK", 10)
        # Light up amber traffic light signal
        display_signal(PIN_AMBER_LED, "DO NOT WALK", 1)
        # Light up red traffic light signal
        display_signal(PIN_RED_LED, "DO NOT WALK", 10)
except KeyboardInterrupt:
    # cleanup
    LED(PIN_RED_LED).off()
    LED(PIN_AMBER_LED).off()
    LED(PIN_GREEN_LED).off()
    lcddriver.lcd().lcd_clear()
