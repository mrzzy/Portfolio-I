#
# button.py
# Portfolio I - Practical
# Lab 1
# Button Control LED
#

from gpiozero import Button, LED

# Setup
# Configure Button at GPIO Pin 13
button = Button(13)
# LED at pin 19
led = LED(19)

while True:
    if button.is_pressed:
        led.on()
    else:
        led.off()
