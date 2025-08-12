"""
- This module gets the joystick values
  and puts them in a single dictionary in realtime.
- The values can be accessed through the keys
- Tested on Flydigi Dune Fox (Bluetooth / USB, XInput mode)
"""

import pygame
from time import sleep

pygame.init()
controller = pygame.joystick.Joystick(0)
controller.init()

# ปุ่มและแกน mapping สำหรับ Flydigi Dune Fox
buttons = {
    'A': 0, 'B': 0, 'X': 0, 'Y': 0,
    'LB': 0, 'RB': 0,
    'BACK': 0, 'START': 0,
    'LS': 0, 'RS': 0,
    'DPAD_UP': 0, 'DPAD_DOWN': 0, 'DPAD_LEFT': 0, 'DPAD_RIGHT': 0,
    'LX': 0., 'LY': 0., 'RX': 0., 'RY': 0.,
    'LT': 0., 'RT': 0.
}

axiss = [0.] * 6  # แกนทั้งหมด
hat_x, hat_y = 0, 0  # d-pad

def getJS(name=''):
    global buttons, axiss, hat_x, hat_y
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axiss[event.axis] = round(event.value, 2)
        elif event.type == pygame.JOYBUTTONDOWN:
            for idx, key in enumerate(['A','B','X','Y','LB','RB','BACK','START','GUIDE','LS','RS']):
                if idx < controller.get_numbuttons() and controller.get_button(idx):
                    if key in buttons:
                        buttons[key] = 1
        elif event.type == pygame.JOYBUTTONUP:
            for idx, key in enumerate(['A','B','X','Y','LB','RB','BACK','START','GUIDE','LS','RS']):
                if idx < controller.get_numbuttons() and event.button == idx:
                    if key in buttons:
                        buttons[key] = 0
        elif event.type == pygame.JOYHATMOTION:
            hat_x, hat_y = event.value

    # แกน analog sticks (invert Y)
    buttons['LX'] = axiss[0]
    buttons['LY'] = -axiss[1]
    buttons['RX'] = axiss[3]
    buttons['RY'] = -axiss[4]

    # triggers (0..1)
    buttons['LT'] = (axiss[2] + 1) / 2
    buttons['RT'] = (axiss[5] + 1) / 2

    # D-Pad
    buttons['DPAD_LEFT']  = 1 if hat_x < 0 else 0
    buttons['DPAD_RIGHT'] = 1 if hat_x > 0 else 0
    buttons['DPAD_UP']    = 1 if hat_y > 0 else 0
    buttons['DPAD_DOWN']  = 1 if hat_y < 0 else 0

    if name == '':
        return buttons
    else:
        return buttons[name]

def main():
    print(getJS())  # ดูค่าทั้งหมด
    sleep(0.05)

if __name__ == '__main__':
    while True:
        main()
