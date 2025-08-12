#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import cv2
from time import sleep
import sys
import signal
import time

# โมดูลของคุณ
import WebcamModule as wM
import DataCollectionModule as dcM
import Joynew as jsM
import MotorModule as mM

# ------------------ CONFIG ------------------
RUN_SWITCH = 23            # BCM23 (ขา 16)
SLEEP_IDLE = 0.02          # หน่วงตอนสวิตช์ OFF
SWITCH_CHECK_INTERVAL = 0.01  # ตรวจสวิตช์ทุก 10ms

maxThrottle = 0.4
max_Turning_speed = 0.4
throttle_step = 0.02
throttle_decay = 0.05

# ------------------ STATE -------------------
stop_now = False
last_switch_state = False
switch_off_time = 0

# ------------------ GPIO SETUP --------------
def setup_gpio():
    """Setup GPIO pins safely"""
    try:
        GPIO.cleanup()  # ทำความสะอาดก่อน
        sleep(0.2)      # รอให้เสร็จ
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RUN_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # ON=LOW, OFF=HIGH
        print("GPIO setup completed (polling mode)")
        return True
    except Exception as e:
        print(f"GPIO setup failed: {e}")
        return False

# ------------------ HELPERS -----------------
def stop_motor_safely():
    """หยุดมอเตอร์อย่างปลอดภัย"""
    try:
        if 'motor' in globals():
            motor.move(0.0, 0.0)
    except Exception as e:
        print(f"Error stopping motor: {e}")

def close_cv_windows_safely(names=("IMG", "Trackbars")):
    """ปิดหน้าต่าง OpenCV ให้ปลอดภัย"""
    try:
        cv2.waitKey(1)
        for name in names:
            try:
                if cv2.getWindowProperty(name, 0) >= 0:
                    cv2.destroyWindow(name)
            except cv2.error:
                pass
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"[WARN] OpenCV error while closing windows: {e}")
    except Exception as e:
        print(f"[WARN] Unexpected error while closing windows: {e}")

def switch_is_on():
    """สวิตช์ ON = ต่อ GND -> LOW"""
    try:
        return GPIO.input(RUN_SWITCH) == GPIO.LOW
    except Exception as e:
        print(f"Error reading switch: {e}")
        return False

def check_switch_with_debounce():
    """ตรวจสวิตช์พร้อม debounce แบบ polling"""
    global last_switch_state, switch_off_time, stop_now
    
    current_state = switch_is_on()
    current_time = time.time()
    
    # ตรวจสอบการเปลี่ยนจาก ON -> OFF
    if last_switch_state and not current_state:
        switch_off_time = current_time
        print("[SWITCH] OFF detected -> Emergency STOP")
        stop_now = True
        stop_motor_safely()
        close_cv_windows_safely()
    
    # ตรวจสอบการเปลี่ยนจาก OFF -> ON
    elif not last_switch_state and current_state:
        # debounce: รอให้สวิตช์เสียงหาย
        if current_time - switch_off_time > 0.1:  # debounce 100ms
            print("[SWITCH] ON detected")
            stop_now = False
    
    last_switch_state = current_state
    return current_state

def cleanup_and_exit(signum=None, frame=None):
    """ทำความสะอาดและออกจากโปรแกรม"""
    global stop_now
    stop_now = True
    print("\nCleaning up and exiting...")
    
    # หยุดมอเตอร์
    stop_motor_safely()
    
    # ปิดหน้าต่าง OpenCV
    close_cv_windows_safely()
    
    # ทำความสะอาด GPIO
    try:
        GPIO.cleanup()
    except Exception:
        pass
    
    print("Clean exit completed.")
    sys.exit(0)

# ------------------ REGISTER SIGNAL HANDLERS -------
signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # Termination

# ------------------ MAIN PROGRAM ---------------
def main():
    global motor, stop_now, last_switch_state
    
    # Setup GPIO
    if not setup_gpio():
        print("Failed to setup GPIO. Exiting.")
        return
    
    # Initialize motor
    try:
        motor = mM.Motor(13, 5, 6, 18, 27, 22)
        print("Motor initialized")
    except Exception as e:
        print(f"Failed to initialize motor: {e}")
        cleanup_and_exit()
        return
    
    # Initialize switch state
    last_switch_state = switch_is_on()
    
    try:
        print("ระบบพร้อม: สวิตช์ ON เพื่อเริ่ม, OFF เพื่อหยุดทันที (Polling Mode)")
        print("กด Ctrl+C เพื่อออกจากโปรแกรม")

        while not stop_now:
            # -------- ตรวจสอบสวิตช์ --------
            switch_on = check_switch_with_debounce()
            
            # -------- รอจนกว่าจะเปิดสวิตช์ --------
            while not switch_on and not stop_now:
                stop_motor_safely()
                sleep(SLEEP_IDLE)
                switch_on = check_switch_with_debounce()

            if stop_now:
                break

            # -------- เริ่มทำงานใหม่จากต้น --------
            print("[SWITCH] ON -> Start Program Fresh")

            # รีเซ็ตตัวแปรทุกครั้งที่เริ่มใหม่
            current_throttle = 0.0
            record = 0
            loop_count = 0

            # ลูปการทำงานปกติ
            while not stop_now:
                loop_count += 1
                
                # ตรวจสวิตช์ทุก 10 loops (ประมาณทุก 50-100ms)
                if loop_count % 10 == 0:
                    switch_on = check_switch_with_debounce()
                    if not switch_on:
                        break
                
                try:
                    joyVal = jsM.getJS() or {}
                    steering = float(joyVal.get('RX', 0.0)) * max_Turning_speed
                    target_throttle = float(joyVal.get('LY', 0.0)) * maxThrottle

                    # ค่อย ๆ ปรับคันเร่ง
                    if target_throttle > current_throttle:
                        current_throttle = min(current_throttle + throttle_step, target_throttle)
                    elif target_throttle < current_throttle:
                        current_throttle = max(current_throttle - throttle_decay, target_throttle)

                    # ปุ่มบันทึก
                    if int(joyVal.get('B', 0)) == 1:
                        if record == 0:
                            print('Recording Started ...')
                        record += 1
                        sleep(0.300)

                    if record == 1:
                        img = wM.getImg(True, size=[240, 120])
                        dcM.saveData(img, steering)
                    elif record == 2:
                        dcM.saveLog()
                        record = 0

                    # ส่งค่าไปมอเตอร์
                    motor.move(current_throttle, steering)

                    cv2.waitKey(1)
                    sleep(0.005)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break

            # ออกจากลูปย่อยเพราะ OFF หรือ stop_now
            stop_motor_safely()
            if not stop_now:
                print("[SWITCH] OFF -> Waiting for next ON...")

    except Exception as e:
        print(f"Unexpected error in main: {e}")
    
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
