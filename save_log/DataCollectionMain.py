#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import signal
import time
from time import sleep

import RPi.GPIO as GPIO
import cv2

# --------- โมดูลของคุณ ----------
import WebcamModule as wM
import DataCollectionModule as dcM
import Joynew as jsM
import MotorModule as mM

# ------------------ CONFIG ------------------
RUN_SWITCH = 23               # BCM23 (ขา 16), ON=LOW, OFF=HIGH (ใช้ PULL-UP ภายใน)
SLEEP_IDLE = 0.02             # เวลาพักตอน OFF
DEBOUNCE_MS = 100             # หน่วงกันเด้ง 100 ms

maxThrottle = 0.4
max_Turning_speed = 0.4
throttle_step = 0.02
throttle_decay = 0.05

# ------------------ STATE -------------------
exiting = False

# ------------------ GPIO SETUP --------------
def setup_gpio():
    # อย่า cleanup ที่นี่ (จะทำให้เตือน/รีเซ็ตขาโดยไม่จำเป็น)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RUN_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # ON = LOW
    print("[GPIO] Setup completed (polling + debounce).")

# ------------------ HELPERS -----------------
def stop_motor_safely(motor=None):
    try:
        if motor is not None:
            motor.move(0.0, 0.0)
    except Exception as e:
        print(f"[WARN] stop motor: {e}")

def close_cv_windows_safely(names=("IMG", "Trackbars")):
    try:
        cv2.waitKey(1)
        for name in names:
            try:
                if cv2.getWindowProperty(name, 0) >= 0:
                    cv2.destroyWindow(name)
            except cv2.error:
                pass
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[WARN] cv close: {e}")

def _stable_read(target_low=True, hold_ms=DEBOUNCE_MS):
    """
    อ่านสวิตช์ให้ 'นิ่ง' อย่างน้อย hold_ms:
    - target_low=True ต้องนิ่งที่ LOW (ON)
    - target_low=False ต้องนิ่งที่ HIGH (OFF)
    """
    want = GPIO.LOW if target_low else GPIO.HIGH
    t0 = time.time()
    while True:
        if GPIO.input(RUN_SWITCH) != want:
            t0 = time.time()  # รีสตาร์ทจับเวลาเมื่อหลุด
        if (time.time() - t0) * 1000.0 >= hold_ms:
            return True
        if exiting:
            return False
        sleep(0.005)

def wait_for_switch_on():
    """รอจน ON แบบนิ่งต่อเนื่อง DEBOUNCE_MS"""
    while not exiting:
        if GPIO.input(RUN_SWITCH) == GPIO.LOW:
            if _stable_read(target_low=True, hold_ms=DEBOUNCE_MS):
                return True
        sleep(SLEEP_IDLE)
    return False

def is_switch_off_stable():
    """เช็คว่า OFF แบบนิ่งต่อเนื่อง DEBOUNCE_MS"""
    if GPIO.input(RUN_SWITCH) == GPIO.HIGH:
        return _stable_read(target_low=False, hold_ms=DEBOUNCE_MS)
    return False

# ------------------ SIGNAL HANDLERS ----------------
def _sig_handler(signum, frame):
    global exiting
    exiting = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ------------------ MAIN -------------------
def main():
    global exiting
    setup_gpio()
    print("วิธี A: init มอเตอร์ครั้งเดียว ใช้ซ้ำทุกครั้งที่ ON (Debounce 100ms)")
    print("Ctrl+C เพื่อออกโปรแกรม")

    motor = None

    # === สร้างมอเตอร์ครั้งเดียว ===
    try:
        motor = mM.Motor(13, 5, 6, 18, 27, 22)  # L: ENA,IN1,IN2 / R: ENA,IN1,IN2
        print("[MOTOR] Initialized (one-time).")
    except Exception as e:
        print(f"[ERROR] Motor init failed: {e}")
        GPIO.cleanup()
        return

    try:
        while not exiting:
            # ---------- รอจนกว่าจะเปิดสวิตช์ (นิ่งจริง) ----------
            if not wait_for_switch_on():
                break
            if exiting:
                break

            print("[SWITCH] ON -> Start running...")

            # ---------- รีเซ็ตตัวแปรวิ่ง ----------
            current_throttle = 0.0
            record = 0
            no_js_warned = False

            # ---------- ลูปทำงานหลักเมื่อสวิตช์ ON ----------
            while not exiting:
                # ออกเมื่อ OFF นิ่งจริง
                if is_switch_off_stable():
                    break

                try:
                    joyVal = jsM.getJS() or {}
                    if not joyVal and not no_js_warned:
                        print("[JOY] ไม่พบจอยสติ๊กหรืออ่านค่าไม่ได้ — ตรวจ /dev/input/js0 และสิทธิ์ผู้ใช้ (กลุ่ม input)")
                        no_js_warned = True

                    steering = float(joyVal.get('RX', 0.0)) * max_Turning_speed
                    target_throttle = float(joyVal.get('LY', 0.0)) * maxThrottle

                    # smooth throttle (นุ่ม)
                    if target_throttle > current_throttle:
                        current_throttle = min(current_throttle + throttle_step, target_throttle)
                    elif target_throttle < current_throttle:
                        current_throttle = max(current_throttle - throttle_decay, target_throttle)

                    # กด B เพื่อบันทึก
                    if int(joyVal.get('B', 0)) == 1:
                        if record == 0:
                            print('[REC] Recording Started ...')
                        record += 1
                        sleep(0.300)

                    if record == 1:
                        img = wM.getImg(True, size=[240, 120])
                        dcM.saveData(img, steering)
                    elif record == 2:
                        dcM.saveLog()
                        record = 0

                    motor.move(current_throttle, steering)

                    cv2.waitKey(1)
                    sleep(0.005)

                except Exception as e:
                    print(f"[ERROR] run loop: {e}")
                    break

            # ---------- OFF -> หยุดอย่างปลอดภัย แต่ไม่ทำ motor.cleanup() ----------
            stop_motor_safely(motor)
            close_cv_windows_safely()
            print("[SWITCH] OFF -> Waiting for next ON...")

            # รอจน OFF มั่นคง (กันเด้ง) ก่อนวนไปสเต็ปถัดไป
            while not exiting and not is_switch_off_stable():
                sleep(SLEEP_IDLE)

    except Exception as e:
        print(f"[FATAL] Unexpected error in main: {e}")
    finally:
        try:
            stop_motor_safely(motor)
        except Exception:
            pass
        close_cv_windows_safely()
        GPIO.cleanup()
        print("[SYS] Exit.")

if __name__ == "__main__":
    main()

