import WebcamModule as wM
import DataCollectionModule as dcM
import Joynew as jsM
import MotorModule as mM
import cv2
from time import sleep

maxThrottle = 0.4  # ความเร็วสูงสุด
max_Turning_speed = 0.4
motor = mM.Motor(13, 5, 6, 18, 27, 22)

record = 0
current_throttle = 0.0        # ความเร็วปัจจุบัน
throttle_step = 0.02          # ความเร็วที่เพิ่มต่อรอบ
throttle_decay = 0.05         # ลดความเร็วเมื่อปล่อย

while True:
    joyVal = jsM.getJS()

    steering = joyVal['RX'] * max_Turning_speed
    target_throttle = joyVal['LY'] * maxThrottle  # ค่าที่ต้องการไปถึง

    # ควบคุมให้ค่อย ๆ ขึ้น
    if target_throttle > current_throttle:
        current_throttle = min(current_throttle + throttle_step, target_throttle)
    elif target_throttle < current_throttle:
        # ลดลงไวกว่าเพิ่ม (ปล่อยคันเร่ง/ถอยหลัง)
        current_throttle = max(current_throttle - throttle_decay, target_throttle)

    # ปุ่มบันทึก
    if joyVal['B'] == 1:
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
