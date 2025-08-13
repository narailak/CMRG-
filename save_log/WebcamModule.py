# WebcamModule.py  — ใช้ Pi Camera (Picamera2) เป็นหลัก, fallback เป็น V4L2
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # กันปัญหา Qt/Wayland

import atexit
import time
import cv2

# สถานะภายใน
_picam2 = None
_cap_v4l2 = None
_started = False

def _start_picam2(width=640, height=480, fps=30):
    """พยายามเปิดกล้องผ่าน Picamera2 (libcamera)"""
    global _picam2, _started
    try:
        from picamera2 import Picamera2
        _picam2 = Picamera2()
        cfg = _picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        _picam2.configure(cfg)
        # ตั้ง FPS ถ้าได้ (ถ้ากล้องไม่รับค่าที่ตั้งก็ยังทำงานได้)
        try:
            us = int(1_000_000 / fps)
            _picam2.set_controls({"FrameDurationLimits": (us, us)})
        except Exception:
            pass
        _picam2.start()
        _started = True
        # อุ่นเครื่องสั้น ๆ
        time.sleep(0.05)
        return True
    except Exception:
        # เปิดด้วย Picamera2 ไม่ได้ (เช่น อุปกรณ์ถูกใช้งาน/แพ็กเกจหาย) -> ค่อยไป fallback
        _picam2 = None
        return False

def _start_v4l2(dev=0, width=640, height=480):
    """fallback: เปิด /dev/video0 ด้วย V4L2"""
    global _cap_v4l2, _started
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False
    # หลายรุ่นต้องบังคับ FOURCC เป็น MJPG เพื่อให้เฟรมมาได้
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    _cap_v4l2 = cap
    _started = True
    time.sleep(0.02)
    return True

def _ensure_started(width=640, height=480, fps=30):
    """เปิดกล้องครั้งแรก (Picamera2 -> V4L2) ถ้ายังไม่เปิด"""
    global _started
    if _started:
        return
    if _start_picam2(width, height, fps):
        return
    if _start_v4l2(0, width, height):
        return
    raise RuntimeError("ไม่สามารถเปิดกล้องได้ทั้ง Picamera2 และ V4L2")

def getImg(display=False, size=(480, 240)):
    """
    คืนภาพ BGR (numpy array) ขนาดตาม size
    - display=True จะ cv2.imshow('IMG', frame) (ไม่เรียก waitKey ที่นี่)
    """
    _ensure_started(max(size[0], 320), max(size[1], 240), 30)

    frame_bgr = None

    if _picam2 is not None:
        try:
            # Picamera2 คืน RGB -> แปลงเป็น BGR ให้ OpenCV
            rgb = _picam2.capture_array()
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            frame_bgr = None

    if frame_bgr is None and _cap_v4l2 is not None:
        ok, frm = _cap_v4l2.read()
        if ok:
            frame_bgr = frm

    if frame_bgr is None:
        # ให้ผู้เรียกจับ error ต่อ (ลูปจะ break ตามโค้ดหลักของคุณ)
        raise RuntimeError("อ่านเฟรมไม่ได้ (ทั้ง Picamera2 และ V4L2)")

    if size:
        frame_bgr = cv2.resize(frame_bgr, (size[0], size[1]))

    if display:
        cv2.imshow("IMG", frame_bgr)  # waitKey ให้อยู่ฝั่ง main

    return frame_bgr

def close():
    """ปล่อยทรัพยากรกล้อง/หน้าต่าง"""
    global _picam2, _cap_v4l2, _started
    try:
        if _picam2 is not None:
            try:
                _picam2.stop()
            except Exception:
                pass
            try:
                _picam2.close()
            except Exception:
                pass
    finally:
        _picam2 = None

    try:
        if _cap_v4l2 is not None:
            _cap_v4l2.release()
    finally:
        _cap_v4l2 = None

    _started = False

atexit.register(close)

