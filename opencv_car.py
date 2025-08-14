#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, signal, atexit
import cv2, numpy as np
from collections import deque
import RPi.GPIO as GPIO

# ---- โมดูลของคุณ (ต้องมีใน PYTHONPATH เดียวกัน) ----
import WebcamModule as wM
import MotorModule as mM
import Joynew as jsM

# ====== CONFIG ======
USE_DATA_LOGGING = False
if USE_DATA_LOGGING:
    import DataCollectionModule as dcM

RUN_SWITCH   = 23
DEBOUNCE_MS  = 200

# ----- UI auto-off if headless -----
SHOW_WINDOWS = True
if not os.environ.get("DISPLAY"):
    SHOW_WINDOWS = False
    print("[UI] No DISPLAY detected -> SHOW_WINDOWS=False")

RECORD_OUT  = False
OUT_PATH    = "lane_follow_all.mp4"

# ====== Lane & Vision Params ======
# ใช้เฟรมรับเข้าที่ 1080x480 ตามที่ขอ
FRAME_W, FRAME_H = 1080, 480

# แนะนำให้ warp เต็มเฟรมเพื่อ invert กลับง่าย (1:1)
WARP_W, WARP_H   = 1080, 480

# ---------- Perspective points ----------
# จุดในภาพต้นฉบับ (1080x480) ที่คุณกำหนด
TL = (160,  30)
BL = (0,   240)
TR = (920,  30)
BR = (1080, 240)

# จัดลำดับ src/dst ให้ตรงรูปแบบ (ซ้ายบน, ขวาบน, ซ้ายล่าง, ขวาล่าง)
SRC_PTS = np.float32([TL, TR, BL, BR])
DST_PTS = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])

# ----- ค่าควบคุมรถ -----
BASE_SPEED = 0.10
TURN_GAIN  = -0.2
SLOWDOWN_K = 0.8
MIN_SPEED  = 0.05
KP_STEER, KD_STEER, MAX_STEER = 0.6, 0.10, 1.0
SMOOTH_WIN, MISS_LIMIT = 5, 8

# ====== STATES ======
master_enable = False   # จากสวิตช์ภายนอก (LOW=enable)
run_active    = False   # ปุ่ม B
last_b_state, last_b_time = 0, 0.0
B_COOLDOWN_SEC = 0.30

record_flag, last_x_state, last_x_time = 0, 0, 0.0
X_COOLDOWN_SEC = 0.30

exiting = False
USE_POLLING, _last_state, _last_change = False, None, 0.0

# ====== Motor ======
motor = mM.Motor(13, 5, 6, 18, 27, 22)

def stop_motor_safely():
    try:
        motor.move(0.0, 0.0)
    except Exception as e:
        print("[WARN] stop motor:", e)

# ====== Switch / GPIO ======
def _set_master_from_pin():
    global master_enable
    state = GPIO.input(RUN_SWITCH)
    master_enable = (state == GPIO.LOW)
    print(f"[SWITCH] {'ENABLE' if master_enable else 'DISABLE'} (BCM{RUN_SWITCH}={'LOW' if master_enable else 'HIGH'})")

def _switch_cb(channel):
    _set_master_from_pin()
    if not master_enable:
        stop_motor_safely()

def setup_gpio():
    global USE_POLLING, _last_state, _last_change
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    try:
        GPIO.remove_event_detect(RUN_SWITCH)
    except Exception:
        pass

    GPIO.setup(RUN_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(0.01)

    try:
        GPIO.add_event_detect(RUN_SWITCH, GPIO.BOTH, callback=_switch_cb, bouncetime=DEBOUNCE_MS)
        print(f"[GPIO] interrupt ready on BCM{RUN_SWITCH}")
        _set_master_from_pin()
    except RuntimeError as e:
        USE_POLLING = True
        _last_state  = GPIO.input(RUN_SWITCH)
        _last_change = time.time()
        _set_master_from_pin()
        print("[GPIO] add_event_detect failed -> polling. reason:", e)

# ====== Signal ======
def _sig_handler(signum, frame):
    global exiting
    exiting = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ====== CV Helpers ======
def get_perspective_mats():
    M    = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    Minv = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)
    return M, Minv

def preprocess_binary(img_bgr):
    """
    คืน binary mask ของเลน (รวม white/yellow + edge) พร้อมตัด ROI ครึ่งล่างของภาพ
    """
    img = cv2.GaussianBlur(img_bgr, (5,5), 0)

    # สี
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H,L,S = cv2.split(hls)
    R = img[:,:,2]
    white  = cv2.bitwise_or(cv2.inRange(L,200,255), cv2.inRange(R,200,255))
    yellow = cv2.inRange(hls, (15,80,80), (35,255,255))

    # ขอบ
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    edges  = cv2.inRange(cv2.convertScaleAbs(sobelx), 40, 255)

    # รวม
    binary = cv2.bitwise_or(white, yellow)
    binary = cv2.bitwise_or(binary, edges)

    # ROI: ใช้ครึ่งล่าง (เริ่มที่ 0.50*H)
    #mask = np.zeros_like(binary)
    #roi = np.array([[(0,int(FRAME_H*0.50)), (FRAME_W,int(FRAME_H*0.50)), (FRAME_W,FRAME_H), (0,FRAME_H)]], np.int32)
    #cv2.fillPoly(mask, roi, 255)
    #binary = cv2.bitwise_and(binary, mask)

    # ปิดรูเล็กๆ
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)
    return binary

def warp_binary(bin_img, M):
    return cv2.warpPerspective(bin_img, M, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)

def sliding_window_fit(binary_warped, nwindows=12, margin=260, minpix=80):
    """
    ปรับ margin ให้เหมาะกับภาพกว้าง 1080 (จากเดิม ~60 เมื่อ 240 กว้าง)
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0]//2
    leftx_base  = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint

    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for _ in range(nwindows):
        win_y_low  = binary_warped.shape[0] - (_+1)*window_height
        win_y_high = binary_warped.shape[0] - _*window_height
        win_xleft_low  = leftx_current  - margin
        win_xleft_high = leftx_current  + margin
        win_xright_low = rightx_current - margin
        win_xright_high= rightx_current + margin

        good_left = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) &
                     (nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]
        good_right= ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) &
                     (nonzerox>=win_xright_low) & (nonzerox<win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds  = np.concatenate(left_lane_inds)  if left_lane_inds  else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    leftx, lefty   = nonzerox[left_lane_inds],  nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # ภาพใหญ่ขึ้น -> เพิ่มเกณฑ์จุดขั้นต่ำ
    left_fit  = np.polyfit(lefty,  leftx,  2) if leftx.size  > 300 and lefty.size  > 300 else None
    right_fit = np.polyfit(righty, rightx, 2) if rightx.size > 300 and righty.size > 300 else None
    return left_fit, right_fit

def draw_lane_overlay(orig, warped_bin, lf, rf, Minv):
    out = orig.copy()
    h, w = warped_bin.shape[:2]
    ploty = np.linspace(0, h-1, h).astype(np.int32)
    color = np.zeros((h, w, 3), np.uint8)

    if lf is not None and rf is not None:
        lx = (lf[0]*ploty**2 + lf[1]*ploty + lf[2]).astype(np.int32)
        rx = (rf[0]*ploty**2 + rf[1]*ploty + rf[2]).astype(np.int32)
        pts_left  = np.array([np.transpose(np.vstack([lx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rx, ploty])) )])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color, [pts.astype(np.int32)], (0,255,0))
        cv2.polylines(color, [pts_left.astype(np.int32)],  False, (255,0,0),  10)
        cv2.polylines(color, [pts_right.astype(np.int32)], False, (0,0,255), 10)

    unwarped = cv2.warpPerspective(color, Minv, (orig.shape[1], orig.shape[0]))
    return cv2.addWeighted(out, 1.0, unwarped, 0.35, 0)

def compute_offset(px_bin, lf, rf):
    if lf is None or rf is None:
        return None
    h = px_bin.shape[0]
    y = h - 1
    lx = lf[0]*y*y + lf[1]*y + lf[2]
    rx = rf[0]*y*y + rf[1]*y + rf[2]
    lane_c = (lx + rx) / 2.0
    img_c  = px_bin.shape[1] / 2.0
    return float(lane_c - img_c)

def steer_from_offset(off, prev=0.0, dt=0.04):
    if off is None:
        off = 0.0
    # normalize offset ด้วยครึ่งความกว้างภาพ warp
    e = np.clip(off / (WARP_W * 0.5), -1.5, 1.5)
    de = (e - prev) / max(dt, 1e-3)
    s = np.clip(KP_STEER*e + KD_STEER*de, -MAX_STEER, MAX_STEER)
    return float(s), float(e)

# ====== Cleanup ======
def cleanup():
    try: stop_motor_safely()
    except: pass
    try: cv2.destroyAllWindows()
    except: pass
    try: wM.close()
    except: pass
    try: GPIO.cleanup()
    except: pass
    print("[SYS] Exit.")
atexit.register(cleanup)

# ====== MAIN LOOP ======
def main():
    global run_active, last_b_state, last_b_time
    global record_flag, last_x_state, last_x_time
    global USE_POLLING, _last_state, _last_change

    print("[BOOT] setup_gpio() ...")
    setup_gpio()

    print("[BOOT] get_perspective_mats() ...")
    M, Minv = get_perspective_mats()

    left_buf, right_buf = deque(maxlen=SMOOTH_WIN), deque(maxlen=SMOOTH_WIN)
    err_prev = 0.0
    miss = 0
    t_prev = time.time()
    last_heartbeat = 0.0

    writer = None
    if RECORD_OUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUT_PATH, fourcc, 30, (FRAME_W, FRAME_H))
        print("[BOOT] video writer ready:", OUT_PATH)

    print("[READY] SWITCH: LOW=ENABLE/HIGH=DISABLE | B: RUN/PAUSE | q: quit")
    while not exiting:
        # Fallback polling
        if USE_POLLING:
            nowp = time.time()
            state = GPIO.input(RUN_SWITCH)
            if state != _last_state and (nowp - _last_change)*1000.0 >= DEBOUNCE_MS:
                _last_change = nowp
                _last_state = state
                _switch_cb(RUN_SWITCH)

        # Joystick toggle run/pause
        joy = jsM.getJS()
        b = joy.get('B', 0)
        now = time.time()
        if b == 1 and last_b_state == 0 and (now - last_b_time) > B_COOLDOWN_SEC:
            run_active = not run_active
            last_b_time = now
            print(f"[JOY] {'RUN' if run_active else 'PAUSE'} by B")
            if not run_active:
                stop_motor_safely()
        last_b_state = b

        # Heartbeat
        if now - last_heartbeat > 1.0:
            print(f"[HB] master_enable={master_enable} run_active={run_active}")
            last_heartbeat = now

        # Camera frame (ทนกรณีชั่วคราวไม่ได้ภาพ)
        try:
            frame = wM.getImg(display=False, size=(FRAME_W, FRAME_H))  # ต้องคืน BGR 1080x480
        except Exception as e:
            print("[CAM] read FAIL, retry in 0.3s ->", e)
            stop_motor_safely()
            time.sleep(0.3)
            continue

        can_run = (master_enable and run_active)

        if can_run:
            # 1) Binary & Warp
            binary = preprocess_binary(frame)
            warped = warp_binary(binary, M)

            # 2) Fit lanes
            lf, rf = sliding_window_fit(warped, nwindows=12, margin=260, minpix=80)
            if lf is not None:
                left_buf.append(lf)
            if rf is not None:
                right_buf.append(rf)

            if len(left_buf) and len(right_buf):
                miss = 0
                lf = np.mean(np.array(left_buf), axis=0)
                rf = np.mean(np.array(right_buf), axis=0)

                # 3) Overlay
                overlay  = draw_lane_overlay(frame, warped, lf, rf, Minv)

                # 4) Control
                offset   = compute_offset(warped, lf, rf)
                t_now = time.time()
                dt = t_now - t_prev
                t_prev = t_now

                steer, err_prev = steer_from_offset(offset, prev=err_prev, dt=dt)
                turn  = float(np.clip(steer * TURN_GAIN, -1.0, 1.0))
                speed = float(np.clip(BASE_SPEED * (1.0 - SLOWDOWN_K*abs(turn)), MIN_SPEED, 1.0))
                motor.move(speed, turn, t=0)

                # UI
                if SHOW_WINDOWS:
                    ui = overlay.copy()
                    cv2.putText(ui, f"RUN turn={turn:+.2f} speed={speed:.2f}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)
                    cv2.imshow("lane_follow_all", ui)
                if writer:
                    writer.write(overlay)
            else:
                miss += 1
                motor.move(0.1, 0.0, t=0)
                if SHOW_WINDOWS:
                    cv2.imshow("lane_follow_all", frame)
                if miss > MISS_LIMIT:
                    left_buf.clear()
                    right_buf.clear()
                    err_prev = 0.0
        else:
            stop_motor_safely()
            if SHOW_WINDOWS:
                idle = frame.copy()
                status = ("DISABLED by SWITCH" if not master_enable else "PAUSED (press B)")
                cv2.putText(idle, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.imshow("lane_follow_all", idle)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    if writer:
        writer.release()

# --- run ---
if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()


