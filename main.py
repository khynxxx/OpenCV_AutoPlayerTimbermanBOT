import cv2
import numpy as np
import mss
import pyautogui
import keyboard
import time
import json
import sys
import argparse
from pathlib import Path

try:
    import pygetwindow as gw
    HAS_GW = True
except:
    HAS_GW = False

H_MIN = 175
H_MAX = 210
S_MIN = 30
S_MAX = 180
V_MIN = 80
V_MAX = 220

CH_MIN = 20
CH_MAX = 40
CS_MIN = 150
CS_MAX = 255
CV_MIN = 150
CV_MAX = 255

THRESHOLD = 60
CPS = 10.0

cfg_file = "roi_config.json"


def check_window():
    title = ""
    if HAS_GW:
        try:
            w = gw.getActiveWindow()
            if w:
                title = w.title
        except:
            pass
    if not title:
        try:
            title = pyautogui.getActiveWindowTitle() or ""
        except:
            pass
    return "BlueStacks" in title or "bluestacks" in title.lower()


def grab_mask(frame, hmin, hmax, smin, smax, vmin, vmax):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,
        np.array([hmin, smin, vmin], dtype=np.uint8),
        np.array([hmax, smax, vmax], dtype=np.uint8))


def px_count(mask, x, y, w, h):
    return int(cv2.countNonZero(mask[y:y+h, x:x+w]))


def calibrate():
    print("\ncalibrating ROI...")
    print("move cursor to TOP-LEFT corner of the log and press C")
    keyboard.wait("c")
    p1 = pyautogui.position()
    print(f"got {p1}")
    time.sleep(0.3)

    print("now BOTTOM-RIGHT corner, press C again")
    keyboard.wait("c")
    p2 = pyautogui.position()
    print(f"got {p2}")
    time.sleep(0.3)

    x = min(p1.x, p2.x)
    y = min(p1.y, p2.y)
    w = abs(p2.x - p1.x)
    h = abs(p2.y - p1.y)

    if w < 20 or h < 20:
        print("too small, try again")
        return calibrate()

    data = {
        "roi": {"left": x, "top": y, "width": w, "height": h},
        "fish_hsv": {"h_min": H_MIN, "h_max": H_MAX, "s_min": S_MIN, "s_max": S_MAX, "v_min": V_MIN, "v_max": V_MAX},
        "clock_hsv": {"h_min": CH_MIN, "h_max": CH_MAX, "s_min": CS_MIN, "s_max": CS_MAX, "v_min": CV_MIN, "v_max": CV_MAX}
    }
    Path(cfg_file).write_text(json.dumps(data, indent=2))
    print(f"saved to {cfg_file}")
    return data


def load_cfg():
    p = Path(cfg_file)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"couldnt load config: {e}")
        return None


def run(cfg):
    roi = cfg["roi"]
    lft = roi["left"]
    tp  = roi["top"]
    w   = roi["width"]
    h   = roi["height"]

    fhsv = cfg.get("fish_hsv") or {"h_min": H_MIN, "h_max": H_MAX, "s_min": S_MIN, "s_max": S_MAX, "v_min": V_MIN, "v_max": V_MAX}
    chsv = cfg.get("clock_hsv") or {"h_min": CH_MIN, "h_max": CH_MAX, "s_min": CS_MIN, "s_max": CS_MAX, "v_min": CV_MIN, "v_max": CV_MAX}

    monitor = {"left": lft, "top": tp, "width": w, "height": h}
    half = w // 2
    bh = max(10, h // 6)


    sy = int(h * 0.15)
    sh = h - sy*2


    sl = (0,      sy, half-1, sh)
    sr = (half+1, sy, half-1, sh)


    ty = sy + sh - bh
    tl = (0,      ty, half-1, bh)
    tr = (half+1, ty, half-1, bh)

    print("bot running. P=pause, D=debug, C=recalibrate, Q=quit")

    paused = False
    debug = False
    side = "right"
    last = 0.0
    interval = 1.0 / CPS

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed("q"):
                print("quit")
                cv2.destroyAllWindows()
                return None

            if keyboard.is_pressed("c"):
                cv2.destroyAllWindows()
                time.sleep(0.4)
                return calibrate()

            if keyboard.is_pressed("p"):
                paused = not paused
                print("paused" if paused else "resumed")
                time.sleep(0.35)

            if keyboard.is_pressed("d"):
                debug = not debug
                print(f"debug: {debug}")
                if not debug:
                    cv2.destroyAllWindows()
                time.sleep(0.35)

            if paused or not check_window():
                time.sleep(0.08)
                continue

            raw = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

            fmask = grab_mask(frame, fhsv["h_min"], fhsv["h_max"],
                              fhsv["s_min"], fhsv["s_max"],
                              fhsv["v_min"], fhsv["v_max"])
            cmask = grab_mask(frame, chsv["h_min"], chsv["h_max"],
                              chsv["s_min"], chsv["s_max"],
                              chsv["v_min"], chsv["v_max"])

            # extra color ranges if defined
            for key in ["fish2_hsv", "fish2_yellow_hsv", "fish1_yellow"]:
                extra = cfg.get(key)
                if extra:
                    fmask = cv2.bitwise_or(fmask, grab_mask(frame,
                        extra["h_min"], extra["h_max"],
                        extra["s_min"], extra["s_max"],
                        extra["v_min"], extra["v_max"]))

            lf_trig = px_count(fmask, *tl)
            rf_trig = px_count(fmask, *tr)
            lf_scan = px_count(fmask, *sl)
            rf_scan = px_count(fmask, *sr)
            lc = px_count(cmask, *sl)
            rc = px_count(cmask, *sr)

            ld = lf_trig > THRESHOLD
            rd = rf_trig > THRESHOLD

            new_side = side

            if ld and rd:
                new_side = "left" if lf_scan <= rf_scan else "right"
            elif ld:
                new_side = "right"
            elif rd:
                new_side = "left"
            else:
                # use clock to decide when safe to switch
                if side == "right" and lc > THRESHOLD and lf_scan == 0:
                    new_side = "left"
                elif side == "left" and rc > THRESHOLD and rf_scan == 0:
                    new_side = "right"

            if debug:
                dbg = frame.copy()
                cv2.rectangle(dbg, (sl[0], sl[1]), (sl[0]+sl[2], sl[1]+sl[3]), (255,180,0), 1)
                cv2.rectangle(dbg, (sr[0], sr[1]), (sr[0]+sr[2], sr[1]+sr[3]), (255,180,0), 1)
                cv2.rectangle(dbg, (tl[0], tl[1]), (tl[0]+tl[2], tl[1]+tl[3]), (0,0,255) if ld else (0,220,0), 2)
                cv2.rectangle(dbg, (tr[0], tr[1]), (tr[0]+tr[2], tr[1]+tr[3]), (0,0,255) if rd else (0,220,0), 2)
                cv2.putText(dbg, f"L:{lf_trig} R:{rf_trig} -> {new_side}", (2,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
                cv2.imshow("debug", dbg)
                cv2.imshow("mask", fmask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return None

            now = time.time()
            if now - last >= interval:
                pyautogui.press("left" if new_side == "left" else "right")
                last = now
                side = new_side

            time.sleep(0.004)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    cfg = None
    if not args.calibrate:
        cfg = load_cfg()

    if cfg is None:
        cfg = calibrate()

    while cfg is not None:
        cfg = run(cfg)

    print("done")


if __name__ == "__main__":
    main()
