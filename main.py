import sys, time, json, argparse
from pathlib import Path

import cv2
import numpy as np
import mss
import pyautogui
import keyboard

try:
    import pygetwindow as gw
except ImportError:
    gw = None

CONFIG_PATH = Path("roi_config.json")

DEFAULT_FISH_HSV = dict(h_min=175, h_max=210, s_min=30, s_max=180, v_min=80, v_max=220)
DEFAULT_CLOCK_HSV = dict(h_min=20, h_max=40, s_min=150, s_max=255, v_min=150, v_max=255)

KEY_LEFT  = 'left'
KEY_RIGHT = 'right'

CPS            = 10.0
PRESS_INTERVAL = 1.0 / CPS

DETECTION_ZONE_FRACTION = 1.0
HIT_THRESHOLD = 60


def is_bluestacks_active() -> bool:
    title = ""
    if gw:
        try:
            aw = gw.getActiveWindow()
            title = aw.title if aw else ""
        except Exception:
            pass
    if not title:
        try:
            title = pyautogui.getActiveWindowTitle() or ""
        except Exception:
            pass
    return "BlueStacks" in title or "bluestacks" in title.lower()


def bgr_to_hsv_mask(bgr: np.ndarray,
                    h_min, h_max, s_min, s_max, v_min, v_max) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo  = np.array([h_min, s_min, v_min], dtype=np.uint8)
    hi  = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def count_pixels(mask: np.ndarray, x, y, w, h) -> int:
    return int(cv2.countNonZero(mask[y:y+h, x:x+w]))


def calibrate_roi() -> dict:
    print()
    print("╔══════════════════════════════════════════╗")
    print("║      CALIBRACIÓN DE ROI  (2 puntos)      ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("  Coloca el cursor en la ESQUINA SUPERIOR-IZQUIERDA")
    print("  del tronco (incluye un poco de margen a cada lado)")
    print("  y presiona [c] ...")
    keyboard.wait("c")
    p1 = pyautogui.position()
    print(f"  ✓ Punto 1: ({p1.x}, {p1.y})")
    time.sleep(0.35)

    print()
    print("  Ahora coloca el cursor en la ESQUINA INFERIOR-DERECHA")
    print("  y presiona [c] ...")
    keyboard.wait("c")
    p2 = pyautogui.position()
    print(f"  ✓ Punto 2: ({p2.x}, {p2.y})")
    time.sleep(0.35)

    left   = min(p1.x, p2.x)
    top    = min(p1.y, p2.y)
    width  = abs(p2.x - p1.x)
    height = abs(p2.y - p1.y)

    if width < 20 or height < 20:
        print("  ✗ Área demasiado pequeña, intenta de nuevo.\n")
        return calibrate_roi()

    cfg = {
        "roi": {"left": left, "top": top, "width": width, "height": height},
        "fish_hsv":  DEFAULT_FISH_HSV,
        "clock_hsv": DEFAULT_CLOCK_HSV,
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print()
    print(f"  ROI guardada → left={left} top={top} w={width} h={height}")
    print(f"  Archivo: {CONFIG_PATH.resolve()}")
    print()
    return cfg


def load_config() -> dict | None:
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            print(f"Config cargada: {CONFIG_PATH.resolve()}")
            return cfg
        except Exception as e:
            print(f"Error leyendo config: {e}")
    return None


def get_hsv_ranges(cfg: dict):
    fish = (
        cfg.get("fish_hsv")
        or cfg.get("fish1_blue")
        or cfg.get("fish1_hsv")
        or DEFAULT_FISH_HSV
    )
    extras = [r for r in [
        cfg.get("fish2_hsv"),
        cfg.get("fish2_yellow_hsv"),
        cfg.get("fish1_yellow"),
    ] if r is not None]

    clock = cfg.get("clock_hsv") or DEFAULT_CLOCK_HSV
    return fish, extras, clock


def run_bot(cfg: dict) -> "dict | None":
    roi   = cfg["roi"]
    left  = roi["left"]
    top   = roi["top"]
    width = roi["width"]
    height= roi["height"]

    fish_hsv, extras, clock_hsv = get_hsv_ranges(cfg)
    monitor = {"left": left, "top": top, "width": width, "height": height}

    num_blocks   = 6
    block_h      = max(10, height // num_blocks)
    half         = width // 2

    scan_top    = max(0, int(height * 0.15))
    scan_bottom = max(0, int(height * 0.15))
    scan_y = scan_top
    scan_h = height - scan_top - scan_bottom

    scan_left  = (0,      scan_y, half - 1, scan_h)
    scan_right = (half+1, scan_y, half - 1, scan_h)

    trigger_y = scan_y + scan_h - block_h
    trigger_h = block_h
    trig_left  = (0,      trigger_y, half - 1, trigger_h)
    trig_right = (half+1, trigger_y, half - 1, trigger_h)

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  BOT ACTIVO                                  ║")
    print("║  [p] pausa/resume   [d] ventana debug        ║")
    print("║  [c] re-calibrar    [q] salir                ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    paused       = False
    debug        = False
    current_side = "right"
    last_press   = 0.0

    with mss.mss() as sct:
        try:
            while True:

                if keyboard.is_pressed("q"):
                    print("Saliendo.")
                    cv2.destroyAllWindows()
                    return None

                if keyboard.is_pressed("c"):
                    cv2.destroyAllWindows()
                    time.sleep(0.4)
                    return calibrate_roi()

                if keyboard.is_pressed("p"):
                    paused = not paused
                    print("⏸  Pausado." if paused else "▶  Reanudado.")
                    time.sleep(0.4)

                if keyboard.is_pressed("d"):
                    debug = not debug
                    print(f"🔍 Debug {'ON' if debug else 'OFF'}.")
                    if not debug:
                        cv2.destroyAllWindows()
                    time.sleep(0.4)

                if paused or not is_bluestacks_active():
                    time.sleep(0.08)
                    continue

                raw   = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

                fish_mask = bgr_to_hsv_mask(frame, **fish_hsv)
                for extra in extras:
                    fish_mask = cv2.bitwise_or(
                        fish_mask, bgr_to_hsv_mask(frame, **extra)
                    )
                clock_mask = bgr_to_hsv_mask(frame, **clock_hsv)

                lf_scan = count_pixels(fish_mask,  *scan_left)
                rf_scan = count_pixels(fish_mask,  *scan_right)

                lf_trig = count_pixels(fish_mask,  *trig_left)
                rf_trig = count_pixels(fish_mask,  *trig_right)

                lc = count_pixels(clock_mask, *scan_left)
                rc = count_pixels(clock_mask, *scan_right)

                l_danger = lf_trig > HIT_THRESHOLD
                r_danger = rf_trig > HIT_THRESHOLD

                l_clock  = lc > HIT_THRESHOLD
                r_clock  = rc > HIT_THRESHOLD

                decided = current_side

                if l_danger and r_danger:
                    decided = "left" if lf_scan <= rf_scan else "right"
                elif l_danger:
                    decided = "right"
                elif r_danger:
                    decided = "left"
                else:
                    if current_side == "right" and l_clock and lf_scan == 0:
                        decided = "left"
                    elif current_side == "left" and r_clock and rf_scan == 0:
                        decided = "right"

                if debug:
                    dbg = frame.copy()
                    x1,y1,w1,h1 = scan_left
                    x2,y2,w2,h2 = scan_right
                    cv2.rectangle(dbg, (x1,y1), (x1+w1,y1+h1), (255,180,0), 1)
                    cv2.rectangle(dbg, (x2,y2), (x2+w2,y2+h2), (255,180,0), 1)
                    tx1,ty1,tw1,th1 = trig_left
                    tx2,ty2,tw2,th2 = trig_right
                    cv2.rectangle(dbg, (tx1,ty1), (tx1+tw1,ty1+th1),
                                  (0,0,255) if l_danger else (0,220,0), 2)
                    cv2.rectangle(dbg, (tx2,ty2), (tx2+tw2,ty2+th2),
                                  (0,0,255) if r_danger else (0,220,0), 2)
                    info = f"Trig L:{lf_trig} R:{rf_trig} | Scan L:{lf_scan} R:{rf_scan} -> {decided.upper()}"
                    cv2.putText(dbg, info, (2,14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
                    cv2.imshow("Timberman debug - frame", dbg)
                    cv2.imshow("Timberman debug - fish mask", fish_mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        return None

                now = time.time()
                if now - last_press >= PRESS_INTERVAL:
                    pyautogui.press(KEY_LEFT if decided == "left" else KEY_RIGHT)
                    last_press   = now
                    current_side = decided

                time.sleep(0.004)

        except KeyboardInterrupt:
            print("Interrumpido por Ctrl-C.")
            cv2.destroyAllWindows()
            return None


def main():
    parser = argparse.ArgumentParser(description="Timberman autoplayer")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Fuerza nueva calibración aunque ya exista roi_config.json"
    )
    args = parser.parse_args()

    cfg = None
    if not args.calibrate:
        cfg = load_config()

    if cfg is None:
        cfg = calibrate_roi()

    while cfg is not None:
        cfg = run_bot(cfg)

    print("Bye!")


if __name__ == "__main__":
    main()
