"""
Autoplayer para Timberman en BlueStacks.
- Detección por HSV de 2 tipos de peces (obstáculos) + reloj
- Calibrador interactivo de ROI (guarda en roi_config.json)
- Hotkeys: 'p' pausa/reanuda, 'q' salir
"""

import time, json, sys
from pathlib import Path
import cv2, numpy as np, pyautogui, keyboard
try:
    import pygetwindow as gw
except Exception:
    gw = None

CONFIG_PATH = Path("roi_config.json")

# --- HSV ranges ---
# Pez 1 (cuerpo azul + espinas amarillas)
FISH1_BLUE = {
    "h_min": 95, "h_max": 120,
    "s_min": 80, "s_max": 255,
    "v_min": 80, "v_max": 255
}
FISH1_YELLOW = {
    "h_min": 20, "h_max": 35,
    "s_min": 120, "s_max": 255,
    "v_min": 150, "v_max": 255
}

# Pez 2 
FISH2_HSV_RANGE = {
    "h_min": 100, "h_max": 120,
    "s_min": 40, "s_max": 255,
    "v_min": 40, "v_max": 255
}

# Reloj amarillo
CLOCK_HSV_RANGE = {
    "h_min": 20, "h_max": 45,
    "s_min": 120, "s_max": 255,
    "v_min": 150, "v_max": 255
}

# Teclas 
KEY_LEFT = 'left'
KEY_RIGHT = 'right'

# Velocidad
CPS = 17.0
PRESS_INTERVAL = 0.7 / CPS
BLOCK_ROWS = 2
HIT_PIXEL_THRESHOLD = 100

def is_bluestacks_active():
    if gw:
        try:
            aw = gw.getActiveWindow()
            if aw and 'BlueStacks' in (aw.title or ''):
                return True
        except: pass
    try:
        title = pyautogui.getActiveWindowTitle()
        if title and 'BlueStacks' in title:
            return True
    except: pass
    return False

def calibrate_roi_interactive():
    print("\nCalibración: señala las 4 esquinas del tronco (no de la ventana).")
    coords = []
    for i, name in enumerate(["sup-izq", "sup-der", "inf-izq", "inf-der"], start=1):
        print(f"[{i}/4] Mueve cursor a esquina {name} y presiona 'c'...")
        keyboard.wait('c')
        pos = pyautogui.position()
        print(f" -> registrado: {pos}")
        coords.append((pos.x, pos.y))
        time.sleep(0.2)
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    roi = {"left": min(xs), "top": min(ys),
           "width": max(xs)-min(xs), "height": max(ys)-min(ys)}
    roi['center_x'] = roi['left'] + roi['width']//2
    roi['block_h'] = max(8, roi['height'] // BLOCK_ROWS)
    cfg = {"roi": roi,
           "fish1_blue": FISH1_BLUE,
           "fish1_yellow": FISH1_YELLOW,
           "fish2_hsv": FISH2_HSV_RANGE,
           "clock_hsv": CLOCK_HSV_RANGE}
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Configuración guardada en {CONFIG_PATH.resolve()}")
    return cfg

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return None

def mask_hsv_from_bgr(bgr_img, h_min, h_max, s_min, s_max, v_min, v_max):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def mask_present(mask, rect, threshold=HIT_PIXEL_THRESHOLD):
    x, y, w, h = rect
    sub = mask[y:y+h, x:x+w]
    count = int(cv2.countNonZero(sub))
    return count > threshold, count

def run_bot(cfg):
    roi = cfg['roi']
    fish1_blue, fish1_yellow = cfg['fish1_blue'], cfg['fish1_yellow']
    fish2, clk = cfg['fish2_hsv'], cfg['clock_hsv']
    left, top, width, height = roi['left'], roi['top'], roi['width'], roi['height']
    block_h = roi['block_h']

    left_block = (0, max(0, height - 2*block_h - 10), width//2 - 2, block_h)
    right_block = (width//2 + 2, max(0, height - 2*block_h - 10), width//2 - 2, block_h)

    print("\nBot iniciado. Hotkeys: 'p' pausa/resume, 'q' salir.")
    paused, current_side, last_press = False, 'right', 0.0

    try:
        while True:
            if keyboard.is_pressed('q'):
                print("Saliendo.")
                break
            if keyboard.is_pressed('p'):
                paused = not paused
                print("Pausado." if paused else "Reanudado.")
                time.sleep(0.5)
            if paused or not is_bluestacks_active():
                time.sleep(0.1)
                continue

            img = pyautogui.screenshot(region=(left, top, width, height))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Fish1 = azul + amarillo
            fish1_mask_blue   = mask_hsv_from_bgr(frame, **fish1_blue)
            fish1_mask_yellow = mask_hsv_from_bgr(frame, **fish1_yellow)
            fish1_mask        = cv2.bitwise_or(fish1_mask_blue, fish1_mask_yellow)

            # Fish2
            fish2_mask = mask_hsv_from_bgr(frame, **fish2)

            # Combinar peces
            fish_mask = cv2.bitwise_or(fish1_mask, fish2_mask)

            # Reloj
            clock_mask = mask_hsv_from_bgr(frame, **clk)

            left_has, left_count   = mask_present(fish_mask, left_block)
            right_has, right_count = mask_present(fish_mask, right_block)
            left_clk, _  = mask_present(clock_mask, left_block)
            right_clk, _ = mask_present(clock_mask, right_block)

            decided_side = current_side
            if left_clk and not left_has:
                decided_side = 'left'
            elif right_clk and not right_has:
                decided_side = 'right'
            else:
                if current_side == 'left' and left_has:
                    decided_side = 'right' if not right_has else ('left' if left_count < right_count else 'right')
                elif current_side == 'right' and right_has:
                    decided_side = 'left' if not left_has else ('left' if left_count < right_count else 'right')

            now = time.time()
            if now - last_press >= PRESS_INTERVAL:
                pyautogui.press(KEY_LEFT if decided_side == 'left' else KEY_RIGHT)
                last_press, current_side = now, decided_side

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Interrumpido por usuario.")

if __name__ == "__main__":
    cfg = load_config()
    if cfg is None:
        if not is_bluestacks_active():
            print("Activa BlueStacks y vuelve a ejecutar para calibrar.")
            sys.exit(1)
        cfg = calibrate_roi_interactive()
    else:
        print(f"Configuración encontrada en {CONFIG_PATH.resolve()}")
    run_bot(cfg)

