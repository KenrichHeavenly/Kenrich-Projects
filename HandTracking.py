# =========================
# Two-Hand Gesture Mouse (No Lock)
# - One hand = cursor
# - Other hand = gestures (click/drag, right-click, scroll)
# OpenCV + MediaPipe + Kalman + Smooth Scroll
# =========================

# --- Quiet logs (set BEFORE importing mediapipe/tensorflow) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "3"
os.environ["absl_log_level"] = "3"
from absl import logging
logging.set_verbosity(logging.ERROR)

# --- Imports ---
import sys, time, math, threading, collections
import cv2
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# =========================
# Settings
# =========================
# Choose which hand controls what (as recognized by MediaPipe)
# Valid values: "Right" or "Left"
CURSOR_HAND  = "Right"
GESTURE_HAND = "Left"

# If the chosen hand is not visible, optionally fall back:
FALLBACK_TO_SINGLE_HAND = True   # if only one hand is seen, use it for both

# Camera (macOS → AVFoundation)
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 960, 540, 60
USE_AVFOUNDATION = (sys.platform == "darwin")

# Visualize MediaPipe landmarks (turn off for more FPS)
DRAW_LANDMARKS = False

# Gestures (scale-normalized thresholds)
PINCH_LEFT  = {"down": 0.26, "up": 0.40}   # thumb–index for left click/drag (hysteresis)
PINCH_RIGHT = {"down": 0.30, "up": 0.44}   # thumb–middle for right click

# Smooth scroll (slow + steady)
SCROLL_DEADZONE  = 0.015   # ignore tiny wobble
SCROLL_K         = 200.0   # lines/sec at dy=1.0 (lower = slower)
SCROLL_MAX_LPS   = 20.0    # cap rate
SCROLL_EMA_ALPHA = 0.14    # smoothing (lower = smoother)
SCROLL_INVERT    = False   # True to invert direction

# Temporal voting for gestures (stability)
VOTE_WIN = 6      # window size (frames)
STABLE_K = 4      # require ≥ this many "true" votes

# Kalman smoothing (cursor)
KALMAN_PROCESS_VAR = 30.0  # higher = snappier
KALMAN_MEAS_VAR    = 6.0   # higher = trusts measurement less (smoother)

# =========================
# Camera grabber (threaded)
# =========================
class FrameGrabber:
    """Background capture that always returns the freshest frame (drops old)."""
    def __init__(self, index=0, backend=None):
        self.cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # helps with USB cams
        except Exception:
            pass
        self.ok = self.cap.isOpened()
        self.frame = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.t = None

    def start(self):
        if not self.ok: return False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return True

    def _loop(self):
        while not self.stop_flag:
            ok, f = self.cap.read()
            if not ok: break
            f = cv2.flip(f, 1)  # mirrored view for comfort
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return (self.frame is not None), (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.stop_flag = True
        if self.t is not None: self.t.join(timeout=0.2)
        if self.cap: self.cap.release()

# =========================
# Kalman filter (2D, CV model)
# =========================
class Kalman2D:
    """Constant-velocity Kalman for (x,y)."""
    def __init__(self, dt=1/120, process_var=40.0, meas_var=4.0):
        self.dt = dt
        self.x = np.zeros((4,1), dtype=float)         # [x, y, vx, vy]
        self.P = np.eye(4, dtype=float) * 1000.0
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1, 0],
                           [0,0,0, 1]], dtype=float)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)
        q = process_var
        self.Q = np.array([[q,0,0,0],
                           [0,q,0,0],
                           [0,0,q,0],
                           [0,0,0,q]], dtype=float)
        r = meas_var
        self.R = np.array([[r,0],
                           [0,r]], dtype=float)

    def predict(self, dt=None):
        if dt is not None and dt > 0:
            self.F[0,2] = dt; self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z, dtype=float).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get(self):
        return float(self.x[0,0]), float(self.x[1,0])

# =========================
# Helpers
# =========================
def norm_to_screen(nx, ny, mirror=True):
    """Map normalized coords -> screen coords. Frame already mirrored, so use mirror=False."""
    if mirror: nx = 1.0 - nx
    x = int(nx * SCREEN_W)
    y = int(ny * SCREEN_H)
    return max(0, min(SCREEN_W-1, x)), max(0, min(SCREEN_H-1, y))

def v2(a,b):
    return np.array([b[0]-a[0], b[1]-a[1]], dtype=float)

def angle_deg(u,v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: return 180.0
    c = np.clip((u@v)/(nu*nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def fingers_extended_by_angle(lm_px, handed="Right"):
    """
    Angle-based finger check: a finger is 'extended' if its PIP angle is small (straighter).
    Uses joints: MCP(5/9/13/17), PIP(6/10/14/18), TIP(8/12/16/20).
    Thumb uses MCP(2), IP(3), TIP(4) with lateral check depending on handedness.
    """
    ext = [False]*5
    # Thumb
    a = lm_px[2]; b = lm_px[3]; c = lm_px[4]
    ang = angle_deg(v2(a,b), v2(b,c))
    if handed == "Right":
        lateral_ok = (lm_px[4][0] < lm_px[3][0])  # mirrored view
    else:
        lateral_ok = (lm_px[4][0] > lm_px[3][0])
    ext[0] = (ang < 42.0) and lateral_ok
    # Index..Pinky
    joints = [(5,6,8), (9,10,12), (13,14,16), (17,18,20)]
    for i,(m,p,t) in enumerate(joints, start=1):
        ang = angle_deg(v2(lm_px[m], lm_px[p]), v2(lm_px[p], lm_px[t]))
        ext[i] = ang < 35.0
    return ext  # [thumb,index,middle,ring,pinky]

def hand_scale(lm_px):
    # wrist (0) to index MCP (5)
    return math.hypot(lm_px[0][0]-lm_px[5][0], lm_px[0][1]-lm_px[5][1]) + 1e-6

# =========================
# Main
# =========================
def main():
    import mediapipe as mp  # <-- MediaPipe in use

    # Camera
    backend = cv2.CAP_AVFOUNDATION if USE_AVFOUNDATION else 0
    grab = FrameGrabber(0, backend)
    if not grab.start():
        print("Camera failed to open"); return

    # MediaPipe Hands (2 hands)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,              # 0 = fastest
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )
    draw = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    # Filters / states
    kf = Kalman2D(process_var=KALMAN_PROCESS_VAR, meas_var=KALMAN_MEAS_VAR)

    left_down = False            # mouse left button state (for drag)
    right_click_cooldown = 0.0
    prev_t = time.time()
    fps_avg = 0.0

    # Smooth scroll state
    scroll_accum = 0.0
    scroll_ema_dy = 0.0
    last_scroll_t = time.time()

    # Temporal gesture votes
    votes_pinch_left  = collections.deque(maxlen=VOTE_WIN)
    votes_pinch_right = collections.deque(maxlen=VOTE_WIN)
    votes_scroll_mode = collections.deque(maxlen=VOTE_WIN)

    while True:
        ok, frame = grab.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()
        dt = max(1e-3, now - prev_t)
        prev_t = now

        res = hands.process(rgb)

        # ---------------- map hands by handedness ----------------
        hands_dict = {}  # "Left"/"Right": { "lm_px":..., "lm":..., "scale":..., "nx":..., "ny":... }
        if res.multi_hand_landmarks:
            for idx, hand_lms in enumerate(res.multi_hand_landmarks):
                label = res.multi_handedness[idx].classification[0].label  # "Left" or "Right"
                lm_px = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms.landmark]
                sc    = hand_scale(lm_px)
                nx, ny = hand_lms.landmark[8].x, hand_lms.landmark[8].y  # index tip (normalized)
                hands_dict[label] = dict(lm_px=lm_px, lm=hand_lms, scale=sc, nx=nx, ny=ny)

        # Decide which streams to use
        use_cursor = hands_dict.get(CURSOR_HAND)
        use_gesture = hands_dict.get(GESTURE_HAND)

        # Fallbacks (if only one hand present or the selected hand missing)
        if FALLBACK_TO_SINGLE_HAND:
            if use_cursor is None and use_gesture is not None:
                use_cursor = use_gesture
            if use_gesture is None and use_cursor is not None:
                use_gesture = use_cursor

        # ---------- draw landmarks (optional) ----------
        if DRAW_LANDMARKS and res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                    styles.get_default_hand_landmarks_style(),
                                    styles.get_default_hand_connections_style())

        # --------------- CURSOR MOVE ---------------
        if use_cursor is not None:
            target_x, target_y = norm_to_screen(use_cursor["nx"], use_cursor["ny"], mirror=False)
            # Kalman
            kf.predict(dt)
            kf.update((target_x, target_y))
            cx, cy = map(int, kf.get())
            # Require index finger up on cursor hand for movement (reduces accidental motion)
            ext_c = fingers_extended_by_angle(use_cursor["lm_px"], handed=CURSOR_HAND)
            if ext_c[1]:  # index
                pyautogui.moveTo(cx, cy)

        # --------------- GESTURES ---------------
        if use_gesture is not None:
            lm_px_g = use_gesture["lm_px"]
            scale_g = use_gesture["scale"]
            nx_g, ny_g = use_gesture["nx"], use_gesture["ny"]

            # finger states for gesture hand
            ext_g = fingers_extended_by_angle(lm_px_g, handed=GESTURE_HAND)
            index_up_g, middle_up_g = ext_g[1], ext_g[2]

            # distances (normalized by hand scale)
            def dist(a,b): return math.hypot(lm_px_g[a][0]-lm_px_g[b][0], lm_px_g[a][1]-lm_px_g[b][1])
            d_ti = dist(4,8)  / scale_g  # thumb-index
            d_tm = dist(4,12) / scale_g  # thumb-middle

            # instantaneous flags
            pinch_left_now  = d_ti < (PINCH_LEFT["down"] if not left_down else PINCH_LEFT["up"])
            pinch_right_now = d_tm < PINCH_RIGHT["down"]
            scroll_now = (index_up_g and middle_up_g) and (not pinch_left_now)

            # temporal votes
            votes_pinch_left.append(1 if pinch_left_now else 0)
            votes_pinch_right.append(1 if pinch_right_now else 0)
            votes_scroll_mode.append(1 if scroll_now else 0)

            pinch_left_stable  = (sum(votes_pinch_left)  >= STABLE_K)
            pinch_right_stable = (sum(votes_pinch_right) >= STABLE_K)
            scroll_mode_stable = (sum(votes_scroll_mode) >= STABLE_K)

            # ---- LEFT CLICK / DRAG (no lock) ----
            if pinch_left_stable and not left_down:
                pyautogui.mouseDown()
                left_down = True
            elif (not pinch_left_stable) and left_down:
                pyautogui.mouseUp()
                left_down = False

            # ---- RIGHT CLICK (edge + cooldown) ----
            if pinch_right_stable and (now - right_click_cooldown > 0.28):
                pyautogui.click(button='right')
                right_click_cooldown = now

            # ---- Smooth scroll (gesture hand) ----
            if scroll_mode_stable and (not left_down):
                t_now = now
                dts = max(1e-3, t_now - last_scroll_t)
                last_scroll_t = t_now

                dy = (0.5 - ny_g)
                if SCROLL_INVERT: dy = -dy
                if abs(dy) < SCROLL_DEADZONE: dy = 0.0

                # EMA smoothing
                scroll_ema_dy = (1.0 - SCROLL_EMA_ALPHA) * scroll_ema_dy + SCROLL_EMA_ALPHA * dy

                # lines-per-second, capped
                lps = max(-SCROLL_MAX_LPS, min(SCROLL_MAX_LPS, SCROLL_K * scroll_ema_dy))
                scroll_accum += lps * dts

                # emit whole-line steps only
                while abs(scroll_accum) >= 1.0:
                    step = 1 if scroll_accum > 0 else -1
                    pyautogui.scroll(step)
                    scroll_accum -= step
            else:
                scroll_accum *= 0.85
                scroll_ema_dy *= 0.85

            # HUD (optional)
            cv2.putText(frame, f"Cursor={CURSOR_HAND if use_cursor else '—'}  Gestures={GESTURE_HAND if use_gesture else '—'}",
                        (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"Dragging={left_down}", (10, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            # If neither hand present, reset
            if left_down: 
                pyautogui.mouseUp()
                left_down = False
            votes_pinch_left.clear(); votes_pinch_right.clear(); votes_scroll_mode.clear()

        # FPS overlay
        now2 = time.time()
        fps = 1.0 / max(1e-6, (now2 - prev_t))
        fps_avg = 0.9*fps_avg + 0.1*fps if fps_avg else fps
        cv2.putText(frame, f"FPS: {fps_avg:0.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Two-Hand Gesture Mouse (Press ESC to quit this screen)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    if left_down:
        pyautogui.mouseUp()
    grab.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
