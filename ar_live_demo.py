# ar_live_demo.py
import cv2 as cv
import numpy as np
import time

# ---------- PARAMETERS ----------
CALIB_FILE = "calibration_results.npz"
ROWS, COLS = 8, 6          # inner corners (height, width)
SQUARE_SIZE = 25.0         # mm or your chosen unit
CAMERA_INDEX = 0

# ---------- HELPER FUNCTIONS ----------
def build_board_object_points(rows, cols, square):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square)
    return objp

def draw_axes(img, K, dist, rvec, tvec, axis_len):
    axes = np.float32([[0, 0, 0],
                       [axis_len, 0, 0],
                       [0, axis_len, 0],
                       [0, 0, -axis_len]])
    pts, _ = cv.projectPoints(axes, rvec, tvec, K, dist)
    o, x, y, z = pts.reshape(-1, 2).astype(int)
    cv.line(img, o, x, (0, 0, 255), 3)
    cv.line(img, o, y, (0, 255, 0), 3)
    cv.line(img, o, z, (255, 0, 0), 3)

def draw_cube(img, K, dist, rvec, tvec, s):
    cube = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
    ])
    pts, _ = cv.projectPoints(cube, rvec, tvec, K, dist)
    p = pts.reshape(-1, 2).astype(int)
    cv.polylines(img, [p[0:4]], True, (255, 255, 255), 2)
    cv.polylines(img, [p[4:8]], True, (255, 255, 255), 2)
    for i in range(4):
        cv.line(img, p[i], p[i + 4], (255, 255, 255), 2)

# ---------- LOAD CALIBRATION ----------
cal = np.load(CALIB_FILE, allow_pickle=True)
K, dist = cal["K"], cal["dist"]

# ---------- INITIALIZE CAMERA ----------
cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {CAMERA_INDEX}")

print("[INFO] Running AR live demo — press 's' to save frame, 'q' to quit.")

# Prepare constants
objp = build_board_object_points(ROWS, COLS, SQUARE_SIZE)
pattern_size = (COLS, ROWS)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis_len = 3 * SQUARE_SIZE
cube_size = 1.5 * SQUARE_SIZE

prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, pattern_size,
                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

    if found:
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, pattern_size, corners, found)
        ok, rvec, tvec = cv.solvePnP(objp, corners, K, dist)
        if ok:
            draw_axes(frame, K, dist, rvec, tvec, axis_len)
            draw_cube(frame, K, dist, rvec, tvec, cube_size)
            cv.putText(frame, "Chessboard detected", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv.putText(frame, "PnP failed", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv.putText(frame, "No chessboard detected", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # FPS overlay
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.imshow("AR Live Demo", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s'):
        cv.imwrite("ar_frame_live.jpg", frame)
        print("[SAVED] ar_frame_live.jpg")

cap.release()
cv.destroyAllWindows()
print("[INFO] Exited cleanly.")
