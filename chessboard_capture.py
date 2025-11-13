# capture_chessboard.py
import argparse
import time
from pathlib import Path

import cv2 as cv

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture quality chessboard images from webcam using cv2.findChessboardCorners."
    )
    p.add_argument("--rows", type=int, default=9,
                   help="Number of INNER corners per column (e.g., 6 for a 9x6 board).")
    p.add_argument("--cols", type=int, default=7,
                   help="Number of INNER corners per row (e.g., 9 for a 9x6 board).")
    p.add_argument("--target", type=int, default=30,
                   help="How many good images to capture (30-40 recommended).")
    p.add_argument("--camera", type=int, default=0,
                   help="Webcam index (0 is the default camera).")
    p.add_argument("--outdir", type=str, default="calib_images",
                   help="Directory to save images.")
    p.add_argument("--cooldown", type=float, default=0.75,
                   help="Seconds to wait after an auto-capture to avoid near-duplicates.")
    p.add_argument("--min_hits", type=int, default=2,
                   help="Require this many consecutive frames detecting a board before saving.")
    return p.parse_args()

def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)  # (columns, rows) = (inner corners along width, height)

    # Prep output folder
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print(f"[INFO] Looking for chessboard {pattern_size} inner corners "
          f"— capturing until {args.target} images are saved.")
    print("[INFO] Controls: 'q' to quit, 's' to save current frame manually (if board found).")

    flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
             cv.CALIB_CB_NORMALIZE_IMAGE |
             cv.CALIB_CB_FAST_CHECK)

    saved = 0
    consecutive_hits = 0
    last_capture_time = 0.0

    # For corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while saved < args.target:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame.")
            break

        # Preserve a clean version to save
        save_frame = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect board
        found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
        status_text = f"Found: {found} | Saved: {saved}/{args.target}"

        if found:
            consecutive_hits += 1
            # Subpixel refine for nicer overlay (optional but helpful later)
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, pattern_size, corners, found)  # GUI only
        else:
            consecutive_hits = 0

        # Auto-capture when pattern is stable
        now = time.time()
        can_capture = (consecutive_hits >= args.min_hits) and (now - last_capture_time >= args.cooldown)

        key = cv.waitKey(1) & 0xFF

        if (can_capture) or (key == ord('s') and found):
            saved += 1
            last_capture_time = now
            filename = outdir / f"chessboard_{saved:02d}.jpg"
            cv.imwrite(str(filename), save_frame)  # SAVE CLEAN IMAGE
            print(f"[SAVE] {filename}")
            # brief visual cue in GUI
            cv.putText(frame, f"Saved {filename.name}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # HUD overlay
        cv.putText(frame, status_text, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 255), 2)
        cv.putText(frame, "Move the board around: angles, distances, lighting.",
                   (10, frame.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1)

        cv.imshow("Chessboard Capture (press 'q' to quit, 's' to save)", frame)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"[DONE] Saved {saved} images to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
