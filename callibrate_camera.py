# calibrate_camera.py
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2 as cv

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate camera from chessboard images and output K, dist, R, T."
    )
    p.add_argument("--images", type=str, default="calib_images/chessboard_*.jpg",
                   help="Glob for input images.")
    p.add_argument("--rows", type=int, default=8,
                   help="INNER corners per column (height). Example: 6 for a 9x6 board.")
    p.add_argument("--cols", type=int, default=6,
                   help="INNER corners per row (width). Example: 9 for a 9x6 board.")
    p.add_argument("--square_size", type=float, default=25.0,
                   help="Square size in any consistent unit (e.g., millimeters).")
    p.add_argument("--preview", action="store_true",
                   help="Show corner detections and undistortion previews.")
    p.add_argument("--save_undistorted", action="store_true",
                   help="Saves an undistorted version of the first image.")
    p.add_argument("--out", type=str, default="calibration_results.npz",
                   help="Where to save calibration arrays (npz).")
    return p.parse_args()

def main():
    args = parse_args()
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise SystemExit(f"No images match: {args.images}")

    pattern_size = (args.cols, args.rows)
    # 3D points in chessboard coordinates (Z=0 plane)
    objp = np.zeros((args.rows*args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []  # 3D points in world space
    imgpoints = []  # 2D points in image plane
    img_size = None

    flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
             cv.CALIB_CB_NORMALIZE_IMAGE)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"[INFO] Finding corners in {len(image_paths)} image(s)…")
    used = 0
    for pth in image_paths:
        img = cv.imread(pth)
        if img is None:
            print(f"[WARN] Could not read {pth}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            print(f"[SKIP] {Path(pth).name}: pattern not found")
            continue

        # sub-pixel refinement
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used += 1

        if args.preview:
            vis = img.copy()
            cv.drawChessboardCorners(vis, pattern_size, corners, True)
            cv.imshow("Corners", vis)
            cv.waitKey(200)

    if used < 5:
        raise SystemExit(f"Not enough valid images ({used}) — collect more that clearly show the pattern.")

    print(f"[INFO] Running calibration with {used} valid image(s)…")
    # calibrate
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print("\n=== Intrinsics (K) ===")
    print(K)
    print("\n=== Distortion (k1, k2, p1, p2, k3[, k4…]) ===")
    print(dist.ravel())
    print(f"\nRMS reprojection error: {rms:.4f} pixels")

    # Per-image reprojection error
    per_image_err = []
    for i, (rv, tv, obj3d, img2d) in enumerate(zip(rvecs, tvecs, objpoints, imgpoints)):
        proj, _ = cv.projectPoints(obj3d, rv, tv, K, dist)
        err = cv.norm(img2d, proj, cv.NORM_L2) / np.sqrt(len(proj))
        per_image_err.append(err)
    per_image_err = np.array(per_image_err)
    print("\nPer-image reprojection errors (px):")
    for i, e in enumerate(per_image_err, 1):
        print(f"  #{i:02d}: {e:.4f}")
    print(f"Mean: {per_image_err.mean():.4f}   Std: {per_image_err.std():.4f}")

    # Convert rvecs -> rotation matrices for extrinsics
    Rs = []
    Ts = []
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv.Rodrigues(rv)
        Rs.append(R)
        Ts.append(tv.reshape(3, 1))

    # Save outputs
    np.savez(args.out,
             K=K, dist=dist, rvecs=np.array(rvecs, dtype=object),
             tvecs=np.array(tvecs, dtype=object),
             Rs=np.array(Rs, dtype=object), Ts=np.array(Ts, dtype=object),
             image_paths=np.array(image_paths, dtype=object),
             img_size=np.array(img_size),
             per_image_err=per_image_err)
    print(f"\n[SAVED] Calibration arrays -> {args.out}")

    # Optional undistortion preview
    if args.save_undistorted or args.preview:
        sample = cv.imread(image_paths[26])
        h, w = sample.shape[:2]
        newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        und = cv.undistort(sample, K, dist, None, newK)
        if args.save_undistorted:
            out_path = Path(args.out).with_suffix("")  # e.g., calibration_results
            out_img = Path(str(out_path) + "_undistorted.jpg")
            cv.imwrite(str(out_img), und)
            print(f"[SAVED] Undistorted preview -> {out_img}")
        if args.preview:
            cv.imshow("Original", sample)
            cv.imshow("Undistorted", und)
            print("[INFO] Press any key to close previews…")
            cv.waitKey(0)
            cv.destroyAllWindows()

    # Brief reminder of meanings
    print("""
Notes:
- K (intrinsics) encodes focal lengths fx, fy and principal point (cx, cy) and governs how 3D points project to the 2D image.
- dist are lens distortion coefficients; use with cv.undistort/cv.initUndistortRectifyMap.
- (R, T) for each image are the extrinsics: rotation and translation that place the chessboard in the camera's coordinate frame.
""")

if __name__ == "__main__":
    main()
