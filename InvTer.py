import cv2
import numpy as np

# Ukuran inner corner checkerboard (jumlah titik sudut, bukan jumlah kotak)
CHECKERBOARD = (5, 8)  # berarti 6 kotak x 9 kotak
square_size = 0.025  # ukuran sisi kotak checkerboard dalam meter (misalnya 2.5 cm)

# Kriteria corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []  # 3D point in real world
imgpoints = []  # 2D point in image plane

# Buat koordinat dunia nyata untuk checkerboard
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # kalikan dengan ukuran sisi kotak

# Buka kamera
cap = cv2.VideoCapture(0)
print("📸 Arahkan checkerboard ke kamera. Tekan 'q' untuk mulai kalibrasi.")

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, refined_corners, found)
        cv2.putText(img, f"✔️ Captured: {len(objpoints)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(img, "❌ Checkerboard Not Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Calibration', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Kalibrasi jika cukup gambar
if len(objpoints) >= 5:
    print("🧠 Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez("camera_calib.npz", mtx=mtx, dist=dist)

    # Tampilkan hasil
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n✅ Calibration complete and saved to 'camera_calib.npz'")
    print("\n=== Camera Intrinsics Matrix ===")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    print("\nCamera Matrix:")
    print(np.array_str(mtx, precision=4, suppress_small=True))

    print("\n=== Distortion Coefficients ===")
    print(dist)
else:
    print("❌ Not enough valid checkerboard captures for calibration.")
