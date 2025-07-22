import cv2
import numpy as np

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],      # top-left
        pts[np.argmin(diff)],   # top-right
        pts[np.argmax(s)],      # bottom-right
        pts[np.argmax(diff)]    # bottom-left
    ], dtype="float32")

def warp_from_contour(image, contour, size=300):
    pts = order_points(np.array([pt[0] for pt in contour]))
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (size, size))

def tune_contour_grid(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Image not found.")
        return

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding for blurry grid lines
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # Dilate to close gaps in grid lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔍 Found {len(contours)} contours. Press SPACE to step through, ESC to exit.")

    # Area range for valid grid detection
    min_area = 200000
    max_area = 280000

    for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

        preview = orig.copy()

        # Draw the rotated rectangle around the contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(preview, [box], 0, (255, 0, 0), 2)  # Blue rectangle

        # Draw the simplified contour too
        cv2.drawContours(preview, [approx], -1, (0, 255, 0), 2)  # Green contour

        cv2.putText(preview, f"Contour {i} | Points: {len(approx)} | Area: {area:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Contour Preview", preview)

        # Warp and show if it's a 4-point candidate
        if len(approx) == 4:
            warped = warp_from_contour(gray, approx)
            cv2.imshow("Warped Grid Candidate", warped)

        key = cv2.waitKey(0)
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()



# Run this script directly
if __name__ == "__main__":
    tune_contour_grid("test_imgs/12.jpg")  # ⬅️ Update path to your test image
