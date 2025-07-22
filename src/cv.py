import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO(r"..\models\TicTacToe_250722_s.pt")
CONFIDENCE_THRESHOLD = 0.1


def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def detect_grid_contour(image):
    """
    Finds the largest 4-sided contour in the image within the expected area range.
    Returns the minAreaRect box (4 corners) as integer points.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area, max_area = 200000, 280000
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            return np.intp(box)  # 4 corner points

    raise Exception("❌ No suitable grid contour found.")


def extract_board_image(image):
    """
    Uses detected grid box (minAreaRect) to warp board to top-down square view.
    Returns a 300x300 pixel cropped image of the board.
    """
    box = detect_grid_contour(image)

    # Sort box points to consistent order: top-left, top-right, bottom-right, bottom-left
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    ordered = np.array([
        box[np.argmin(s)],
        box[np.argmin(diff)],
        box[np.argmax(s)],
        box[np.argmax(diff)]
    ], dtype="float32")

    size = 300
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped


def classify_board_yolo(board_img, model):
    """
    Run YOLO on the top-down board image and return the 9-cell board list with 'X', 'O', or ' '.
    """
    results = model(board_img, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
    symbols = [' '] * 9
    h, w = board_img.shape[:2]
    cell_w, cell_h = w // 3, h // 3

    for box in results.boxes:
        x_center, y_center = box.xywh[0][:2]
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()

        if label == 'cross':
            symbol = 'X'
        elif label == 'circle':
            symbol = 'O'
        else:
            continue  # Unknown class

        col = int(x_center / cell_w)
        row = int(y_center / cell_h)
        idx = row * 3 + col
        symbols[idx] = symbol

    return symbols


def tune_contour_grid(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Image not found.")
        return

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔍 Found {len(contours)} contours. Press SPACE to step through, ESC to exit.")

    min_area, max_area = 200000, 280000

    for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

        preview = orig.copy()
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(preview, [box], 0, (255, 0, 0), 2)
        cv2.drawContours(preview, [approx], -1, (0, 255, 0), 2)

        cv2.putText(preview, f"Contour {i} | Points: {len(approx)} | Area: {area:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Contour Preview", preview)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
