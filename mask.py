import sys
import cv2
import numpy as np
from tracking import BallTracker

tracker = BallTracker()

source = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    if isinstance(source, str):
        print(f"Cannot open video file: {source}")
    else:
        print("Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower = np.array([8, 183, 178])
    # upper = np.array([16, 213, 198])
    lower = np.array([12, 200, 222])
    upper = np.array([21, 270, 262])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_score = 0.0
    min_area = 250

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        score = circularity * np.log1p(area)
        if score > best_score:
            best_score = score
            best_contour = c

    predicted = tracker.predict()

    display = frame.copy()
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        cx, cy = x + w / 2, y + h / 2
        tracker.update(cx, cy)
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        tracker.miss()
        if tracker.is_tracking() and predicted is not None:
            px, py = int(predicted[0]), int(predicted[1])
            size = 20
            cv2.rectangle(display, (px - size, py - size),
                          (px + size, py + size), (0, 255, 255), 2)

    cv2.imshow("Tracking", display)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
