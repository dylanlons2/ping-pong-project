import cv2
import numpy as np

cap = cv2.VideoCapture("data/input/output.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # jump to a frame where the ball is visible
ret, frame = cap.read()
cap.release()

blurred = cv2.GaussianBlur(frame, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

def nothing(x): pass

cv2.namedWindow("Tuner")
cv2.createTrackbar("H min", "Tuner", 0, 179, nothing)
cv2.createTrackbar("H max", "Tuner", 179, 179, nothing)
cv2.createTrackbar("S min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("S max", "Tuner", 255, 255, nothing)
cv2.createTrackbar("V min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("V max", "Tuner", 255, 255, nothing)

while True:
    lo = np.array([cv2.getTrackbarPos("H min", "Tuner"),
                   cv2.getTrackbarPos("S min", "Tuner"),
                   cv2.getTrackbarPos("V min", "Tuner")])
    hi = np.array([cv2.getTrackbarPos("H max", "Tuner"),
                   cv2.getTrackbarPos("S max", "Tuner"),
                   cv2.getTrackbarPos("V max", "Tuner")])
    mask = cv2.inRange(hsv, lo, hi)
    cv2.imshow("Tuner", np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        print(f"lower = np.array([{lo[0]}, {lo[1]}, {lo[2]}])")
        print(f"upper = np.array([{hi[0]}, {hi[1]}, {hi[2]}])")
        break

cv2.destroyAllWindows()
