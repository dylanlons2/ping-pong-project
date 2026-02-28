import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press SPACE to capture", frame)

    key = cv2.waitKey(1)

    if key % 256 == 32:  # Space key
        cv2.imwrite("photo.jpg", frame)
        print("Photo saved!")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print(hsv)
        break
    elif key % 256 == 27:  # ESC key
        break
    else:
        print("nothing happened")
print()
cap.release()
cv2.destroyAllWindows()
