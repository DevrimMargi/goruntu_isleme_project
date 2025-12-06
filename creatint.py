import cv2
import numpy as np

# Kamera aç
cam = cv2.VideoCapture(0)

# Mavi renk aralığı (HSV formatında)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # 1. BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Mavi rengi maskele
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3. Maske temizleme
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 4. Kontur bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # küçük noktaları alma
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "MAVI ALGILANDI", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Ekranda göster
    cv2.imshow("Kamera", frame)
    cv2.imshow("Maske", mask)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
