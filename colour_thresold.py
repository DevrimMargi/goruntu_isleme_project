import cv2
import numpy as np

img = cv2.imread("img/resim.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mavi renk aralığı
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])

# Renk eşikleme
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Sonucu uygula
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Orijinal", img)
cv2.imshow("Mask", mask)          # 0 = siyah, 255 = beyaz (seçilmiş alan)
cv2.imshow("Sonuc", result)       # sadece mavi yerler görünür

cv2.waitKey(0)
cv2.destroyAllWindows()
