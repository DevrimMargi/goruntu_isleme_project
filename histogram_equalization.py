import cv2
import matplotlib.pyplot as plt

# Resmi oku
img = cv2.imread("img/resim.png", 0)  # direkt grayscale olarak oku

# Histogram equalization
eq = cv2.equalizeHist(img)

# Göster
cv2.imshow("Orijinal", img)
cv2.imshow("Equalized", eq)

cv2.waitKey(0)
cv2.destroyAllWindows()
