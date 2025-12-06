import cv2

img = cv2.imread("img/resim.png")   # <-- BURASI PNG OLMALI
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Orijinal", img)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(img)