import cv2

img = cv2.imread("img/resim.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

cv2.imshow("Orijinal", gray)
cv2.imshow("Bilateral Filter", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
