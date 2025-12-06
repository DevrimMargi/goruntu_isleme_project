import cv2

img = cv2.imread("img/resim.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (7, 7), 0)

cv2.imshow("Orijinal", gray)
cv2.imshow("Gaussian Blur", gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()
