import cv2

img = cv2.imread("img/resim.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

median = cv2.medianBlur(gray, 5)  # kernel = 5

cv2.imshow("Orijinal", gray)
cv2.imshow("Median Blur", median)

cv2.waitKey(0)
cv2.destroyAllWindows()
