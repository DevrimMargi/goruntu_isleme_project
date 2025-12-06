import cv2

img = cv2.imread("img/resim.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gürültüyü azaltmak için Gaussian Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Canny Edge Detection
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("Orijinal", gray)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
