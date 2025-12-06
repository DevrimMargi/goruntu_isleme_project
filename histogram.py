import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img/resim.png")

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.title("Color Histogram (BGR)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()
