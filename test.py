import pytesseract
from PIL import Image

# Tesseract yolu
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

print("Pytesseract yüklü!")
print("Tesseract yolu:", pytesseract.pytesseract.tesseract_cmd)
